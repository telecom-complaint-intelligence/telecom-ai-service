import os
import sys
import json
import pytest
from unittest.mock import MagicMock
import types

# Dynamic mock to prevent VertexAI import issues inside Ragas
m = types.ModuleType("langchain_community.chat_models.vertexai")
m.ChatVertexAI = MagicMock
sys.modules["langchain_community.chat_models.vertexai"] = m

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_recall, context_precision
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Optional, Any
from huggingface_hub import InferenceClient
from langchain_community.embeddings import HuggingFaceEmbeddings

# Setup path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# Custom LangChain ChatModel wrapper using HuggingFace Hub (Qwen)
class HuggingFaceChatModel(BaseChatModel):
    model_name: str = "Qwen/Qwen2.5-72B-Instruct"
    api_key: Optional[str] = None
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        api_messages = []
        for msg in messages:
            if msg.type == "human":
                role = "user"
            elif msg.type == "system":
                role = "system"
            else:
                role = "assistant"
            api_messages.append({"role": role, "content": msg.content})
            
        client = InferenceClient(api_key=self.api_key or os.getenv("HF_TOKEN"))
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=api_messages,
                max_tokens=1024,
                temperature=0.0
            )
            content = response.choices[0].message.content
        except Exception as e:
            err_msg = str(e)
            if "402" in err_msg or "payment required" in err_msg.lower():
                raise RuntimeError(
                    "Ragas LLM evaluator API is unavailable: HuggingFace 402 Payment Required (quota depleted)."
                ) from e
            raise e
            
        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self) -> str:
        return "huggingface_hub"

def test_ragas_retrieval_quality():
    # Load expected context from golden_dataset.json
    golden_path = os.path.join(PROJECT_ROOT, "tests", "evaluation", "golden_dataset.json")
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_dataset = json.load(f)
        
    # Setup LLM and Embeddings wrappers
    hf_token = os.getenv("HF_TOKEN")
    llm = HuggingFaceChatModel(api_key=hf_token)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Wrap for RAGAS compatibility if needed
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        llm_arg = LangchainLLMWrapper(llm)
        embeddings_arg = LangchainEmbeddingsWrapper(embeddings)
    except ImportError:
        llm_arg = llm
        embeddings_arg = embeddings
        
    # Initialize retriever
    from agents.solution.knowledge.retriever import VectorKnowledgeRetriever
    retriever = VectorKnowledgeRetriever()
    
    # Scroll existing IDs from Qdrant Cloud to verify expected IDs coverage
    try:
        all_points = retriever.client.scroll(collection_name=retriever.collection_name, limit=100)[0]
        existing_ids = {p.payload.get("id") for p in all_points if p.payload.get("id")}
    except Exception as e:
        print(f"\nWarning: Failed to fetch Qdrant collection points: {e}")
        existing_ids = set()

    # Bypassed High-Agent case IDs
    high_agent_bypasses = {"EVAL-004", "EVAL-010", "EVAL-011", "EVAL-014"}
    
    results_summary = []
    eval_cases = []
    
    questions = []
    contexts_list = []
    ground_truths = []
    case_indices = []
    
    # Iterate through all records
    for record in golden_dataset:
        test_id = record["test_id"]
        
        if test_id in high_agent_bypasses:
            # Skip Solution Agent retrieval execution for High-Agent bypasses
            continue
            
        # 1. Load inputs dynamically
        input_file = record["input_file"]
        input_file_path = os.path.join(PROJECT_ROOT, input_file)
        with open(input_file_path, "r", encoding="utf-8") as f:
            complaint_data = json.load(f)
            
        expected_kb = record["expected_relevant_knowledge"]
        reference_context = "\n".join([doc["content"] for doc in expected_kb])
        
        # Extract same retriever arguments as Solution Agent retrieve_knowledge node
        question = complaint_data.get("complaint", complaint_data.get("complaint_text", ""))
        category = complaint_data.get("category", "")
        tech_info = complaint_data.get("technical_information", {})
        
        components = tech_info.get("component", [])
        component = components[0] if components else ""
        
        failure_types = tech_info.get("failure_type", [])
        failure_type = failure_types[0] if failure_types else ""
        
        # 2. Retrieve actual top-3 documents using Qdrant Cloud retriever
        docs = retriever.search(
            domain=category,
            component=component,
            failure_type=failure_type,
            complaint_text=question,
            top_k=3
        )
        
        # Retrieved KB IDs & Titles
        retrieved_ids = [doc.get("id") or doc.get("knowledge_id") or "UNKNOWN_ID" for doc in docs]
        retrieved_titles = [doc.get("title") or "No Title" for doc in docs]
        retrieved_contents = [doc.get("content", "") for doc in docs if doc.get("content")]
        
        # Expected KB IDs & Titles
        expected_ids = [doc["knowledge_id"] for doc in expected_kb]
        expected_titles = [doc["title"] for doc in expected_kb]
        
        # Determine if expected knowledge exists in the Qdrant Cloud collection
        has_coverage = all(eid in existing_ids for eid in expected_ids)
        
        # Add to batch evaluation list
        case_idx = len(eval_cases)
        eval_cases.append({
            "test_id": test_id,
            "complaint": question,
            "expected_ids": expected_ids,
            "expected_titles": expected_titles,
            "retrieved_ids": retrieved_ids,
            "retrieved_titles": retrieved_titles,
            "has_coverage": has_coverage,
            "status": None,
            "context_precision": 0.0,
            "context_recall": 0.0
        })
        
        questions.append(question)
        contexts_list.append(retrieved_contents)
        ground_truths.append(reference_context)
        case_indices.append(case_idx)

    # 3. Batch RAGAS Evaluation
    if questions:
        dataset_dict = {
            "question": questions,
            "contexts": contexts_list,
            "ground_truth": ground_truths
        }
        dataset = Dataset.from_dict(dataset_dict)
        
        print(f"\nRunning batch RAGAS evaluation for {len(questions)} cases...")
        try:
            ragas_result = evaluate(
                dataset,
                metrics=[context_precision, context_recall],
                llm=llm_arg,
                embeddings=embeddings_arg
            )
            
            scores_list = ragas_result.scores
            for row_idx, case_idx in enumerate(case_indices):
                if isinstance(scores_list, list) and row_idx < len(scores_list):
                    score_dict = scores_list[row_idx]
                elif isinstance(scores_list, dict):
                    score_dict = scores_list
                else:
                    score_dict = {}
                
                precision = score_dict.get("context_precision", 0.0)
                recall = score_dict.get("context_recall", 0.0)
                
                eval_cases[case_idx]["context_precision"] = precision
                eval_cases[case_idx]["context_recall"] = recall
        except Exception as eval_exc:
            print(f"RAGAS evaluation failed: {eval_exc}. Defaulting scores to 0.0.")

    # 4. Print and Save output
    print("\n==================================================")
    print("DETAILED RAGAS EVALUATION RESULTS")
    print("==================================================")
    
    for record in golden_dataset:
        test_id = record["test_id"]
        
        if test_id in high_agent_bypasses:
            # Report bypass case
            expected_kb = record["expected_relevant_knowledge"]
            expected_ids = [doc["knowledge_id"] for doc in expected_kb]
            expected_titles = [doc["title"] for doc in expected_kb]
            
            print(f"\n{test_id}")
            print("--------------------------------------------------")
            print("Status: HIGH_AGENT_BYPASS")
            print("Retrieved KB IDs/Titles:")
            print("  - [Bypassed Solution Agent - processed directly by High Agent]")
            print("Expected KB IDs/Titles:")
            for eid, etitle in zip(expected_ids, expected_titles):
                print(f"  - {eid}: {etitle}")
            print("==================================================")
            
            results_summary.append({
                "test_id": test_id,
                "status": "HIGH_AGENT_BYPASS",
                "retrieved_kb_ids": [],
                "expected_kb_ids": expected_ids,
                "context_precision": None,
                "context_recall": None
            })
            continue
            
        # Find matching evaluated case
        case = next(c for c in eval_cases if c["test_id"] == test_id)
        
        # Classify case
        if not case["has_coverage"]:
            status = "KB_COVERAGE"
        elif case["context_recall"] >= 0.5:
            status = "GOOD"
        else:
            status = "RETRIEVER"
            
        case["status"] = status
        
        # Clean any float NaN to None to ensure standard JSON compliance
        import math
        precision_val = case["context_precision"]
        recall_val = case["context_recall"]
        
        try:
            if math.isnan(precision_val):
                precision_val = None
        except TypeError:
            pass
            
        try:
            if math.isnan(recall_val):
                recall_val = None
        except TypeError:
            pass
        
        print(f"\n{case['test_id']}")
        print("--------------------------------------------------")
        print(f"Status: {status}")
        print("Complaint:")
        print(case["complaint"])
        print("\nRetrieved KB IDs/Titles:")
        for rid, rtitle in zip(case["retrieved_ids"], case["retrieved_titles"]):
            print(f"  - {rid}: {rtitle}")
        print("\nExpected KB IDs/Titles:")
        for eid, etitle in zip(case["expected_ids"], case["expected_titles"]):
            print(f"  - {eid}: {etitle}")
        print(f"\nContext relevance (precision) -> {precision_val if precision_val is not None else 0.0:.4f}")
        print(f"Context recall                -> {recall_val if recall_val is not None else 0.0:.4f}")
        print("==================================================")
        
        results_summary.append({
            "test_id": test_id,
            "status": status,
            "retrieved_kb_ids": case["retrieved_ids"],
            "expected_kb_ids": case["expected_ids"],
            "context_precision": precision_val,
            "context_recall": recall_val
        })
        
    # Save a machine-readable summary
    results_path = os.path.join(PROJECT_ROOT, "tests", "evaluation", "ragas", "ragas_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
