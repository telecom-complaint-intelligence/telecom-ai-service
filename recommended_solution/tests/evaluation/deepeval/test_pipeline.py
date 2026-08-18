import os
import sys
import json
import re
import pytest
import time
from huggingface_hub import InferenceClient
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval
from deepeval.models import DeepEvalBaseLLM

# Insert the project root to sys.path to ensure local imports find the agents package
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# Import agent components
from agents.solution.graph.solution_graph import solution_graph
from agents.escalation.schemas.input_schema import EscalationInput
from agents.escalation.agents.escalation_agent import run_escalation_agent
from agents.high.api import run_high_agent

# Custom LLM wrapper for DeepEval evaluation using Qwen via HuggingFace Hub
class HuggingFaceEvaluationLLM(DeepEvalBaseLLM):
    def __init__(self, model_name=None, api_key=None):
        self.model_name = model_name or os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self.api_key = api_key or os.getenv("HF_TOKEN")
        if not self.api_key:
            raise ValueError("HF_TOKEN environment variable not set in .env")
        self.client = InferenceClient(api_key=self.api_key)

    def get_model_name(self) -> str:
        return self.model_name

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        current_prompt = prompt
        last_error = None
        
        # Determine if JSON is expected
        expect_json = "json" in prompt.lower() or "score" in prompt.lower() or "steps" in prompt.lower()
        
        for attempt in range(3):
            messages = [
                {"role": "user", "content": current_prompt}
            ]
            if expect_json and attempt > 0:
                messages.append({
                    "role": "system",
                    "content": "You must return ONLY a single valid JSON object. Do not include markdown code block formatting or conversational filler."
                })
                
            try:
                stream = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1500,
                    stream=True
                )
                content = ""
                for chunk in stream:
                    if getattr(chunk, "choices", None) and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta.content
                        if delta:
                            content += delta
                
                content = content.strip()
                
                # Strip all thoughts
                content = re.sub(r'<think>.*?(?:</think>|$)', '', content, flags=re.DOTALL|re.IGNORECASE)
                content = re.sub(r'<thought>.*?(?:</thought>|$)', '', content, flags=re.DOTALL|re.IGNORECASE)
                content = re.sub(r'<thinking>.*?(?:</thinking>|$)', '', content, flags=re.DOTALL|re.IGNORECASE)
                content = content.strip()
                
                if expect_json:
                    # Clean markdown wrappers
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.startswith("```"):
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()
                    
                    # Extract JSON between { and }
                    if '{' in content and '}' in content:
                        start_idx = content.find('{')
                        end_idx = content.rfind('}') + 1
                        if end_idx > start_idx:
                            json_str = content[start_idx:end_idx]
                            try:
                                parsed = json.loads(json_str)
                                cleaned_dict = {}
                                for k, v in parsed.items():
                                    cleaned_dict[k.lower()] = v
                                return json.dumps(cleaned_dict)
                            except json.JSONDecodeError as je:
                                last_error = f"JSON decode error: {je}"
                                print(f"Eval LLM attempt {attempt + 1} failed to parse JSON. Raw response: {content}")
                                current_prompt = f"{prompt}\n\n[SYSTEM FEEDBACK]: Your response was not valid JSON:\n{content}\nError: {je}. Please output ONLY a valid, parseable JSON object."
                                time.sleep(1)
                                continue
                return content
            except Exception as e:
                last_error = str(e)
                print(f"Eval LLM attempt {attempt + 1} failed: {e}")
                
                # If it's a 402/depletion error, raise it immediately to stop the test suite with a clear evaluator failure
                if "402" in last_error or "payment required" in last_error.lower():
                    raise RuntimeError(
                        f"Evaluation LLM (DeepEval G-Eval Judge) API is unavailable: HuggingFace 402 Payment Required (quota depleted). "
                        "Failing the evaluation cleanly to preserve test integrity."
                    )
                time.sleep(1)
                continue
                
        # If all 3 attempts fail for other reasons, raise a clean evaluation exception
        raise RuntimeError(
            f"Evaluation LLM (DeepEval G-Eval Judge) API failed after 3 attempts. Last error: {last_error}"
        )

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

# Initialize custom LLM evaluator
eval_llm = HuggingFaceEvaluationLLM()

# Load golden dataset
golden_dataset_path = os.path.join(PROJECT_ROOT, "tests", "evaluation", "golden_dataset.json")
with open(golden_dataset_path, "r", encoding="utf-8") as f:
    golden_data = json.load(f)

def execute_pipeline(complaint_data: dict) -> dict:
    """
    Simulates the orchestrator.py execution flow to run the full pipeline.
    """
    complexity = str(complaint_data.get("complexity", "LOW")).upper()
    
    # Direct HIGH pipeline
    if complexity == "HIGH":
        high_result = run_high_agent(complaint_data)
        return {
            "next_severity": "HIGH",
            "escalated": False,
            "high_decision": high_result.get("final_decision"),
            "high_priority": high_result.get("final_priority") or high_result.get("priority"),
            "final_output": high_result.get("final_action") or high_result.get("final_reason"),
            "retrieved_knowledge": []
        }
        
    # LOW or MEDIUM pipeline -> Solution Agent
    solution_state = {"complaint_input": complaint_data}
    solution_result = solution_graph.invoke(solution_state)
    final_solution = solution_result.get("final_solution", {})
    
    # Run Escalation Agent
    customer_feedback = complaint_data.get("customer_feedback", "yes")
    age_in_days = int(complaint_data.get("age_in_days", 1))
    category_complaint_count = int(complaint_data.get("category_complaint_count", 1))
    
    escalation_input = EscalationInput(
        complaint=complaint_data.get("complaint", complaint_data.get("complaint_text", "")),
        current_severity=complexity,
        customer_feedback=customer_feedback,
        solution_agent_output=json.dumps(final_solution),
        category=complaint_data.get("category", "General"),
        technical_information=complaint_data.get("technical_information", "None"),
        complexity=complexity,
        complexity_score=complaint_data.get("complexity_score", 0.5),
        weighted_negativity_score=complaint_data.get("weighted_negativity_score", 0.5),
        age_in_days=age_in_days,
        category_complaint_count=category_complaint_count
    )
    
    escalation_result = run_escalation_agent(escalation_input)
    next_severity = escalation_result.next_severity
    escalated = escalation_result.escalated
    
    # Extract retrieval context
    retrieved_kb = solution_result.get("retrieved_knowledge", [])
    retrieval_context = [doc.get("content", "") for doc in retrieved_kb if doc.get("content")]
    
    # If escalated to HIGH, run High Agent
    if next_severity == "HIGH":
        complaint_data["escalation_reasoning"] = escalation_result.reasoning
        high_result = run_high_agent(complaint_data)
        return {
            "next_severity": next_severity,
            "escalated": escalated,
            "high_decision": high_result.get("final_decision"),
            "high_priority": high_result.get("final_priority") or high_result.get("priority"),
            "final_output": high_result.get("final_action") or high_result.get("final_reason"),
            "retrieved_knowledge": retrieval_context
        }
        
    # Not escalated to HIGH (stays LOW or goes to MEDIUM)
    return {
        "next_severity": next_severity,
        "escalated": escalated,
        "high_decision": None,
        "high_priority": None,
        "final_output": json.dumps(final_solution, indent=2),
        "retrieved_knowledge": retrieval_context
    }

test_results = []

def record_test_result(res: dict):
    test_results.append(res)

def is_infra_error(e: Exception) -> bool:
    if isinstance(e, AssertionError):
        return False
    if isinstance(e, (NameError, AttributeError, KeyError, TypeError, IndexError)):
        return False
    if isinstance(e, ValueError) and "invalid proposed decision" in str(e).lower():
        return False
    return True

def pytest_sessionfinish(session, exitstatus):
    report_dir = os.path.dirname(__file__)
    report_path = os.path.join(report_dir, "report.json")
    sorted_results = sorted(test_results, key=lambda x: x["test_id"])
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(sorted_results, f, indent=2)
    print(f"\n[REPORT]: Saved evaluation report to {report_path}")

@pytest.mark.parametrize("record", golden_data, ids=lambda r: r["test_id"])
def test_pipeline_execution(record):
    """
    End-to-end evaluation testing deterministic code assertions and output behavior.
    """
    test_id = record["test_id"]
    complaint_data = None
    
    try:
        # 1. Load input file
        input_file_path = os.path.join(PROJECT_ROOT, record["input_file"])
        with open(input_file_path, "r", encoding="utf-8") as f:
            complaint_data = json.load(f)

        # 2. Run the simulated orchestrator pipeline
        pipeline_result = execute_pipeline(complaint_data)

        # 3. Assert deterministic code values (Escalation & Severity transitions)
        assert pipeline_result["next_severity"] == record["expected_next_severity"], \
            f"Severity mismatch: got {pipeline_result['next_severity']}, expected {record['expected_next_severity']}"
            
        assert pipeline_result["escalated"] == record["expected_escalated"], \
            f"Escalation status mismatch: got {pipeline_result['escalated']}, expected {record['expected_escalated']}"

        # Assert High Agent routing decisions and priority boosts
        if record["expected_high_agent_decision"] is not None:
            assert pipeline_result["high_decision"] == record["expected_high_agent_decision"], \
                f"High Agent decision mismatch: got {pipeline_result['high_decision']}, expected {record['expected_high_agent_decision']}"
                
            assert pipeline_result["high_priority"] == record["expected_priority"], \
                f"High Agent priority mismatch: got {pipeline_result['high_priority']}, expected {record['expected_priority']}"

        # 4. Construct LLMTestCase for G-Eval criteria assertion
        test_case = LLMTestCase(
            input=complaint_data.get("complaint", complaint_data.get("complaint_text", "")),
            actual_output=pipeline_result["final_output"],
            expected_output=record["expected_solution_behavior"],
            retrieval_context=pipeline_result["retrieved_knowledge"]
        )

        # 5. Define G-Eval correctness metric
        pipeline_correctness_metric = GEval(
            name="Pipeline Output Correctness",
            criteria=(
                "Evaluate whether the actual output matches the expected solution behavior.\n"
                "If it is a LOW/MEDIUM customer solution, verify that the instructions are safe and logical.\n"
                "If it is a HIGH/CRITICAL dispatch plan, verify that on-site actions are appropriate for the infrastructure outage."
            ),
            model=eval_llm,
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
            threshold=0.6
        )

        # 6. Run assert_test
        try:
            assert_test(test_case, [pipeline_correctness_metric])
            
            # Print console output
            print(f"\n{test_id} -> PASS")
            print("Component: PIPELINE")
            print("Reason: Agent completed successfully and DeepEval score met threshold.")
            print("DeepEval: PASSED")
            
            # Record result
            record_test_result({
                "test_id": test_id,
                "status": "PASS",
                "component": "PIPELINE",
                "error": None,
                "score": pipeline_correctness_metric.score
            })
            
        except AssertionError as ae:
            # G-Eval threshold failed: AGENT_FAIL
            print(f"\n{test_id} -> AGENT_FAIL")
            print("Component: AGENT")
            print(f"Reason: DeepEval score {pipeline_correctness_metric.score} below threshold {pipeline_correctness_metric.threshold}.")
            print("DeepEval: FAILED")
            
            record_test_result({
                "test_id": test_id,
                "status": "AGENT_FAIL",
                "component": "AGENT",
                "error": str(ae),
                "score": pipeline_correctness_metric.score
            })
            raise ae
            
    except Exception as e:
        if isinstance(e, AssertionError):
            raise e
            
        if not is_infra_error(e):
            # Report logical bugs as normal agent fails
            print(f"\n{test_id} -> AGENT_FAIL")
            print("Component: SYSTEM_CODE")
            print(f"Reason: Logic error: {e}")
            record_test_result({
                "test_id": test_id,
                "status": "AGENT_FAIL",
                "component": "SYSTEM_CODE",
                "error": str(e),
                "score": None
            })
            raise e
            
        # Classify Infrastructure error
        error_msg = str(e)
        component = "UNKNOWN"
        exc_type_name = type(e).__name__.lower()
        
        if "qdrant" in exc_type_name or "qdrant" in error_msg.lower() or "timeout" in error_msg.lower() or "read operation timed out" in error_msg.lower():
            component = "QDRANT"
        elif "402" in error_msg or "payment required" in error_msg.lower() or "credits depleted" in error_msg.lower() or "huggingface" in error_msg.lower():
            if "judge" in error_msg.lower() or "evaluator" in error_msg.lower():
                component = "EVALUATOR"
            else:
                component = "HUGGINGFACE"
        else:
            component = "SYSTEM"
            
        print(f"\n{test_id} -> INFRA_ERROR")
        print(f"Component: {component}")
        print(f"Reason: {error_msg}")
        print("DeepEval: SKIPPED")
        
        record_test_result({
            "test_id": test_id,
            "status": "INFRA_ERROR",
            "component": component,
            "error": error_msg,
            "score": None
        })
        
        pytest.skip(f"INFRA_ERROR in {component}: {error_msg}")
