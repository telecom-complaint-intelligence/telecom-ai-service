# Golden Evaluation Dataset Documentation

This directory contains the golden evaluation dataset used to validate the CTS-genC telecom complaint multi-agent pipeline. The dataset is designed to be **completely framework and tool-agnostic**, serving as a core schema that can be easily mapped to specialized evaluation frameworks such as **DeepEval** or **RAGAS** in the future.

---

## 1. Purpose of the Golden Dataset

The golden evaluation dataset defines the **expected (reference) outcomes** for the agents in the pipeline. Rather than just repeating or housing the raw input complaints, this dataset defines the ground-truth behaviors, correct escalation severities, historical knowledge linkages, and operational decisions that a correctly functioning pipeline must output.

By separating the inputs from the expected outcomes, we can run automated evaluations to assert:
* **Solution Agent Accuracy**: Whether the synthesized instructions contain appropriate, safe customer troubleshooting details.
* **Escalation Agent Compliance**: Whether the escalation engine correctly applies deterministic policies and transitions severities under specified conditions.
* **High Agent Decision Rigor**: Whether the multi-agent high-priority diagnosis matches historical recommendations and SLA directives.

---

## 2. Dataset Immutability & Input Relationships

> [!IMPORTANT]
> **Constraint Checklist**:
> 1. The existing 14 input test cases (`tests/inputs/eval_001.json` through `tests/inputs/eval_014.json`) **must remain completely unchanged**. They are the sole source of input complaint data.
> 2. No new input test cases are defined.
> 3. `tests/evaluation/golden_dataset.json` maps 1-to-1 to each of these input files via the `input_file` field.

```mermaid
graph LR
    subgraph Input Files (Immutable)
        I1[eval_001.json]
        I2[eval_002.json]
        I3[eval_014.json]
    end
    subgraph Golden Dataset
        G1[EVAL-001 Record]
        G2[EVAL-002 Record]
        G3[EVAL-014 Record]
    end
    I1 -->|"corresponds to"| G1
    I2 -->|"corresponds to"| G2
    I3 -->|"corresponds to"| G3
```

---

## 3. Field Schema Definitions

Each record in `golden_dataset.json` contains the following structured fields:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `test_id` | `string` | Unique identifier matching `EVAL-001` through `EVAL-014`. |
| `input_file` | `string` | The exact file path to the corresponding immutable input JSON (e.g., `tests/inputs/eval_001.json`). |
| `expected_solution_behavior` | `string` | Detailed description of what the **Solution Agent** must correctly understand, analyze, and synthesize in its output instructions. |
| `expected_relevant_knowledge` | `array` | A list of structured objects representing relevant knowledge base articles or historical complaints. Each object contains `knowledge_id`, `title`, and `content`. |
| `expected_next_severity` | `string` | Mapped severity output of the Escalation Agent deterministic policy: `LOW`, `MEDIUM`, or `HIGH`. |
| `expected_escalated` | `boolean` | True if the Escalation Agent should transition the severity of the ticket (LOW -> MEDIUM, LOW -> HIGH, or MEDIUM -> HIGH); False otherwise. |
| `expected_high_agent_decision` | `string` or `null` | Prototypical action decision expected from the **High Agent** (e.g., `ESCALATE`, `CRITICAL`, `CRITICAL_DISPATCH`). Set to `null` if the High Agent is not involved. |
| `expected_priority` | `string` or `null` | Expected priority level assigned by the High Agent (e.g., `HIGH`, `CRITICAL`). Set to `null` if the High Agent is not involved. |
| `evaluation_notes` | `string` | Operational notes outlining critical safety rules, expected deterministic triggers (such as SLA age breaches or chronic recurrence), and verification focus areas. |

---

## 4. Usage for Automated Evaluation

When setting up a test harness or pipeline executor, you can iterate over `golden_dataset.json` to drive evaluations automatically:

1. **Load a golden record**: Read the corresponding `input_file` and parse the raw payload.
2. **Execute the Orchestrator Pipeline**: Pass the payload through the orchestrator.
3. **Capture Agent Outputs**: Capture the generated outputs at the agent boundaries:
   - Solution Agent output `final_solution`
   - Escalation Agent output `escalation_result`
   - High Agent output `high_result`
4. **Assert Deterministic Fields**: Assert exact matches for:
   - `next_severity == expected_next_severity`
   - `escalated == expected_escalated`
   - `priority == expected_priority` (if High Agent was run)
   - `final_decision == expected_high_agent_decision` (if High Agent was run)
5. **Evaluate LLM Syntheses (LLM-as-a-Judge)**: Pass `expected_solution_behavior` and the generated instructions to an LLM evaluator to score semantic accuracy and safety compliance.

---

## 5. Adaptation for DeepEval

To use this dataset in **DeepEval**, write a script to map each golden record to DeepEval's test case formats:

### Solution Agent Test Script (using G-Eval Correctness)
We have implemented a ready-to-run DeepEval evaluation test suite:
* [`test_solution_agent.py`](file:///c:/Users/pooja/Desktop/agent/recommended_solution/tests/evaluation/deepeval/test_solution_agent.py): A parameterized test suite that runs the Solution Agent against all LOW/MEDIUM cases in the golden dataset and evaluates output correctness and safety using a custom LLM-as-a-judge (Hugging Face Qwen model).

To run the entire test suite:
```bash
python -m pytest tests/evaluation/deepeval/test_solution_agent.py
```

To run a single test case (e.g., `EVAL-001`):
```bash
python -m pytest tests/evaluation/deepeval/test_solution_agent.py -k "EVAL-001"
```

### Mapping Golden Fields into DeepEval's `LLMTestCase`:
```python
from deepeval.test_case import LLMTestCase

# For LOW/MEDIUM solution verification
test_case = LLMTestCase(
    input=complaint_input_text,                  # Loaded from input_file
    actual_output=agent_customer_instructions,  # From final_solution["customer_instructions"]
    expected_output=golden_record["expected_solution_behavior"] # Reference expectation
)
```

### Escalation or High Agent Evaluation
For asserting policy decisions, map to standard unit tests or custom LLM-based policy metrics in DeepEval, using `golden_next_severity` or `expected_high_agent_decision` as the reference value.

---

## 6. Adaptation for RAGAS

To validate the retrieval system (context recall/precision) and output faithfulness using **RAGAS**, map `expected_relevant_knowledge` to the context lists:

```python
# Format dataset for RAGAS evaluation
ragas_sample = {
    "user_input": complaint_input_text,
    "retrieved_contexts": [doc["content"] for doc in agent_retrieved_knowledge],
    "response": agent_customer_instructions,
    "reference": golden_record["expected_solution_behavior"],
    "reference_contexts": [item["content"] for item in golden_record["expected_relevant_knowledge"]]
}
```

### RAGAS Metrics Mapping:
* **Context Recall**: Compares `retrieved_contexts` (retrieved by the Solution Agent) against `reference_contexts` (loaded from `expected_relevant_knowledge`) to verify if the retriever successfully found the target ground truth.
* **Context Relevancy**: Checks if the chunks in `retrieved_contexts` are concise and relevant to the `user_input`.
* **Faithfulness**: Compares the generated `response` against `retrieved_contexts` to ensure no hallucinations were introduced.
* **Answer Relevance**: Checks how well the generated `response` answers the original `user_input`.
