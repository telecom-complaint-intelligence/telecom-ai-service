import os
import re
import sys
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

load_dotenv(override="pytest" not in sys.modules)


class MockChatLLM(BaseChatModel):

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt_text = ""
        for m in messages:
            prompt_text += f"\n{m.content}"
        relevance_evaluation = (
            "The proposed solution is relevant to the user's issue."
        )
        resolution_status = "unresolved"
        solution_match = re.search(
            r"Solution Proposed by Solution Agent:\s*(.*)",
            prompt_text,
            re.IGNORECASE,
        )
        complaint_match = re.search(
            r"Original Complaint:\s*(.*)", prompt_text, re.IGNORECASE
        )
        proposed_sol = (
            solution_match.group(1).lower() if solution_match else ""
        )
        complaint_txt = (
            complaint_match.group(1).lower() if complaint_match else ""
        )
        if "irrelevant" in proposed_sol or "irrelevant" in complaint_txt:
            relevance_evaluation = (
                "The proposed solution is completely irrelevant to the"
                " complaint."
            )
            resolution_status = "unresolved"
        elif ("billing" in complaint_txt or "refund" in complaint_txt) and (
            "reboot" in proposed_sol or "restart" in proposed_sol
        ):
            relevance_evaluation = (
                "The proposed solution (rebooting) is irrelevant to the"
                " billing/refund complaint."
            )
            resolution_status = "unresolved"
        feedback_match = re.search(
            r"Customer Feedback:\s*(.*)", prompt_text, re.IGNORECASE
        )
        customer_resp = (
            feedback_match.group(1).lower() if feedback_match else ""
        )

        def has_word(word: str, text: str) -> bool:
            return bool(re.search(rf"\b{word}\b", text, re.IGNORECASE))

        has_work = (
            has_word("work", customer_resp)
            or has_word("worked", customer_resp)
            or has_word("works", customer_resp)
            or "true" in customer_resp
        )
        has_solved = (
            has_word("solved", customer_resp)
            or has_word("solve", customer_resp)
            or has_word("fixed", customer_resp)
            or has_word("thanks", customer_resp)
            or has_word("resolved", customer_resp)
        )
        has_unresolved_indicators = (
            has_word("does not", customer_resp)
            or has_word("doesn't", customer_resp)
            or has_word("didn't", customer_resp)
            or has_word("failed", customer_resp)
            or has_word("still", customer_resp)
            or has_word("fail", customer_resp)
            or "false" in customer_resp
            or "red light" in customer_resp
        )
        has_not_tried = (
            has_word("haven't", customer_resp)
            or has_word("not yet", customer_resp)
            or has_word("didn't try", customer_resp)
            or has_word("not attempted", customer_resp)
        )
        has_partial = (
            has_word("partially", customer_resp)
            or has_word("half", customer_resp)
            or has_word("some", customer_resp)
        )
        has_unknown_kw = has_word("unknown", customer_resp) or has_word(
            "vague", customer_resp
        )

        if has_unresolved_indicators:
            resolution_status = "unresolved"
        elif has_work or has_solved:
            resolution_status = "resolved"
        elif has_not_tried:
            resolution_status = "not_attempted"
        elif has_partial:
            resolution_status = "partially_resolved"
        elif has_unknown_kw or len(customer_resp.strip()) < 3 or (
            "still" in customer_resp
            or "offline" in customer_resp
            or "crash" in customer_resp
            or "fail" in customer_resp
            or "broken" in customer_resp
        ):
            resolution_status = "unresolved"
        else:
            resolution_status = "unresolved"

        if "irrelevant" in relevance_evaluation:
            resolution_status = "unresolved"

        json_content = (
            f"{{\n"
            f'  "relevance_evaluation": "{relevance_evaluation}",\n'
            f'  "resolution_status": "{resolution_status}",\n'
            '  "reasoning": "Decision policy analysis: Customer feedback='
            f'{customer_resp}, Resolution={resolution_status}."\n'
            f"}}"
        )
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=json_content))]
        )

    @property
    def _llm_type(self) -> str:
        return "mock-chat-llm"


def get_llm():
    hf_token = os.getenv("HF_TOKEN", "")
    model_name = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    is_mock = (
        not hf_token
        or hf_token.strip() == ""
        or hf_token.strip() == "your_new_token_here"
    )

    if not is_mock:
        try:
            from langchain_huggingface import (
                ChatHuggingFace,
                HuggingFaceEndpoint,
            )

            llm = HuggingFaceEndpoint(
                repo_id=model_name,
                huggingfacehub_api_token=hf_token,
                task="text-generation",
                max_new_tokens=512,
                temperature=0.1,
            )
            return ChatHuggingFace(llm=llm)
        except Exception:
            pass

    return MockChatLLM()
