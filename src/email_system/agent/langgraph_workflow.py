from __future__ import annotations

import time
from typing import Any, Callable, TypedDict

from email_system.mcp import MailMCPClient, NoopMailMCPClient
from email_system.memory import InMemoryLongTermMemory, MemoryRecord, ShortTermMemory
from email_system.memory.long_term import LongTermMemory
from email_system.models import LLMClient
from email_system.schemas import ActionItem, AgentOutput, Confidence, Email, Entities
from email_system.skills import ClassifyEmailSkill, DraftReplySkill, ExtractActionItemsSkill, SummarizeEmailSkill


class EmailGraphState(TypedDict, total=False):
    email: Email
    context: dict[str, Any]
    outputs: dict[str, Any]
    timings_ms: dict[str, float]
    workflow_trace: list[dict[str, Any]]
    graph_backend: str


class EmailAgentWorkflow:
    """LangGraph-style automatic email processing workflow."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        short_term_memory: ShortTermMemory | None = None,
        long_term_memory: LongTermMemory | None = None,
        mail_client: MailMCPClient | None = None,
    ) -> None:
        self.llm = llm
        self.short_term_memory = short_term_memory or ShortTermMemory()
        self.long_term_memory = long_term_memory if long_term_memory is not None else InMemoryLongTermMemory()
        self.mail_client = mail_client or NoopMailMCPClient()
        self.classifier = ClassifyEmailSkill()
        self.summarizer = SummarizeEmailSkill()
        self.action_extractor = ExtractActionItemsSkill()
        self.reply_drafter = DraftReplySkill()
        self.app, self.graph_backend = self._compile_graph()

    def run(self, email: Email) -> AgentOutput:
        initial_state: EmailGraphState = {
            "email": email,
            "context": {},
            "outputs": {},
            "timings_ms": {},
            "workflow_trace": [],
            "graph_backend": self.graph_backend,
        }
        final_state = self.app.invoke(initial_state)
        return self._build_output(final_state)

    def _compile_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            return _FallbackGraphApp(self), "fallback"

        graph = StateGraph(EmailGraphState)
        graph.add_node("read_email", self._node_read_email)
        graph.add_node("classify_intent", self._node_classify_intent)
        graph.add_node("bug_tracking", self._node_bug_tracking)
        graph.add_node("search_documentation", self._node_search_documentation)
        graph.add_node("write_response", self._node_write_response)
        graph.add_node("human_review", self._node_human_review)
        graph.add_node("send_reply", self._node_send_reply)

        graph.add_edge(START, "read_email")
        graph.add_edge("read_email", "classify_intent")
        graph.add_conditional_edges(
            "classify_intent",
            self._route_after_classify,
            {"bug_tracking": "bug_tracking", "search_documentation": "search_documentation"},
        )
        graph.add_edge("bug_tracking", "write_response")
        graph.add_edge("search_documentation", "write_response")
        graph.add_edge("write_response", "human_review")
        graph.add_edge("human_review", "send_reply")
        graph.add_edge("send_reply", END)
        return graph.compile(), "langgraph"

    def _node_read_email(self, state: EmailGraphState) -> EmailGraphState:
        return self._timed("read_email", state, self._read_email)

    def _node_classify_intent(self, state: EmailGraphState) -> EmailGraphState:
        return self._timed("classify_intent", state, self._classify_intent)

    def _node_bug_tracking(self, state: EmailGraphState) -> EmailGraphState:
        return self._timed("bug_tracking", state, self._bug_tracking)

    def _node_search_documentation(self, state: EmailGraphState) -> EmailGraphState:
        return self._timed("search_documentation", state, self._search_documentation)

    def _node_write_response(self, state: EmailGraphState) -> EmailGraphState:
        return self._timed("write_response", state, self._write_response)

    def _node_human_review(self, state: EmailGraphState) -> EmailGraphState:
        return self._timed("human_review", state, self._human_review)

    def _node_send_reply(self, state: EmailGraphState) -> EmailGraphState:
        return self._timed("send_reply", state, self._send_reply)

    def _read_email(self, state: EmailGraphState) -> None:
        email = state["email"]
        short_term = self.short_term_memory.load(email)
        long_term = [record.to_dict() for record in self.long_term_memory.search(email)]
        mcp_result = self.mail_client.read_email(email).to_dict()
        state["context"]["memory"] = {"short_term": short_term, "long_term": long_term}
        state["context"]["mail_mcp"] = {"read_email": mcp_result}
        state["outputs"]["read_email"] = {"memory": state["context"]["memory"], "mail_mcp": mcp_result}

    def _classify_intent(self, state: EmailGraphState) -> None:
        output = self.classifier.run(state["email"], state["context"], self.llm)
        state["outputs"]["classify_intent"] = output
        state["context"]["classify_intent"] = output

    def _route_after_classify(self, state: EmailGraphState) -> str:
        category = state.get("outputs", {}).get("classify_intent", {}).get("category", "other")
        if category in {"support", "bug", "bug_tracking"}:
            return "bug_tracking"
        return "search_documentation"

    def _bug_tracking(self, state: EmailGraphState) -> None:
        result = self.mail_client.bug_tracking(state["email"], state["context"]).to_dict()
        state["context"]["route"] = "bug_tracking"
        state["context"]["mail_mcp"]["bug_tracking"] = result
        state["outputs"]["bug_tracking"] = result

    def _search_documentation(self, state: EmailGraphState) -> None:
        result = self.mail_client.search_documentation(state["email"], state["context"]).to_dict()
        state["context"]["route"] = "search_documentation"
        state["context"]["mail_mcp"]["search_documentation"] = result
        state["outputs"]["search_documentation"] = result

    def _write_response(self, state: EmailGraphState) -> None:
        summary = self.summarizer.run(state["email"], state["context"], self.llm)
        actions = self.action_extractor.run(state["email"], state["context"], self.llm)
        state["context"]["summarize_email"] = summary
        state["context"]["extract_action_items"] = actions
        reply = self.reply_drafter.run(state["email"], state["context"], self.llm)
        state["outputs"]["summarize_email"] = summary
        state["outputs"]["extract_action_items"] = actions
        state["outputs"]["draft_reply"] = reply
        state["outputs"]["write_response"] = {
            "summary": summary.get("summary", ""),
            "action_items": actions.get("action_items", []),
            "reply_draft": reply.get("reply_draft", ""),
        }

    def _human_review(self, state: EmailGraphState) -> None:
        classify = state["outputs"].get("classify_intent", {})
        priority = classify.get("priority", "normal")
        skill_errors = {
            name: str(output["parse_error"])
            for name, output in state["outputs"].items()
            if isinstance(output, dict) and output.get("parse_error")
        }
        output = {
            "requires_human_review": priority in {"high", "urgent"} or bool(skill_errors),
            "skill_errors": skill_errors,
        }
        state["context"]["human_review"] = output
        state["outputs"]["human_review"] = output

    def _send_reply(self, state: EmailGraphState) -> None:
        reply_text = state["outputs"].get("draft_reply", {}).get("reply_draft", "")
        result = self.mail_client.send_reply(state["email"], reply_text, state["context"]).to_dict()
        state["context"]["mail_mcp"]["send_reply"] = result
        state["outputs"]["send_reply"] = result
        self._save_memory(state)

    def _save_memory(self, state: EmailGraphState) -> None:
        output = self._output_projection(state)
        self.short_term_memory.save(state["email"], output)
        self.long_term_memory.save(MemoryRecord.from_agent_output(state["email"], output))

    def _timed(self, node: str, state: EmailGraphState, fn: Callable[[EmailGraphState], None]) -> EmailGraphState:
        start = time.perf_counter()
        status = "ok"
        details: dict[str, Any] = {}
        try:
            fn(state)
        except Exception as exc:
            status = "error"
            details = {"error": str(exc)}
            raise
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            state["timings_ms"][node] = latency_ms
            state["workflow_trace"].append({"node": node, "status": status, "latency_ms": latency_ms, "details": details})
        return state

    def _output_projection(self, state: EmailGraphState) -> dict[str, Any]:
        classify = state["outputs"].get("classify_intent", {})
        summary = state["outputs"].get("summarize_email", {})
        actions = state["outputs"].get("extract_action_items", {})
        review = state["outputs"].get("human_review", {})
        return {
            "category": classify.get("category", "other"),
            "priority": classify.get("priority", "normal"),
            "summary": summary.get("summary", ""),
            "action_items": actions.get("action_items", []),
            "requires_human_review": review.get("requires_human_review", False),
        }

    def _build_output(self, state: EmailGraphState) -> AgentOutput:
        classify = state["outputs"].get("classify_intent", {})
        summary = state["outputs"].get("summarize_email", {})
        actions = state["outputs"].get("extract_action_items", {})
        reply = state["outputs"].get("draft_reply", {})
        review = state["outputs"].get("human_review", {})
        priority = classify.get("priority", "normal")
        return AgentOutput(
            email_id=state["email"].email_id,
            category=classify.get("category", "other"),
            priority=priority,
            summary=summary.get("summary", ""),
            action_items=[ActionItem(**item) for item in actions.get("action_items", [])],
            entities=Entities(),
            reply_draft=reply.get("reply_draft", ""),
            confidence=Confidence(category=float(classify.get("confidence", 0.0)), summary=float(summary.get("confidence", 0.0))),
            requires_human_review=review.get("requires_human_review", priority in {"high", "urgent"}),
            timings_ms=state["timings_ms"],
            skill_errors=review.get("skill_errors", {}),
            memory=state["outputs"].get("read_email", {}).get("memory", {}),
            workflow_trace=state["workflow_trace"],
            model_backend=type(self.llm).__name__,
            graph_backend=state.get("graph_backend", self.graph_backend),
            route=state.get("context", {}).get("route", ""),
            delivery_status=state["outputs"].get("send_reply", {}).get("status", ""),
        )


class _FallbackGraphApp:
    def __init__(self, workflow: EmailAgentWorkflow) -> None:
        self.workflow = workflow

    def invoke(self, state: EmailGraphState) -> EmailGraphState:
        state = self.workflow._node_read_email(state)
        state = self.workflow._node_classify_intent(state)
        route = self.workflow._route_after_classify(state)
        if route == "bug_tracking":
            state = self.workflow._node_bug_tracking(state)
        else:
            state = self.workflow._node_search_documentation(state)
        state = self.workflow._node_write_response(state)
        state = self.workflow._node_human_review(state)
        state = self.workflow._node_send_reply(state)
        return state
