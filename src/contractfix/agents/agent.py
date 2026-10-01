"""Narrow Deep Agents explorer. No task checkout, arbitrary shell, or evaluator tools."""

from __future__ import annotations
import json
from pathlib import Path
from contextlib import contextmanager
from contractfix.config import LLMSettings
from contractfix.llm.llm_client import get_llm
from contractfix.llm.callbacks import LLMRunMetrics
from contractfix.utils.artifacts import digest


def system_prompt():
    """Keep stage instructions in a versioned prompt asset, not Python literals."""
    return (Path(__file__).parent / "prompts" / "explorer_system.j2").read_text()


def _tool_name(tool):
    if isinstance(tool, dict):
        return tool.get("name") or tool.get("function", {}).get("name")
    return getattr(tool, "name", getattr(tool, "__name__", ""))


def curated_texts():
    root = Path(__file__).resolve().parent
    texts = {"/memory/AGENTS.md": (root / "memory/AGENTS.md").read_text()}
    for p in sorted((root / "skills").glob("*/SKILL.md")):
        texts["/skills/" + p.parent.name + "/SKILL.md"] = p.read_text()
    return texts


def make_agent(model, session, checkpointer, settings: LLMSettings, *, max_model_calls=12):
    from deepagents import (
        create_deep_agent,
        FilesystemPermission,
        HarnessProfile,
        GeneralPurposeSubagentProfile,
        register_harness_profile,
    )
    from deepagents.backends import StateBackend
    from langchain.agents.middleware import wrap_model_call, wrap_tool_call
    from langchain_core.messages import ToolMessage

    # Application-owned, exact-model profiles keep this small stage from creating
    # an unrequested general-purpose subagent or hidden summarization requests.
    # These clients report Ollama or OpenAI even when ChatOpenAI uses a custom URL.
    reported_provider = settings.provider if settings.provider in {"ollama", "anthropic"} else "openai"
    register_harness_profile(
        reported_provider + ":" + settings.model,
        HarnessProfile(
            excluded_tools=frozenset({"task", "execute", "write_file", "edit_file", "write_todos"}),
            excluded_middleware=frozenset({"SummarizationMiddleware", "TodoListMiddleware"}),
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        ),
    )
    permitted = {"evaluate_clause", "submit_clause", "report_unsupported", "read_file", "ls", "glob", "grep"}

    @wrap_model_call
    def bounded_call(request, handler):
        tools = [t for t in request.tools if _tool_name(t) in permitted]
        # Defense in depth: custom model aliases/profile mismatches still cannot expose
        # or dispatch task, shell, writes, or evaluator tools through this stage.
        parts = [str(getattr(m, "content", m)) for m in request.messages]
        parts.append(str(getattr(request.system_message, "content", "")))
        parts += [json.dumps(getattr(t, "args", t), default=str) for t in tools]
        prompt_bytes = len("\n".join(parts).encode())
        if session.events:
            session.events.emit(
                "assembled_prompt",
                stage="deep_agent",
                prompt_bytes=prompt_bytes,
                tool_names=[_tool_name(t) for t in tools],
                byte_guard=settings.prompt_max_bytes,
            )
        if prompt_bytes > settings.prompt_max_bytes:
            raise RuntimeError("assembled prompt/tool-schema byte budget exceeded; reduce context")
        if not session.reserve_model_call(max_model_calls):
            raise RuntimeError("durable model call budget exhausted")
        return handler(request.override(tools=tools))

    @wrap_tool_call
    def allowed_call(request, handler):
        if request.tool_call["name"] not in permitted:
            return ToolMessage(
                content="Tool unavailable in this host-controlled stage.",
                tool_call_id=request.tool_call["id"],
            )
        return handler(request)

    return create_deep_agent(
        model=model,
        system_prompt=system_prompt(),
        tools=[session.evaluate_clause, session.submit_clause, session.report_unsupported],
        backend=StateBackend(),
        checkpointer=checkpointer,
        subagents=[],
        skills=["/skills/"],
        memory=["/memory/AGENTS.md"],
        permissions=[FilesystemPermission(operations=["write"], paths=["/**"], mode="deny")],
        middleware=[bounded_call, allowed_call],
    )


@contextmanager
def open_agent(session, settings: LLMSettings, *, model=None):
    """Keep SQLite checkpointer alive while invoking. Receipts are persisted separately."""
    from langgraph.checkpoint.sqlite import SqliteSaver
    from deepagents.backends.utils import create_file_data

    with SqliteSaver.from_conn_string(str(session.output_dir / "checkpoints.sqlite")) as checkpointer:
        texts = curated_texts()
        graph = make_agent(model or get_llm(settings=settings), session, checkpointer, settings)
        files = {path: create_file_data(content) for path, content in texts.items()}
        yield graph, files, digest(texts)


def explore(session, settings: LLMSettings, prompt: str, *, resume=False) -> dict:
    metrics = LLMRunMetrics(session.events, "deep_agent")
    with open_agent(session, settings) as (graph, files, memory_hash):
        config = {
            "configurable": {"thread_id": session.identity},
            "callbacks": [metrics],
            "recursion_limit": 30,
        }
        # Resume uses canonical host evaluation receipts. The graph history is supplementary.
        payload = {"messages": [{"role": "user", "content": prompt}]}
        if not resume:
            payload["files"] = files
        graph.invoke(payload, config=config)
        return {
            "selected": session.selected,
            "unsupported": session.state["unsupported"],
            "memory_sha256": memory_hash,
            "metrics": metrics.as_dict(),
            "assurance": "clause_exploration_only_not_end_to_end_APR",
        }
