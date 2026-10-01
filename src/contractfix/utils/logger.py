"""Rich terminal presentation; JSONL evidence lives in ``utils.artifacts``."""

from __future__ import annotations

import logging
import json
import os
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Callable, Generator

from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table

from contractfix.utils.artifacts import redact


class ContractFixLogger:
    """Small application logger with redacted terminal output."""

    def __init__(self, console: Console | None = None) -> None:
        plain = os.getenv("CONTRACTFIX_LOG_STYLE", "rich").lower() == "plain"
        self.console = console or Console(stderr=True, force_terminal=False if plain else None,
                                          color_system=None if plain else "auto")
        self.transcript: Console | None = None

    def attach(self, root: str | Path) -> None:
        """Start one ANSI-free, human-readable transcript for a run."""
        path = Path(root) / "run.log"
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.transcript is not None and hasattr(self.transcript.file, "close"):
            self.transcript.file.close()
        self.transcript = Console(
            file=path.open("a", encoding="utf-8"),
            force_terminal=False,
            color_system=None,
            width=120,
        )

    def _render(self, renderable: object) -> None:
        self.console.print(renderable)
        if self.transcript is not None:
            self.transcript.print(renderable)

    def banner(self, title: str, subtitle: str = "") -> None:
        body = Markdown(redact(subtitle)) if subtitle else ""
        self._render(Panel(body, title=redact(title), title_align="left", border_style="bold magenta"))

    def panel(self, title: str, body: str, *, style: str = "cyan") -> None:
        self._render(Panel(Markdown(redact(body)), title=redact(title),
                           title_align="left", border_style=style))

    def code(self, title: str, source: str, *, lexer: str = "python") -> None:
        self._render(Panel(Syntax(redact(source), lexer, word_wrap=True), title=redact(title),
                           title_align="left", border_style="green"))

    def table(self, title: str, columns: list[str], rows: list[list[object]]) -> None:
        table = Table(title=redact(title), header_style="bold cyan")
        for column in columns:
            table.add_column(redact(column))
        for row in rows:
            table.add_row(*(redact(str(value)) for value in row))
        self._render(table)

    def conformance_review(
        self,
        candidate_id: str,
        *,
        phase: str,
        back_translation: str,
        missing_meaning: list[str],
        added_meaning: list[str],
        semantic_duplicates: list[str],
        reason: str,
    ) -> None:
        """Render the semantic comparison without dumping its JSON receipt."""
        verdict = "QUALIFIED" if not missing_meaning and not added_meaning else "REJECTED"
        missing = "\n".join(f"- {item}" for item in missing_meaning) or "- none"
        added = "\n".join(f"- {item}" for item in added_meaning) or "- none"
        duplicates = ", ".join(semantic_duplicates) or "none"
        self.panel(
            f"Conformance · candidate {candidate_id} · {phase}",
            f"**Back-translation:** {back_translation}\n\n"
            f"**Missing meaning:**\n{missing}\n\n"
            f"**Added meaning:**\n{added}\n\n"
            f"**Semantic duplicates:** {duplicates}\n\n"
            f"**Verdict:** {verdict}\n\n"
            f"**Reason:** {reason}",
            style="green" if verdict == "QUALIFIED" else "yellow",
        )

    def conformance_input(
        self,
        *,
        precondition: str,
        normal_postcondition: str | None,
        exceptional_postcondition: str | None,
        support: list[dict],
        candidate_ids: list[str],
    ) -> None:
        """Show the complete semantic input while leaving raw transport JSON in artifacts."""
        evidence = "\n".join(
            f"- **{item.get('role', 'support')}:** {item.get('text', '')}"
            for item in support
        ) or "- none"
        self.panel(
            "Conformance input",
            "**Method:** LLM bidirectional semantic review after deterministic parsing, "
            "runtime-monitor, assertion-exercise, intended-violation, and determinism checks.\n\n"
            f"**Candidates:** {', '.join(candidate_ids)}\n\n"
            f"**Precondition:** {precondition}\n\n"
            f"**Normal postcondition:** {normal_postcondition or 'not specified'}\n\n"
            f"**Exceptional postcondition:** {exceptional_postcondition or 'not specified'}\n\n"
            f"**Supporting evidence:**\n{evidence}\n\n"
            "**Runtime rule:** operational failures are not implicit contract clauses.",
            style="blue",
        )

    def conformance_input_receipt(self, packet: dict, *, max_chars: int = 6000) -> None:
        """Show the bounded reviewer receipt at the terminal boundary.

        The complete JSON packet is persisted by the workflow.  This rendering is
        intentionally human-readable and bounded so long source files or command
        logs do not make the live terminal unusable.
        """
        def excerpt(value: object, limit: int = max_chars) -> str:
            text = value if isinstance(value, str) else json.dumps(
                value, ensure_ascii=False, indent=2, default=str
            )
            if len(text) <= limit:
                return text
            return text[:limit] + "\n… [truncated; full receipt is in the saved artifact]"

        obligation = packet.get("obligation", {})
        sections = [
            "**Frozen NLC obligation**\n```json\n"
            + excerpt(obligation)
            + "\n```",
        ]
        for candidate in packet.get("candidates", []):
            execution = candidate.get("execution", {})
            sections.append(
                f"**Candidate {candidate.get('id', '?')} source**\n```python\n"
                + excerpt(candidate.get("source", ""))
                + "\n```"
            )
            sections.append(
                f"**Candidate {candidate.get('id', '?')} raw execution events**\n```json\n"
                + excerpt(execution.get("events", []))
                + "\n```"
            )
            sections.append(
                f"**Candidate {candidate.get('id', '?')} rendered execution trace**\n````text\n"
                + excerpt("\n".join(execution.get("execution_trace", [])), 4000)
                + "\n````"
            )
            sections.append(
                f"**Candidate {candidate.get('id', '?')} assertion execution receipt**\n```json\n"
                + excerpt(execution.get("assertion_execution", {}), 4000)
                + "\n```"
            )
            sections.append(
                f"**Candidate {candidate.get('id', '?')} command output**\n````text\n"
                + excerpt(execution.get("command_log_tail", ""), 4000)
                + "\n````"
            )
        batch = packet.get("synthesis_batch", "?")
        self.panel(
            f"EC conformance reviewer receipt · synthesis batch {batch}",
            "\n\n".join(sections),
            style="blue",
        )

    def contract_recovery(
        self,
        *,
        synthesis_batch: int,
        feedback: list[dict],
        remaining_budget: int,
    ) -> None:
        """Show why another executable-contract synthesis call is or is not useful."""
        rows = []
        for item in feedback:
            execution = item.get("execution") or {}
            raised = [
                event
                for event in execution.get("events", [])
                if event.get("event") == "contracted_operation_raised"
            ]
            observed = "; ".join(
                f"{event.get('exception_type', 'unknown')}: {event.get('detail', '')}".rstrip(": ")
                for event in raised
            ) or execution.get("outcome", "unknown")
            rows.append(
                [
                    item.get("candidate_id", "?"),
                    item.get("failure_owner", "UNKNOWN"),
                    ", ".join(item.get("failures") or []) or "none",
                    observed,
                    item.get("required_action", ""),
                ]
            )
        self.table(
            f"Contract recovery · after batch {synthesis_batch} · remaining budget {remaining_budget}",
            ["Candidate", "Owner", "Failures", "Observed evidence", "Required correction"],
            rows,
        )

    def final_summary(self, report: dict, *, artifacts: str | Path | None = None) -> None:
        """Render the one terminal human summary after all chronological events."""
        accounting = report.get("model_accounting") or {}
        rows: list[list[object]] = [
            ["Status", report.get("status", "UNKNOWN")],
        ]
        if report.get("total_elapsed_seconds") is not None:
            rows.append(
                ["Total elapsed time", self._duration(report["total_elapsed_seconds"])]
            )
        aliases = {
            "executable_contract_sha256": report.get("probe_sha256"),
            "contract_synthesis_batches": report.get("probe_attempts"),
        }
        for label, key in (
            ("Diagnostic", "diagnostic"),
            ("Failure owner", "failure_owner"),
            ("Qualification", "qualification"),
            ("Instance", "instance_id"),
            ("Primary candidate", "primary_candidate_id"),
            ("Frozen SHA-256", "frozen_sha256"),
            ("Executable-contract SHA-256", "executable_contract_sha256"),
            ("Localization rounds", "localization_rounds"),
            ("Contract synthesis batches", "contract_synthesis_batches"),
            ("Candidates", "candidates_generated"),
        ):
            value = report.get(key, aliases.get(key))
            if value is not None:
                rows.append([label, value])
        repair = report.get("repair") or {}
        if repair:
            for label, key in (
                ("Repair mode requested", "requested_variant"),
                ("Repair mode effective", "effective_mode"),
                ("Patch attempts", "patch_attempts"),
                ("Patch model calls", "patch_model_calls"),
                ("Patch context rounds", "context_rounds"),
                ("Selected patch attempt", "selected_attempt"),
                ("Patch validation scope", "validation_scope"),
                ("Repository checks", "repository_check_status"),
                ("Frozen EC gate", "ec_pass"),
                ("Selected patch SHA-256", "selected_patch_sha256"),
            ):
                value = repair.get(key)
                if value is not None:
                    rows.append([label, value])
        if accounting:
            configuration = accounting.get("configuration") or {}
            if configuration:
                reasoning = "disabled"
                if configuration.get("reasoning_max_tokens") is not None:
                    reasoning = f"max {configuration['reasoning_max_tokens']} tokens"
                elif configuration.get("reasoning_effort"):
                    reasoning = f"effort {configuration['reasoning_effort']}"
                elif configuration.get("thinking"):
                    reasoning = "enabled"
                rows.extend(
                    [
                        ["Provider", configuration.get("provider", "unreported")],
                        ["Model", configuration.get("model", "unreported")],
                        ["Provider route", configuration.get("provider_route") or "automatic"],
                        ["Context window", configuration.get("context_window", "unreported")],
                        ["Completion ceiling", configuration.get("max_tokens", "unreported")],
                        ["Base client reasoning", reasoning],
                        [
                            "Sampling",
                            "temperature={} · top_p={} · top_k={}".format(
                                configuration.get("temperature", "unreported"),
                                configuration.get("top_p", "unreported"),
                                configuration.get("top_k", "unreported"),
                            ),
                        ],
                    ]
                )
            by_stage = accounting.get("by_stage") or {}
            if by_stage:
                stage_policy = []
                for stage, values in by_stage.items():
                    requested = values.get("requested_effort") or "unconfigured"
                    resolved = values.get("resolved_effort") or "unconfigured"
                    ceiling = values.get("completion_limit", "unreported")
                    stage_policy.append(f"{stage}={requested}→{resolved}/{ceiling}")
                rows.append(["Effective stage policy", "; ".join(stage_policy)])
            rows.extend(
                [
                    ["Model calls", accounting.get("calls", "unreported")],
                    ["Model stages", accounting.get("stage_calls", "unreported")],
                    ["Total model time", self._duration(accounting.get("model_elapsed_seconds"))],
                    ["Total input tokens", accounting.get("input_tokens", "unreported")],
                    ["Total output tokens", accounting.get("output_tokens", "unreported")],
                    ["Total reasoning tokens", accounting.get("reasoning_tokens", "unreported")],
                    [
                        "Thinking text captured",
                        (
                            f"{accounting['thinking_text_calls']} call(s); "
                            f"{accounting.get('thinking_text_unavailable_calls', 0)} unavailable"
                            if accounting.get("thinking_text_calls") is not None
                            else "unreported"
                        ),
                    ],
                    [
                        "Thinking-text assurance",
                        accounting.get("thinking_text_assurance", "unreported"),
                    ],
                    [
                        "Calls missing reasoning-token count",
                        accounting.get("missing_reasoning_token_calls", "unreported"),
                    ],
                    ["Final finish reason", accounting.get("last_finish_reason", "unreported")],
                    ["Cumulative cost (USD)", self._cost(accounting.get("provider_reported_cost_usd"))],
                    ["Cost accounting", accounting.get("cost_assurance", "unreported")],
                ]
            )
        if artifacts is not None:
            rows.append(["Artifacts", str(artifacts)])
            for name in ("report.md", "summary.json", "status.json"):
                path = Path(artifacts) / name
                if path.exists():
                    rows.append([name, str(path)])
            selected_patch = Path(artifacts) / "repair" / "selected.patch"
            if selected_patch.exists():
                rows.append(["Selected patch", str(selected_patch)])
        self.table("ContractFix · Final summary", ["Field", "Value"], rows)

    @staticmethod
    def _duration(value: object) -> str:
        if not isinstance(value, (int, float)):
            return "unreported"
        seconds = max(0.0, float(value))
        minutes, remainder = divmod(seconds, 60)
        hours, minutes = divmod(int(minutes), 60)
        if hours:
            return f"{hours}h {minutes:02d}m {remainder:05.2f}s"
        if minutes:
            return f"{minutes}m {remainder:05.2f}s"
        return f"{remainder:.2f}s"

    @staticmethod
    def _cost(value: object) -> str:
        return f"${value:.8f}" if isinstance(value, (int, float)) else "unreported"

    def _write(self, message: str, style: str) -> None:
        value = redact(str(message))
        self.console.print(value, style=style, markup=False, highlight=False)
        if self.transcript is not None:
            self.transcript.print(value, markup=False, highlight=False)

    def debug(self, message: str) -> None:
        if os.getenv("CONTRACTFIX_LOG_LEVEL", "INFO").upper() == "DEBUG":
            self._write(message, "dim")

    def muted(self, message: str) -> None:
        self._write(message, "dim")

    def info(self, message: str) -> None:
        self._write(message, "cyan")

    def success(self, message: str) -> None:
        self._write(message, "green")

    def warning(self, message: str) -> None:
        self._write(message, "yellow")

    def error(self, message: str) -> None:
        self._write(message, "bold red")

    @contextmanager
    def activity(
        self, label: str, *, detail: str | Callable[[], str]
    ) -> Generator[None, None, None]:
        """Show indeterminate work as one live line, with file-only heartbeats.

        A callable detail lets a controller update the stage without emitting a
        new terminal line for every heartbeat.
        """
        raw_interval = os.getenv("CONTRACTFIX_PROGRESS_INTERVAL_SECONDS", "15")
        try:
            interval = float(raw_interval)
        except ValueError:
            interval = 15.0
        interval = max(0.0, interval)
        started = perf_counter()
        stopped = Event()

        def current_detail() -> str:
            return detail() if callable(detail) else detail

        progress_console: Console | None = None
        tty_stream = None
        style = os.getenv("CONTRACTFIX_PROGRESS_STYLE", "spinner").lower()
        if style == "spinner":
            if self.console.is_terminal:
                progress_console = self.console
            else:
                try:
                    tty_stream = open("/dev/tty", "w", encoding="utf-8", buffering=1)
                except OSError:
                    tty_stream = None
                else:
                    progress_console = Console(
                        file=tty_stream,
                        force_terminal=True,
                        color_system="auto",
                    )
        status = (
            Status(
                f"{label} · {current_detail()} · elapsed=0.00s",
                console=progress_console,
                spinner="dots",
            )
            if progress_console is not None
            else None
        )
        if status is not None:
            status.start()
        else:
            self.muted(f"{label} {current_detail()}")
        if status is not None and self.transcript is not None:
            self.transcript.print(
                redact(f"{label} {current_detail()}"), markup=False, highlight=False
            )

        def heartbeat() -> None:
            next_transcript = interval
            while not stopped.wait(1.0):
                elapsed_seconds = perf_counter() - started
                elapsed = self._duration(elapsed_seconds)
                if status is not None:
                    status.update(
                        f"{label} · {current_detail()} · elapsed={elapsed}"
                    )
                if (
                    interval
                    and self.transcript is not None
                    and elapsed_seconds >= next_transcript
                ):
                    self.transcript.print(
                        redact(
                            f"{label} still active | elapsed={elapsed} | {current_detail()}"
                        ),
                        markup=False,
                        highlight=False,
                    )
                    next_transcript += interval

        worker = Thread(
            target=heartbeat,
            name="contractfix-progress-heartbeat",
            daemon=True,
        )
        if status is not None or (interval and self.transcript is not None):
            worker.start()
        try:
            yield
        finally:
            stopped.set()
            if worker.is_alive():
                worker.join(timeout=1.0)
            if status is not None:
                status.stop()
            if tty_stream is not None:
                tty_stream.close()

    @contextmanager
    def timed(self, label: str) -> Generator[None, None, None]:
        started = perf_counter()
        self.info(f"{label} started")
        try:
            yield
        except BaseException:
            self.error(f"{label} failed after {perf_counter() - started:.2f}s")
            raise
        else:
            self.success(f"{label} finished in {perf_counter() - started:.2f}s")


class RedactFilter(logging.Filter):
    """Remove credentials from standard-library log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


logger = ContractFixLogger()


def configure_logging(level: str | None = None) -> None:
    """Configure project loggers without resetting an embedding application's handlers."""
    handler = RichHandler(
        console=logger.console,
        rich_tracebacks=False,
        show_path=False,
        markup=False,
    )
    handler.addFilter(RedactFilter())
    resolved_level = (level or os.getenv("CONTRACTFIX_LOG_LEVEL", "INFO")).upper()
    for logger_name in ("contractfix", "baselines"):
        target = logging.getLogger(logger_name)
        target.setLevel(resolved_level)
        if not target.handlers:
            target.addHandler(handler)
        target.propagate = False
