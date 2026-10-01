"""Durable clause-evaluation receipts, independent of model memory or graph state."""

from __future__ import annotations
import copy
import json
import threading
from pathlib import Path
from contractfix.contracts.core import seal
from contractfix.contracts.runner import run
from contractfix.contracts.snapshots import repository_identity
from contractfix.utils.artifacts import atomic_json, digest


class ContractEvaluationSession:
    def __init__(
        self, repo, command, host_clause, output_dir, *, max_evaluations=3, timeout=120, events=None
    ):
        if max_evaluations < 1:
            raise ValueError("positive evaluation budget required")
        self.repo = Path(repo).resolve()
        self.command = list(command)
        if not self.command or not all(isinstance(x, str) and x for x in self.command):
            raise ValueError("command argv required")
        self.host = copy.deepcopy(host_clause)
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "contract_evaluation_state.json"
        self.max_evaluations, self.timeout, self.events = max_evaluations, timeout, events
        self._lock = threading.RLock()
        # Canonical state belongs to the host and is never exposed as an agent filesystem.
        self.source_identity = repository_identity(self.repo)
        if self.output_dir.is_relative_to(self.repo):
            raise ValueError("session artifacts must be outside the target repository")
        identity = {
            "repo": str(self.repo),
            "source_sha256": self.source_identity,
            "command": self.command,
            "host": self.host,
            "max_evaluations": max_evaluations,
            "timeout": timeout,
        }
        self.identity = digest(identity)
        self.state = {"identity": self.identity, "attempts": [], "selected": None, "unsupported": None}
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
            if self.state.get("identity") != self.identity:
                raise ValueError("evaluation resume configuration changed")
        else:
            atomic_json(self.path, self.state)

    def reserve_model_call(self, limit: int) -> bool:
        """Reserve before inference, including failed requests, across graph restarts."""
        if limit < 1:
            raise ValueError("positive model call limit required")
        with self._lock:
            prior = self.state.get("model_call_limit")
            if prior is not None and prior != limit:
                raise ValueError("model call limit changed on resume")
            self.state["model_call_limit"] = limit
            count = self.state.get("model_calls", 0)
            if count >= limit:
                return False
            self.state["model_calls"] = count + 1
            atomic_json(self.path, self.state)
            return True

    @property
    def selected(self):
        return self.state["selected"]

    def _reply(self, value):
        return json.dumps(value, ensure_ascii=False)

    def evaluate_clause(self, when: str, ensure: str) -> str:
        """Observe one clause on the base. A REPAIR violation may expose the bug."""
        with self._lock:
            if repository_identity(self.repo) != self.source_identity:
                return self._reply(
                    {"state": "SOURCE_CHANGED", "detail": "new qualification session required"}
                )
            key = digest({"when": when, "ensure": ensure})
            for prior in self.state["attempts"]:
                if prior["key"] == key and prior.get("response"):
                    return self._reply({**prior["response"], "cached": True})
            if self.selected is not None:
                return self._reply({"state": "FROZEN"})
            if len(self.state["attempts"]) >= self.max_evaluations:
                return self._reply({"state": "BUDGET_EXHAUSTED"})
            attempt = {"key": key, "when": when, "ensure": ensure, "status": "started"}
            self.state["attempts"].append(attempt)
            atomic_json(self.path, self.state)
            index = len(self.state["attempts"])
            try:
                candidate = {**self.host, "when": when, "ensure": ensure}
                bundle = seal([candidate])
                report = run(self.repo, bundle, self.command, timeout=self.timeout)
                atomic_json(
                    self.output_dir / f"contract-evaluation-{index}.json",
                    {"bundle": bundle, "report": report},
                )
                attempt.update(
                    status="completed", bundle=bundle, report_file=f"contract-evaluation-{index}.json"
                )
                feedback = {
                    "summary": report["summary"],
                    "examples": [
                        e
                        for e in report["events"]
                        if e["state"] in {"VIOLATED", "CHECKER_ERROR", "INCONCLUSIVE"}
                    ][:2],
                    "remaining_evaluations": self.max_evaluations - index,
                }
                if self.events:
                    self.events.emit(
                        "contract_evaluated",
                        stage="ec",
                        evaluation=index,
                        bundle_sha256=bundle["sha256"],
                        summary=report["summary"],
                    )
            except Exception as exc:
                attempt["status"] = "error"
                feedback = {
                    "state": "EVALUATION_ERROR",
                    "error_type": type(exc).__name__,
                    "detail": str(exc)[:300],
                }
            attempt["response"] = feedback
            atomic_json(self.path, self.state)
            return self._reply(feedback)

    def submit_clause(self, when: str, ensure: str) -> str:
        """Select an observed, non-crashing clause. This does NOT establish intent fidelity."""
        with self._lock:
            if repository_identity(self.repo) != self.source_identity:
                return self._reply(
                    {"state": "SOURCE_CHANGED", "detail": "new qualification session required"}
                )
            key = digest({"when": when, "ensure": ensure})
            prior = next(
                (a for a in self.state["attempts"] if a["key"] == key and a.get("report_file")), None
            )
            if prior is None:
                return self._reply({"state": "EVALUATION_REQUIRED"})
            evidence = json.loads((self.output_dir / prior["report_file"]).read_text())
            report = evidence["report"]
            summary = report["summary"]
            if (
                summary["infrastructure_error"]
                or summary["unexercised_clauses"]
                or any(e["state"] in {"CHECKER_ERROR", "INCONCLUSIVE"} for e in report["events"])
            ):
                return self._reply({"state": "CANDIDATE_NOT_EXECUTABLE"})
            if self.host["role"] == "PRESERVE" and summary["disposition"] != "OBSERVED_PASS":
                return self._reply({"state": "PRESERVATION_CONFLICT"})
            if self.selected is not None and self.selected != evidence["bundle"]:
                return self._reply({"state": "FROZEN"})
            self.state["selected"] = evidence["bundle"]
            atomic_json(self.path, self.state)
            return self._reply(
                {
                    "state": "CANDIDATE_READY",
                    "bundle_sha256": self.selected["sha256"],
                    "assurance": "executable_on_observed_inputs; host_intent_review_still_required",
                }
            )

    def report_unsupported(self, reason: str) -> str:
        """Stop rather than invent observations or produce a vacuous contract."""
        with self._lock:
            self.state["unsupported"] = reason[:1000]
            atomic_json(self.path, self.state)
            return self._reply({"state": "UNSUPPORTED", "reason": reason[:500]})
