"""Workflow-level failures distinct from model or repository behavior."""
class WorkflowError(RuntimeError):
    """A host-controlled protocol or evidence failure, never silent acceptance."""


class EvidenceGroundingError(WorkflowError):
    """A model-authored obligation failed deterministic provenance checks."""

    def __init__(self, code: str, detail: str, **context: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.context = context

    def feedback(self) -> dict[str, str]:
        return {"failure": self.code, "detail": self.detail, **self.context}


class LocalizationExhausted(WorkflowError):
    """The bounded localizer could not obtain enough repository context."""

    def __init__(self, detail: str, *, rounds: int):
        super().__init__(detail)
        self.detail = detail
        self.rounds = rounds
