"""Only small model-facing outputs. Provenance, roles and hashes are host-owned."""
from pydantic import BaseModel, ConfigDict, Field

class NaturalClause(BaseModel):
    """One atomic natural-language DbC obligation, not a proposed implementation."""
    model_config=ConfigDict(extra="forbid", strict=True)
    applies_when: str=Field(min_length=1,max_length=400)
    must_hold: str=Field(min_length=1,max_length=600)
    evidence_id: str=Field(min_length=1,max_length=100)

class ExecutableClause(BaseModel):
    """One executable guard and Boolean consequence over allowed observations."""
    model_config=ConfigDict(extra="forbid", strict=True)
    when: str=Field(min_length=1,max_length=2000)
    ensure: str=Field(min_length=1,max_length=2000)

class Echo(BaseModel):
    """Provider transport check, not a contract-quality benchmark."""
    model_config=ConfigDict(extra="forbid",strict=True)
    value: str
