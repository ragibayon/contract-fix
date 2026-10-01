"""Compatibility imports for the Python prompt pack.

New code should import from ``workflow.languages.python.promptbook``.
"""

from .languages.python.promptbook import PromptBook, SCHEMAS

__all__ = ["PromptBook", "SCHEMAS"]
