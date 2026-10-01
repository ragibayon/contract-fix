"""Compatibility exports for the former combined Python schema module."""

from .languages.python.models import *  # noqa: F403
from .task import Task

__all__ = ["Task"]
