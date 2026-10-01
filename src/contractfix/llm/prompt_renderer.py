"""Jinja2 prompt-template rendering utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

def render_prompt(
    template_dir: str | Path,
    template_name: str,
    **context: Any,
) -> str:
    """
    Render a Jinja2 prompt from a caller-provided template directory.
    """
    template_dir = Path(template_dir).resolve()

    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    try:
        template = env.get_template(template_name)
    except TemplateNotFound as exc:
        raise FileNotFoundError(
            f"Prompt template '{template_name}' not found in {template_dir}"
        ) from exc

    return template.render(**context)


def render_prompt_from_path(
    template_path: str | Path,
    **context: Any,
) -> str:
    """
    Render a Jinja2 prompt from a direct template file path.
    """
    template_path = Path(template_path).resolve()

    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {template_path}")

    return render_prompt(
        template_dir=template_path.parent,
        template_name=template_path.name,
        **context,
    )
