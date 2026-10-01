"""Attach monitors to qualified function names, never guessed patch line numbers."""
from __future__ import annotations
import ast
from .core import validate_clause
from .expressions import Unsupported


def instrument(source: str, relative: str, clauses: list[dict]) -> str:
    tree = ast.parse(source)
    wanted = {clause['target']['symbol'] for clause in clauses}
    found: set[str] = set()
    edits: list[tuple[int, str]] = []
    alias = '_contractfix_ec_monitor'
    while alias in source:
        alias += '_'
    lines = source.splitlines(keepends=True)

    def visit(body: list[ast.stmt], prefix: str = '') -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, prefix + node.name + '.')
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + node.name
                if name not in wanted:
                    continue
                if name in found:
                    raise Unsupported('ambiguous_target:' + name)
                if isinstance(node, ast.AsyncFunctionDef):
                    raise Unsupported('async_target')
                if any(isinstance(x, (ast.Yield, ast.YieldFrom)) for x in ast.walk(node)):
                    raise Unsupported('generator_target')
                params = {a.arg for a in node.args.posonlyargs + node.args.args + node.args.kwonlyargs}
                params |= {a.arg for a in (node.args.vararg, node.args.kwarg) if a is not None}
                for clause in clauses:
                    if clause['target']['symbol'] == name:
                        validate_clause(clause, params)
                # Import before existing decorators, monitor innermost. This
                # preserves staticmethod/classmethod/property decorator ordering.
                first = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
                indent = lines[node.lineno - 1][:node.col_offset]
                import_text = indent + 'from contractfix.contracts.runtime import monitor as ' + alias + '\n'
                decorator = indent + '@' + alias + '(' + repr(relative + ':' + name) + ')\n'
                if first == node.lineno - 1:
                    edits.append((first, import_text + decorator))
                else:
                    edits.extend([(first, import_text), (node.lineno - 1, decorator)])
                found.add(name)
    visit(tree.body)
    if found != wanted:
        raise Unsupported('missing_target:' + repr(sorted(wanted - found)))
    for position, text in sorted(edits, reverse=True):
        lines.insert(position, text)
    updated = ''.join(lines)
    compile(updated, relative, 'exec')
    return updated
