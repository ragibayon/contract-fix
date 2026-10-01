"""Optional equivalence checks for pure Python runtime assertions."""

from __future__ import annotations

import ast
from typing import Any


def _entrypoint(source: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "contractfix_contract"
    ]
    if len(functions) != 1:
        raise ValueError("expected one contractfix_contract function")
    return functions[0]


def _witness_shape(function: ast.FunctionDef) -> str:
    clone = ast.FunctionDef(
        name=function.name,
        args=function.args,
        body=[node for node in function.body if not isinstance(node, ast.Assert)],
        decorator_list=[],
        returns=function.returns,
        type_comment=function.type_comment,
        type_params=getattr(function, "type_params", []),
    )
    return ast.dump(clone, annotate_fields=True, include_attributes=False)


def _translate(node: ast.AST, z3: Any, names: dict[str, Any]):
    if isinstance(node, ast.Constant) and type(node.value) in {bool, int}:
        return z3.BoolVal(node.value) if isinstance(node.value, bool) else z3.IntVal(node.value)
    if isinstance(node, ast.Name):
        return names.setdefault(node.id, z3.Int(node.id))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return z3.Not(_translate(node.operand, z3, names))
    if isinstance(node, ast.BoolOp):
        values = [_translate(value, z3, names) for value in node.values]
        return z3.And(*values) if isinstance(node.op, ast.And) else z3.Or(*values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult)):
        left, right = _translate(node.left, z3, names), _translate(node.right, z3, names)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        return left * right
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        left = _translate(node.left, z3, names)
        right = _translate(node.comparators[0], z3, names)
        operation = node.ops[0]
        if isinstance(operation, ast.Eq):
            return left == right
        if isinstance(operation, ast.NotEq):
            return left != right
        if isinstance(operation, ast.Lt):
            return left < right
        if isinstance(operation, ast.LtE):
            return left <= right
        if isinstance(operation, ast.Gt):
            return left > right
        if isinstance(operation, ast.GtE):
            return left >= right
    raise ValueError(f"unsupported assertion expression: {type(node).__name__}")


def compare_assertions_with_z3(first: str, second: str) -> dict[str, str]:
    """Prove equivalence for a narrow pure subset; otherwise return ``UNKNOWN``."""
    try:
        import z3  # type: ignore[import-not-found]
    except ImportError:
        return {"status": "UNKNOWN", "reason": "z3-solver is not installed"}
    try:
        first_function, second_function = _entrypoint(first), _entrypoint(second)
        if _witness_shape(first_function) != _witness_shape(second_function):
            return {"status": "UNKNOWN", "reason": "contract witness shapes differ"}
        first_assertions = [
            node.test for node in ast.walk(first_function) if isinstance(node, ast.Assert)
        ]
        second_assertions = [
            node.test for node in ast.walk(second_function) if isinstance(node, ast.Assert)
        ]
        names: dict[str, Any] = {}
        left = z3.And(*[_translate(node, z3, names) for node in first_assertions])
        right = z3.And(*[_translate(node, z3, names) for node in second_assertions])
        solver = z3.Solver()
        solver.add(z3.Xor(left, right))
        result = solver.check()
        if result == z3.unsat:
            return {"status": "PROVED_EQUIVALENT", "reason": "predicate XOR is unsatisfiable"}
        if result == z3.sat:
            return {"status": "PROVED_DIFFERENT", "reason": "predicate XOR is satisfiable"}
        return {"status": "UNKNOWN", "reason": "solver returned unknown"}
    except (SyntaxError, ValueError, TypeError, z3.Z3Exception) as exc:
        return {"status": "UNKNOWN", "reason": str(exc)}
