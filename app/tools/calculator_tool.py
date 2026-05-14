from __future__ import annotations

import ast
import operator
from typing import Any

from langchain_core.tools import tool

_BINOPS: dict[type[ast.operator], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY: dict[type[ast.unaryop], Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_EXPONENT = 256


class UnsafeExpressionError(ValueError):
    pass


def evaluate_arithmetic(expression: str) -> float:
    raw = expression.strip().replace("^", "**")
    if not raw:
        raise UnsafeExpressionError("Empty expression.")
    tree = ast.parse(raw, mode="eval")
    if not isinstance(tree, ast.Expression):
        raise UnsafeExpressionError("Invalid parse tree.")
    return _eval_node(tree.body)


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise UnsafeExpressionError("Only numeric literals are allowed.")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return float(_UNARY[type(node.op)](_eval_node(node.operand)))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise UnsafeExpressionError("Exponent too large.")
        try:
            return float(_BINOPS[type(node.op)](left, right))
        except ZeroDivisionError as exc:
            raise UnsafeExpressionError("Division by zero.") from exc
    raise UnsafeExpressionError("Unsupported syntax.")


@tool
def calculator(expression: str) -> str:
    """Safe arithmetic: + - * / // % ** and parentheses only."""
    try:
        value = evaluate_arithmetic(expression)
        if value.is_integer():
            return str(int(value))
        return str(value)
    except (UnsafeExpressionError, SyntaxError, TypeError) as exc:
        return f"Could not evaluate: {exc}"
