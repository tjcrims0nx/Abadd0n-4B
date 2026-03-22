"""
Math evaluation for numerical and symbolic computation.
Safe eval: only numbers, math module, and basic operators.
"""

from __future__ import annotations

import ast
import math
import operator

# Whitelist of builtins for safe eval
_SAFE_NAMESPACE = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "pi": math.pi,
    "e": math.e,
    "floor": math.floor,
    "ceil": math.ceil,
    "degrees": math.degrees,
    "radians": math.radians,
    "factorial": math.factorial,
}


def evaluate_math(expr: str) -> dict:
    """
    Safely evaluate a math expression. Returns {ok, result?, error?}.
    Supports: +, -, *, /, //, %, **, parentheses, math functions.
    """
    expr = (expr or "").strip()
    if not expr:
        return {"ok": False, "error": "Empty expression"}
    try:
        tree = ast.parse(expr, mode="eval")
        result = _eval_node(tree.body)
        return {"ok": True, "result": result}
    except SyntaxError as e:
        return {"ok": False, "error": f"Syntax: {e}"}
    except ValueError as e:
        return {"ok": False, "error": str(e)}
    except ZeroDivisionError:
        return {"ok": False, "error": "Division by zero"}
    except OverflowError:
        return {"ok": False, "error": "Overflow"}


def _eval_node(node: ast.AST):
    """Evaluate AST node with whitelisted ops."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.USub):
            return -_eval_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return _eval_node(node.operand)
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return operator.add(left, right)
        if isinstance(node.op, ast.Sub):
            return operator.sub(left, right)
        if isinstance(node.op, ast.Mult):
            return operator.mul(left, right)
        if isinstance(node.op, ast.Div):
            return operator.truediv(left, right)
        if isinstance(node.op, ast.FloorDiv):
            return operator.floordiv(left, right)
        if isinstance(node.op, ast.Mod):
            return operator.mod(left, right)
        if isinstance(node.op, ast.Pow):
            return operator.pow(left, right)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Invalid function call")
        name = node.func.id
        if name not in _SAFE_NAMESPACE:
            raise ValueError(f"Unknown function or constant: {name}")
        args = [_eval_node(a) for a in node.args]
        return _SAFE_NAMESPACE[name](*args)
    if isinstance(node, ast.Name):
        if node.id in _SAFE_NAMESPACE:
            val = _SAFE_NAMESPACE[node.id]
            if callable(val):
                raise ValueError(f"{node.id} is a function, not a value")
            return val
        raise ValueError(f"Unknown symbol: {node.id}")
    raise ValueError("Unsupported expression")
