from __future__ import annotations

import ast
from typing import Any


class ExprToGLSL(ast.NodeVisitor):
    """Translate a small, safe Python expression subset to GLSL.

    Supported:
    - Names: f0..f7 (vector proxies), true/false
    - Subscript: f0[0..3] → .x/.y/.z/.w
    - Unary: +/-/not
    - Binary: +,-,*,/,** (→ pow)
    - Compare: <,<=,>,>=,==,!=
    - Calls: abs, min, max, clamp, relu, dot, length/norm, select(cond,a,b),
             vec2/vec3/vec4 constructors, argmax(v)
    """

    def __init__(self, array_sizes: dict[str, int]) -> None:
        super().__init__()
        self.result = ""
        self.array_sizes = dict(array_sizes)

    def translate(self, expr: str) -> str:
        tree = ast.parse(expr, mode="eval")
        return self.visit(tree.body)

    def visit_Name(self, node: ast.Name) -> str:
        name = node.id
        if name in {"True", "true"}:
            return "true"
        if name in {"False", "false"}:
            return "false"
        if name.startswith("f") and name[1:].isdigit():
            if name not in self.array_sizes:
                raise ValueError(f"Unknown array name: {name}")
            return name  # mapped to helpers in shader
        raise ValueError(f"Unknown name: {name}")

    def visit_Constant(self, node: ast.Constant) -> str:
        v = node.value
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            if isinstance(v, int):
                return f"{float(v):.6g}"
            return f"{float(v):.6g}"
        raise ValueError(f"Unsupported constant: {v!r}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        op = node.op
        v = self.visit(node.operand)
        if isinstance(op, ast.USub):
            return f"(-{v})"
        if isinstance(op, ast.UAdd):
            return f"(+{v})"
        if isinstance(op, ast.Not):
            return f"(!({v}))"
        raise ValueError(f"Unsupported unary op: {ast.dump(op)}")

    def visit_BinOp(self, node: ast.BinOp) -> str:
        a = self.visit(node.left)
        b = self.visit(node.right)
        op = node.op
        if isinstance(op, ast.Add):
            return f"({a} + {b})"
        if isinstance(op, ast.Sub):
            return f"({a} - {b})"
        if isinstance(op, ast.Mult):
            return f"({a} * {b})"
        if isinstance(op, ast.Div):
            return f"({a} / {b})"
        if isinstance(op, ast.Pow):
            return f"pow({a}, {b})"
        raise ValueError(f"Unsupported binary op: {ast.dump(op)}")

    def visit_Compare(self, node: ast.Compare) -> str:
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("Only single comparisons supported")
        a = self.visit(node.left)
        b = self.visit(node.comparators[0])
        op = node.ops[0]
        if isinstance(op, ast.Lt):
            return f"({a} < {b})"
        if isinstance(op, ast.Gt):
            return f"({a} > {b})"
        if isinstance(op, ast.LtE):
            return f"({a} <= {b})"
        if isinstance(op, ast.GtE):
            return f"({a} >= {b})"
        if isinstance(op, ast.Eq):
            return f"({a} == {b})"
        if isinstance(op, ast.NotEq):
            return f"({a} != {b})"
        raise ValueError(f"Unsupported comparator: {ast.dump(op)}")

    def visit_Subscript(self, node: ast.Subscript) -> str:
        if not isinstance(node.value, ast.Name):
            raise ValueError("Indexing is only supported on f0..fN")
        base_name = node.value.id
        if base_name not in self.array_sizes:
            raise ValueError(f"Unknown array name: {base_name}")
        target = self.visit(node.value)
        idx = None
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
            idx = node.slice.value
        elif isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Constant):  # type: ignore[attr-defined]
            idx = node.slice.value.value  # type: ignore[assignment]
        else:
            raise ValueError("Only constant indices 0..3 supported")
        total = self.array_sizes.get(base_name, 0)
        if idx < 0 or idx >= total:
            raise ValueError(f"Index {idx} out of range for {base_name} (size {total})")
        if idx <= 3:
            swz = [".x", ".y", ".z", ".w"][idx]
            return f"({target}{swz})"
        return f"{target}_at({idx})"

    def visit_Call(self, node: ast.Call) -> str:
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "np":
            fname = node.func.attr
        elif isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname.startswith("f") and fname.endswith("_argmin") and fname[1:-7].isdigit():
                return f"{fname}()"
            if fname.startswith("f") and fname.endswith("_argmax") and fname[1:-7].isdigit():
                return f"{fname}()"
        else:
            raise ValueError("Unsupported call")
        args = [self.visit(a) for a in node.args]
        if fname in ("abs", "min", "max", "clamp", "dot", "floor", "ceil", "float"):
            if fname == "clamp":
                return f"clamp({args[0]}, {args[1]}, {args[2]})"
            if fname in ("floor", "ceil"):
                return f"{fname}({args[0]})"
            if fname == "float":
                return f"float({args[0]})"
            return f"{fname}({', '.join(args)})"
        if fname in ("length", "norm"):
            return f"length({args[0]})"
        if fname == "relu":
            return f"relu({args[0]})"
        if fname in ("vec2", "vec3", "vec4"):
            return f"{fname}({', '.join(args)})"
        if fname == "select":
            # select(cond, a, b) → (cond ? a : b)
            if len(args) != 3:
                raise ValueError("select(cond,a,b) requires 3 arguments")
            return f"(({args[0]}) ? ({args[1]}) : ({args[2]}))"
        if fname == "argmax":
            return f"argmax({args[0]})"
        if fname == "argmin":
            return f"argmin({args[0]})"
        raise ValueError(f"Unsupported function: {fname}")


def to_glsl(expr: str, array_sizes: dict[str, int]) -> str:
    return ExprToGLSL(array_sizes).translate(expr)
