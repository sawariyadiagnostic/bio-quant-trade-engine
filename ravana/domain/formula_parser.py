"""Mathematical Formula String Parser & AST Compiler for RAVANA.

Parses formula strings (including infix arithmetic, function calls, scientific notation,
unary negation, redundant parentheses, BLEND, and rolling correlation) into typed ASTNode hierarchies.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .formula_ast import (
    ASTNode,
    OPERATORS_BINARY_ARITHMETIC,
    OPERATORS_BINARY_ROLLING,
    OPERATORS_UNARY,
    OPERATORS_ROLLING_UNARY,
    OPERATORS_TERNARY,
)

OP_MAP: Dict[str, str] = {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV_SAFE"}


class FormulaParser:
    """Compiles formula strings into executable ASTNode expression graphs."""

    @staticmethod
    def _strip_redundant_parens(s: str) -> str:
        """Strips redundant outer matching parentheses in O(N) linear time."""
        s = s.strip()
        if not s.startswith("(") or not s.endswith(")"):
            return s
        stack: List[int] = []
        matches: Dict[int, int] = {}
        for i, c in enumerate(s):
            if c == "(":
                stack.append(i)
            elif c == ")":
                if stack:
                    matches[stack.pop()] = i

        left, right = 0, len(s) - 1
        while left < right:
            while left < right and s[left].isspace():
                left += 1
            while left < right and s[right].isspace():
                right -= 1
            if left < right and s[left] == "(" and matches.get(left) == right:
                left += 1
                right -= 1
            else:
                break
        return s[left: right + 1].strip()

    @staticmethod
    def parse(formula_str: str, _depth: int = 0) -> ASTNode:
        """Recursively parses a mathematical formula string into an ASTNode."""
        s = formula_str.strip()
        if not s or _depth > 40:
            return ASTNode(op="CONST", val="0.0")

        if _depth == 0 and not FormulaParser._is_balanced(s):
            raise ValueError(f"Unbalanced parentheses in formula: {s}")

        s = FormulaParser._strip_redundant_parens(s)

        split = FormulaParser._find_top_level_binary_op(s)
        if split:
            op_sym, left, right = split
            return ASTNode(
                op=OP_MAP.get(op_sym, op_sym),
                children=[
                    FormulaParser.parse(left, _depth + 1),
                    FormulaParser.parse(right, _depth + 1)
                ]
            )

        if s.startswith("-"):
            try:
                float(s)
            except ValueError:
                return ASTNode(
                    op="NEG",
                    children=[FormulaParser.parse(s[1:].strip(), _depth + 1)],
                )

        if s.startswith("+"):
            try:
                float(s)
            except ValueError:
                return FormulaParser.parse(s[1:].strip(), _depth + 1)

        m = re.match(r"^([A-Za-z0-9_]+)\s*\((.*)\)$", s)
        if m:
            op_name = m.group(1).upper()
            if op_name in ("BLEND", "MEGA_SWITCH"):
                sub_args = FormulaParser._split_top_level_args(m.group(2))
                return ASTNode(
                    op="BLEND",
                    children=[FormulaParser.parse(a, _depth + 1) for a in sub_args]
                )
            if op_name in ("ADD", "SUB", "MUL", "DIV", "DIV_SAFE", "TS_CORR_20"):
                sub_args = FormulaParser._split_top_level_args(m.group(2))
                if len(sub_args) == 2:
                    op_clean = "DIV_SAFE" if op_name == "DIV" else op_name
                    return ASTNode(
                        op=op_clean,
                        children=[
                            FormulaParser.parse(sub_args[0], _depth + 1),
                            FormulaParser.parse(sub_args[1], _depth + 1)
                        ]
                    )
            if op_name in (OPERATORS_UNARY + OPERATORS_ROLLING_UNARY):
                return ASTNode(
                    op=op_name,
                    children=[FormulaParser.parse(m.group(2), _depth + 1)]
                )

        # Check if numerical constant
        try:
            float(s)
            return ASTNode(op="CONST", val=s.replace(" ", ""))
        except ValueError:
            return ASTNode(op="FEATURE", val=s.replace(" ", ""))

    @staticmethod
    def _split_top_level_args(s: str) -> List[str]:
        args: List[str] = []
        d = 0
        cur: List[str] = []
        for c in s:
            if c == "(":
                d += 1
            elif c == ")":
                d -= 1
            if c == "," and d == 0:
                args.append("".join(cur).strip())
                cur = []
            else:
                cur.append(c)
        if cur:
            args.append("".join(cur).strip())
        return args

    @staticmethod
    def _is_balanced(s: str) -> bool:
        d = 0
        for c in s:
            d += (c == "(") - (c == ")")
            if d < 0:
                return False
        return d == 0

    @staticmethod
    def _find_top_level_binary_op(s: str) -> Optional[Tuple[str, str, str]]:
        for target_ops in [("+", "-"), ("*", "/")]:
            d = 0
            for i in range(len(s) - 1, -1, -1):
                c = s[i]
                d += (c == ")") - (c == "(")
                if d == 0 and c in target_ops:
                    before = s[:i].rstrip()
                    if c in ("+", "-") and (not before or before[-1] in ("(", "+", "-", "*", "/")):
                        continue
                    if c in ("+", "-") and i > 0 and s[i - 1] in ("e", "E") and (s[:i - 1].rstrip()[-1:].isdigit() or s[:i - 1].rstrip()[-1:] == "."):
                        continue
                    left, right = s[:i].strip(), s[i + 1:].strip()
                    if left and right:
                        return c, left, right
        return None
