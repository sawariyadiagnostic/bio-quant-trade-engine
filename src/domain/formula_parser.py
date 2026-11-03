"""Mathematical Formula String Parser & AST Compiler (Ponytail MVC)."""
import re
from typing import Optional
from .formula_ast import ASTNode, OPERATORS_UNARY, OPERATORS_ROLLING

OP_MAP = {"+": "ADD", "-": "SUB", "*": "MUL", "/": "DIV_SAFE"}


class FormulaParser:
    @staticmethod
    def parse(formula_str: str, active_regimes: set[int] = None, _depth: int = 0) -> ASTNode:
        regimes = active_regimes if active_regimes is not None else set(range(10))
        s = formula_str.strip()
        if not s or _depth > 40:
            return ASTNode(op="FEATURE", val="0.0", active_regimes=set(regimes))

        while s.startswith("(") and s.endswith(")") and FormulaParser._is_balanced(s[1:-1]):
            s = s[1:-1].strip()

        split = FormulaParser._find_top_level_binary_op(s)
        if split:
            op_sym, left, right = split
            return ASTNode(
                op=OP_MAP.get(op_sym, op_sym),
                active_regimes=set(regimes),
                children=[FormulaParser.parse(left, regimes, _depth + 1), FormulaParser.parse(right, regimes, _depth + 1)]
            )

        m = re.match(r"^([A-Za-z0-9_]+)\s*\((.*)\)$", s)
        if m:
            op_name = m.group(1).upper()
            if op_name == "MEGA_SWITCH":
                sub_args = FormulaParser._split_top_level_args(m.group(2))
                return ASTNode(
                    op="MEGA_SWITCH",
                    active_regimes=set(regimes),
                    children=[FormulaParser.parse(a, regimes, _depth + 1) for a in sub_args]
                )
            if op_name in (OPERATORS_UNARY + OPERATORS_ROLLING):
                return ASTNode(
                    op=op_name,
                    active_regimes=set(regimes),
                    children=[FormulaParser.parse(m.group(2), regimes, _depth + 1)]
                )

        return ASTNode(op="FEATURE", val=s.replace(" ", ""), active_regimes=set(regimes))

    @staticmethod
    def _split_top_level_args(s: str) -> list[str]:
        args = []
        d = 0
        cur = []
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
    def _find_top_level_binary_op(s: str) -> Optional[tuple[str, str, str]]:
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


class AlphaSeedLibrary:
    @staticmethod
    def get_classical_seeds() -> list[str]:
        return [
            "EMA_10(EMA_Slope)", "DELTA_5(Close_Adj) / (True_Range + 1e-8)", "RELU(EMA_Slope) * TS_RANK_20(Close_Adj)",
            "Vol_Price_Flow * SIGN(Vol_ZScore)", "TS_RANK_20(Volume) - TS_MIN_10(Volume)", "Vol_Price_Flow / (Downside_Vol_Ratio + 1e-8)",
            "TS_RANK_20(Dist_VWAP) * -1.0", "SIGN(0.5 - RSI_14) * True_Range", "Dist_EMA20 / (Parkinson_Vol + 1e-8)",
            "SP500_Ret * Dist_VWAP", "ADR_Ret + (EMA_Slope / (Garman_Klass_Vol + 1e-8))", "Global_Confluence * TS_MAX_10(TR_Expansion)",
            "LOG_ABS(Dist_EMA20) * (1.0 - Downside_Vol_Ratio)", "SIGMOID(Vol_ZScore) * DELTA_5(True_Range)"
        ]

    @staticmethod
    def load_seed_nodes(custom_formulas: list[str] = None) -> list[ASTNode]:
        formulas = list(AlphaSeedLibrary.get_classical_seeds()) + [f for f in (custom_formulas or []) if f.strip()]
        nodes = []
        for f in formulas:
            try:
                nodes.append(FormulaParser.parse(f))
            except Exception:
                pass
        return nodes
