"""Dimensional Tree Generator and Genetic Search Operators for Formula ASTs.

Enforces dimensional soundness across AST tree generation, point/subtree mutation,
and genetic crossover, preventing physical incompatibilities (e.g., adding Price to Volume).
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

from ravana.domain.formula_ast import (
    ASTNode,
    FeatureDimension,
    FEATURE_DIMENSIONS,
    OPERATORS_BINARY,
    OPERATORS_BINARY_ARITHMETIC,
    OPERATORS_BINARY_ROLLING,
    OPERATORS_ROLLING_UNARY,
    OPERATORS_TERNARY,
    OPERATORS_UNARY,
)


class TreeGenerator:
    """Randomly generates dimensionally sound AST expression trees with bounded depth."""

    @staticmethod
    def infer_dimension(node: ASTNode) -> FeatureDimension:
        """Infers the physical dimension of an AST node subtree."""
        if node.op == "CONST":
            return FeatureDimension.DIMENSIONLESS
        if node.op == "FEATURE":
            return FEATURE_DIMENSIONS.get(node.val, FeatureDimension.DIMENSIONLESS)
        if node.op in (
            "DIV_SAFE", "TS_CORR_20", "TS_RANK_20", "TS_ZSCORE_20",
            "SIGMOID", "TANH", "SIGN", "TS_ARGMAX_10", "TS_ARGMIN_10"
        ):
            return FeatureDimension.DIMENSIONLESS
        if node.op == "ADD" and len(node.children) >= 2:
            d0 = TreeGenerator.infer_dimension(node.children[0])
            d1 = TreeGenerator.infer_dimension(node.children[1])
            return d0 if d0 != FeatureDimension.DIMENSIONLESS else d1
        if node.op in ("ADD", "SUB") and node.children:
            return TreeGenerator.infer_dimension(node.children[0])
        if node.children:
            return TreeGenerator.infer_dimension(node.children[0])
        return FeatureDimension.DIMENSIONLESS

    @staticmethod
    def _leaf_node(target_dimension: FeatureDimension, method: str) -> ASTNode:
        """Helper to create a terminal leaf node matching target dimension."""
        matching_feats = [
            f for f, dim in FEATURE_DIMENSIONS.items()
            if dim == target_dimension
        ]
        if target_dimension == FeatureDimension.DIMENSIONLESS and random.random() < 0.25:
            const_val = str(round(random.uniform(-2.0, 2.0), 2))
            return ASTNode("CONST", const_val)
        if matching_feats:
            return ASTNode("FEATURE", random.choice(matching_feats))
        return ASTNode("CONST", "1.0")

    @classmethod
    def generate(
        cls,
        max_depth: int = 4,
        target_dimension: FeatureDimension = FeatureDimension.DIMENSIONLESS,
        method: str = "grow"
    ) -> ASTNode:
        """Generates an AST tree guaranteed to conform to target dimension."""
        if max_depth <= 1 or (method == "grow" and random.random() < 0.35):
            return cls._leaf_node(target_dimension, method)

        if target_dimension == FeatureDimension.DIMENSIONLESS:
            op_choices = (
                OPERATORS_BINARY_ARITHMETIC + OPERATORS_BINARY_ROLLING +
                OPERATORS_UNARY + OPERATORS_ROLLING_UNARY + OPERATORS_TERNARY
            )
            op = random.choice(op_choices)
        else:
            op = random.choice(("ADD", "SUB", "DELTA_5", "EMA_10", "TS_MAX_10", "TS_MIN_10", "TS_STD_20", "TS_MEDIAN_10"))

        return cls._build_operator_node(op, max_depth, target_dimension, method)

    @classmethod
    def _build_operator_node(
        cls, op: str, max_depth: int, target_dim: FeatureDimension, method: str
    ) -> ASTNode:
        """Dispatches operator-specific child tree generation."""
        if op == "BLEND":
            cond = cls.generate(max_depth - 1, FeatureDimension.DIMENSIONLESS, method)
            a = cls.generate(max_depth - 1, target_dim, method)
            b = cls.generate(max_depth - 1, target_dim, method)
            return ASTNode("BLEND", children=[cond, a, b])

        if op in OPERATORS_BINARY_ROLLING:
            a = cls.generate(max_depth - 1, FeatureDimension.DIMENSIONLESS, method)
            b = cls.generate(max_depth - 1, FeatureDimension.DIMENSIONLESS, method)
            return ASTNode(op, children=[a, b])

        if op in OPERATORS_BINARY_ARITHMETIC:
            if op == "DIV_SAFE":
                dim_choice = random.choice([FeatureDimension.PRICE, FeatureDimension.DIMENSIONLESS])
                left = cls.generate(max_depth - 1, dim_choice, method)
                right = cls.generate(max_depth - 1, dim_choice, method)
                return ASTNode("DIV_SAFE", children=[left, right])
            left = cls.generate(max_depth - 1, target_dim, method)
            right = cls.generate(max_depth - 1, target_dim, method)
            return ASTNode(op, children=[left, right])

        child_dim = (
            FeatureDimension.DIMENSIONLESS
            if op in ("TS_ZSCORE_20", "SIGMOID", "TANH", "TS_ARGMAX_10", "TS_ARGMIN_10")
            else target_dim
        )
        child = cls.generate(max_depth - 1, child_dim, method)
        return ASTNode(op, children=[child])


class GeneticMutator:
    """Applies point and subtree mutations respecting dimensional soundness."""

    @staticmethod
    def mutate_subtree(tree: ASTNode, max_depth: int = 3) -> ASTNode:
        """Replaces a randomly selected subtree with a newly generated dimensional branch."""
        cloned = tree.clone()
        all_nodes = cloned.get_all_subnodes()
        target = random.choice(all_nodes)
        target_dim = TreeGenerator.infer_dimension(target)
        sub = TreeGenerator.generate(max_depth, target_dim)
        target.op, target.val, target.children = sub.op, sub.val, sub.children
        return cloned.simplify()

    @staticmethod
    def mutate_point(tree: ASTNode) -> ASTNode:
        """Mutates a single node operator or feature without breaking arity."""
        cloned = tree.clone()
        target = random.choice(cloned.get_all_subnodes())
        if target.op == "FEATURE":
            current_dim = FEATURE_DIMENSIONS.get(target.val, FeatureDimension.DIMENSIONLESS)
            candidates = [f for f, d in FEATURE_DIMENSIONS.items() if d == current_dim]
            if candidates:
                target.val = random.choice(candidates)
        elif target.op == "CONST":
            target.val = str(round(random.uniform(-2.0, 2.0), 2))
        elif target.op in OPERATORS_BINARY_ARITHMETIC:
            target.op = random.choice(OPERATORS_BINARY_ARITHMETIC)
        elif target.op in OPERATORS_UNARY:
            target.op = random.choice(OPERATORS_UNARY)
        elif target.op in OPERATORS_ROLLING_UNARY:
            target.op = random.choice(OPERATORS_ROLLING_UNARY)
        return cloned.simplify()


class GeneticCrossover:
    """Performs genetic crossover with strict dimensional integrity and arity hardening."""

    @staticmethod
    def _harden_arity(node: ASTNode) -> None:
        """Hardens node arity after node exchange to prevent child count mismatches."""
        if node.op == "BLEND":
            while len(node.children) < 3:
                node.children.append(ASTNode("CONST", val="1.0"))
            if len(node.children) > 3:
                node.children = node.children[:3]
        elif node.op in OPERATORS_BINARY:
            while len(node.children) < 2:
                node.children.append(ASTNode("CONST", val="1.0"))
            if len(node.children) > 2:
                node.children = node.children[:2]
        elif node.op in (OPERATORS_UNARY + OPERATORS_ROLLING_UNARY):
            if len(node.children) == 0:
                node.children.append(ASTNode("CONST", val="1.0"))
            elif len(node.children) > 1:
                node.children = node.children[:1]
        elif node.op in ("FEATURE", "CONST"):
            node.children = []

    @classmethod
    def crossover(cls, parent1: ASTNode, parent2: ASTNode) -> Tuple[ASTNode, ASTNode]:
        """Exchanges compatible subtrees between two parents, enforcing dimensional parity."""
        c1, c2 = parent1.clone(), parent2.clone()
        nodes1 = c1.get_all_subnodes()
        nodes2 = c2.get_all_subnodes()

        t1 = random.choice(nodes1)
        dim1 = TreeGenerator.infer_dimension(t1)

        matching_t2 = [n for n in nodes2 if TreeGenerator.infer_dimension(n) == dim1]
        t2 = random.choice(matching_t2) if matching_t2 else random.choice(nodes2)

        t1.op, t2.op = t2.op, t1.op
        t1.val, t2.val = t2.val, t1.val
        t1.children, t2.children = [c.clone() for c in t2.children], [c.clone() for c in t1.children]

        cls._harden_arity(t1)
        cls._harden_arity(t2)

        return c1.simplify(), c2.simplify()
