"""Dimensional Tree Generator and Genetic Search Operators for Formula ASTs.

Enforces dimensional soundness and active feature vocabulary constraints across AST
tree generation, point/subtree mutation, non-destructive bloat pruning, and crossover.
"""
from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

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
    def _leaf_node(
        target_dimension: FeatureDimension,
        allowed_features: Optional[Sequence[str]] = None,
    ) -> ASTNode:
        """Create a terminal leaf node matching target dimension and active vocabulary."""
        if allowed_features is not None:
            matching_feats = [
                f for f in allowed_features
                if FEATURE_DIMENSIONS.get(f, FeatureDimension.DIMENSIONLESS) == target_dimension
            ]
            if not matching_feats and target_dimension == FeatureDimension.DIMENSIONLESS:
                matching_feats = list(allowed_features)
        else:
            matching_feats = [
                f for f, dim in FEATURE_DIMENSIONS.items()
                if dim == target_dimension
            ]

        if target_dimension == FeatureDimension.DIMENSIONLESS and random.random() < 0.20:
            const_val = str(round(random.uniform(-2.0, 2.0), 2))
            return ASTNode("CONST", const_val)
        if matching_feats:
            return ASTNode("FEATURE", random.choice(matching_feats))
        if allowed_features:
            return ASTNode("FEATURE", random.choice(list(allowed_features)))
        return ASTNode("CONST", "1.0")

    @classmethod
    def generate(
        cls,
        max_depth: int = 4,
        target_dimension: FeatureDimension = FeatureDimension.DIMENSIONLESS,
        method: str = "grow",
        allowed_features: Optional[Sequence[str]] = None,
    ) -> ASTNode:
        """Generates an AST tree guaranteed to conform to target dimension and vocabulary."""
        if max_depth <= 1 or (method == "grow" and random.random() < 0.35):
            return cls._leaf_node(target_dimension, allowed_features)

        if target_dimension == FeatureDimension.DIMENSIONLESS:
            op_choices = (
                OPERATORS_BINARY_ARITHMETIC + OPERATORS_BINARY_ROLLING +
                OPERATORS_UNARY + OPERATORS_ROLLING_UNARY + OPERATORS_TERNARY
            )
            op = random.choice(op_choices)
        else:
            op = random.choice(("ADD", "SUB", "DELTA_5", "EMA_10", "TS_MAX_10", "TS_MIN_10", "TS_STD_20", "TS_MEDIAN_10"))

        return cls._build_operator_node(op, max_depth, target_dimension, method, allowed_features)

    @classmethod
    def _build_operator_node(
        cls,
        op: str,
        max_depth: int,
        target_dim: FeatureDimension,
        method: str,
        allowed_features: Optional[Sequence[str]] = None,
    ) -> ASTNode:
        """Dispatches operator-specific child tree generation."""
        if op == "BLEND":
            cond = cls.generate(max_depth - 1, FeatureDimension.DIMENSIONLESS, method, allowed_features)
            a = cls.generate(max_depth - 1, target_dim, method, allowed_features)
            b = cls.generate(max_depth - 1, target_dim, method, allowed_features)
            return ASTNode("BLEND", children=[cond, a, b])

        if op in OPERATORS_BINARY_ROLLING:
            a = cls.generate(max_depth - 1, FeatureDimension.DIMENSIONLESS, method, allowed_features)
            b = cls.generate(max_depth - 1, FeatureDimension.DIMENSIONLESS, method, allowed_features)
            return ASTNode(op, children=[a, b])

        if op in OPERATORS_BINARY_ARITHMETIC:
            if op == "DIV_SAFE":
                has_price = allowed_features is None or any(
                    FEATURE_DIMENSIONS.get(f) == FeatureDimension.PRICE for f in allowed_features
                )
                dim_choice = (
                    random.choice([FeatureDimension.PRICE, FeatureDimension.DIMENSIONLESS])
                    if has_price else FeatureDimension.DIMENSIONLESS
                )
                left = cls.generate(max_depth - 1, dim_choice, method, allowed_features)
                right = cls.generate(max_depth - 1, dim_choice, method, allowed_features)
                return ASTNode("DIV_SAFE", children=[left, right])
            left = cls.generate(max_depth - 1, target_dim, method, allowed_features)
            right = cls.generate(max_depth - 1, target_dim, method, allowed_features)
            return ASTNode(op, children=[left, right])

        child_dim = (
            FeatureDimension.DIMENSIONLESS
            if op in ("TS_ZSCORE_20", "SIGMOID", "TANH", "TS_ARGMAX_10", "TS_ARGMIN_10")
            else target_dim
        )
        child = cls.generate(max_depth - 1, child_dim, method, allowed_features)
        return ASTNode(op, children=[child])

    @classmethod
    def prune_to_limits(
        cls,
        tree: ASTNode,
        max_depth: int = 4,
        max_nodes: int = 15,
        allowed_features: Optional[Sequence[str]] = None,
    ) -> ASTNode:
        """Non-destructively prunes deep internal branches while preserving root structure."""
        root = tree.clone().simplify()
        root = cls._trim_depth(root, max_depth, allowed_features).simplify()

        attempts = 0
        while root.size() > max_nodes and attempts < 30:
            attempts += 1
            # Find internal nodes whose children are all leaves (deepest collapsible twigs)
            twigs = [
                n for n in root.get_all_subnodes()
                if n is not root and n.children and all(not c.children for c in n.children)
            ]
            if not twigs:
                break
            victim = random.choice(twigs)
            feat_children = [c for c in victim.children if c.op == "FEATURE"]
            replacement = random.choice(feat_children) if feat_children else victim.children[0]
            victim.op, victim.val, victim.children = replacement.op, replacement.val, []
            root = root.simplify()

        return root

    @classmethod
    def _trim_depth(
        cls,
        node: ASTNode,
        remaining_depth: int,
        allowed_features: Optional[Sequence[str]] = None,
    ) -> ASTNode:
        if remaining_depth <= 1 or not node.children:
            if not node.children:
                return node
            feat_leaves = [n for n in node.get_all_subnodes() if n.op == "FEATURE"]
            if feat_leaves:
                chosen = random.choice(feat_leaves)
                return ASTNode("FEATURE", chosen.val)
            return cls._leaf_node(cls.infer_dimension(node), allowed_features)
        node.children = [cls._trim_depth(c, remaining_depth - 1, allowed_features) for c in node.children]
        return node


class GeneticMutator:
    """Applies point and subtree mutations respecting dimensional soundness and active vocabulary."""

    @staticmethod
    def mutate_subtree(
        tree: ASTNode,
        max_depth: int = 3,
        allowed_features: Optional[Sequence[str]] = None,
    ) -> ASTNode:
        """Replaces a randomly selected subtree with a newly generated dimensional branch."""
        cloned = tree.clone()
        all_nodes = cloned.get_all_subnodes()
        target = random.choice(all_nodes)
        target_dim = TreeGenerator.infer_dimension(target)
        sub = TreeGenerator.generate(max_depth, target_dim, allowed_features=allowed_features)
        target.op, target.val, target.children = sub.op, sub.val, sub.children
        return cloned.simplify()

    @staticmethod
    def mutate_point(
        tree: ASTNode,
        allowed_features: Optional[Sequence[str]] = None,
    ) -> ASTNode:
        """Mutates a single node operator or feature without breaking arity or vocabulary."""
        cloned = tree.clone()
        target = random.choice(cloned.get_all_subnodes())
        if target.op == "FEATURE":
            current_dim = FEATURE_DIMENSIONS.get(target.val, FeatureDimension.DIMENSIONLESS)
            if allowed_features:
                matching = [
                    f for f in allowed_features
                    if FEATURE_DIMENSIONS.get(f, FeatureDimension.DIMENSIONLESS) == current_dim
                ]
                target.val = random.choice(matching) if matching else random.choice(list(allowed_features))
            else:
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
