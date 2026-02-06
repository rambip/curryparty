"""Core engine of lambda-calculus.

This file defines 2 main types:
- `AbstractTerm`, a Lambda-calculus term
- `NodeId`, a node in the term.

Because this is the most complex module of the library, it is useful to define some terms.
I will refer to these terms in the above comments for concision.

# 1. Term

A `Term` is a complete Lambda-calculus expression.

# 2. Node

A `Node` is what constitutes a `Term`. There are 3 types of Nodes.

2.1 There is `Lambda` Node, that has a single children
2.2 There is a `Variable` Node. It has no children and refers to a `Lambda` Node.
    We say the variable is `bound` to the corresponding Lambda.
2.3 There is an `Application` Node. It has 2 children.
    The first children is the function and the second children is the argument.

# 3. Subtrees

Each `Node` in the `Term` has a subtree. It is made up of itself, his children, his grandchildren ... and so on.
    The **strict** subtree of a node is its subtree but without the node itself.

# 4. Redex

A "Redex Node" is a `Node` in the `Term` that we can reduce. It must have the following properties:

4.1 A "Redex Node" is a `Node` of type "Application". The subtree of the "Redex node" is called simply the "Redex".
4.2 The first children of the Redex node must be a lambda. The trunk node is the child of the lambda node.
    The "Trunk" is defined as the strict subtree of the lambda node (i.e the subtree of the trunk).
4.3 The variables that are bound to the "trunk node" are called the "Stumps".
4.4 The second children of the Redex node can be anything. We call it the **Substitute node** of the Redex.
    The "Subsitute" is simply the subtree of the "substitute node" of the Redex.

# 5. Beta-reduction

The operation of "beta-reduction" transforms a Redex inside the Term, leaving the rest of the term unchanged.
    The beta-reduction of a Term on a Redex R consists in 2 steps:

5.1 Beta-reduction replaces all stumps by a copy of the substitute
5.1.1 If an application had a stump node as one of its children, the stump node is replaced by a copy of the substitute.
5.1.2 All variables in the substitute are now bound to the corresponding duplicated lambda in their duplicated substitute.
    But if a variable was bound to a lambda higher in the tree (not in the substitute), the bound remains
5.2 Beta-reduction removes the redex node and the lambda node
5.3 Beta-reduction removes the substitute (it has already been duplicated for each stump)
5.4 All we have left is the trunk, and it is placed where the original redex was.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, List, NewType, Optional

import polars as pl
from polars import Schema, UInt32

from .term import App, Lam, Term, Var

__all__ = ["AbstractTerm", "NodeId"]

# See `Node`
PREV_ID_SCHEMA = {"stump": UInt32, "local": UInt32}

SCHEMA = Schema(
    {"id": UInt32, "ref": UInt32, "arg": UInt32, "prev": pl.Struct(PREV_ID_SCHEMA)},
)


def _shift(nodes: pl.DataFrame, offset: int):
    return nodes.with_columns(
        pl.col("id") + offset,
        pl.col("ref") + offset,
        pl.col("arg") + offset,
        prev=None,
    )


NodeId = NewType("NodeId", int)


@dataclass
class Node[NodeId]:
    """
    Represents a node in the lambda calculus abstract syntax tree.

    Attributes:
        ref:
            The lambda this variable is bound to, or None if this node is not a variable.
        children:
            Tuple containing either:

            - Single id: child of lambda node
            - Two ids: function and argument of application node
            - Empty tuple: no children (in this case, this is a variable)

        previous:
            If this NodeId comes from a beta-reduction, corresponds to the id of the node in the term before reduction.
            If the node has been created by the reduction, `previous` is the id in the "Substitute" (see 5.1)

        previous_stump:
            If the node has been created by a beta-reduction, the `Stump` it originates from (see 4.3)
    """

    ref: Optional[NodeId]
    children: List[NodeId]
    previous_local: NodeId
    previous_stump: NodeId

    def get_arg(self) -> Optional[NodeId]:
        if len(self.children) == 2:
            return self.children[1]
        return None

    def get_left(self) -> Optional[NodeId]:
        if len(self.children) == 0:
            return None
        return self.children[0]

    def previous(self):
        return (self.previous_local, self.previous_stump)


class AbstractTerm:
    nodes: pl.DataFrame

    def __init__(self, nodes: pl.DataFrame | pl.LazyFrame):
        if hasattr(nodes, "match_to_schema"):
            nodes = nodes.match_to_schema(SCHEMA)
        if isinstance(nodes, pl.LazyFrame):
            self.nodes = nodes.collect()
        else:
            self.nodes = nodes

    def __eq__(self, other: object):
        if not isinstance(other, AbstractTerm):
            return False
        return self.nodes.select("arg", "ref").equals(other.nodes.select("arg", "ref"))

    def root(self) -> NodeId:
        return NodeId(0)

    def node(self, node_id: NodeId):
        """
        Get a node in the expression tree by id
        """

        id, ref, arg, prev = self.nodes.row(node_id)
        children = []
        if ref is None:
            children.append(id + 1)
        if arg is not None:
            children.append(arg)

        return Node(
            ref=ref,
            children=children,
            previous_local=prev["local"] if prev else None,
            previous_stump=prev["stump"] if prev else None,
        )

    def find_variables(self, lamb: NodeId) -> Generator[NodeId]:
        """
        Find all variables bound to a specific lambda.

        Args:
            lamb: the id of the lambda to consider
        """
        return self.nodes.filter(pl.col("ref") == lamb)["id"].__iter__()

    def find_redexes(self) -> Generator[NodeId]:
        """
        Find all candidate redex nodes (see 4.)

        The first redex must be the leftmost-outermost redex.
        """
        return self.nodes.filter(
            pl.col("arg").is_not_null(),
            pl.col("arg").shift(-1).is_null(),
            pl.col("ref").shift(-1).is_null(),
        )["id"].__iter__()

    def _get_subtree(self, root: NodeId) -> range:
        """
        Get the subtree of the node in a postfix left -> right order (see 3.)

        Args:
            root: the ancestor to consider
        """
        refs = self.nodes["ref"]
        args = self.nodes["arg"]
        rightmost = root
        while True:
            ref = refs[rightmost]
            if ref is not None:
                return range(NodeId(root), NodeId(rightmost + 1))

            arg = args[rightmost]
            rightmost = arg if arg is not None else rightmost + 1

    def get_subtree(self, root: NodeId) -> Generator[NodeId]:
        return (NodeId(x) for x in self._get_subtree(root))

    def __call__(self, other: AbstractTerm) -> AbstractTerm:
        r"""
        Compose this term with another one with an application.
        ```
         application
            /   \
           /     \
        self     other
        ```
        """
        n = len(self.nodes)
        return AbstractTerm(
            pl.concat(
                [
                    pl.DataFrame(
                        [{"id": 0, "arg": n + 1}],
                        schema=SCHEMA,
                    ),
                    _shift(self.nodes, 1),
                    _shift(other.nodes, n + 1),
                ],
            )
        )

    def beta_reduce(self, redex: NodeId) -> AbstractTerm:
        """
        Compute the beta-reduction of this term on this redex.
        """
        lamb = redex + 1
        a = redex + 2
        id_subst = self.node(redex).get_arg()
        assert id_subst is not None

        # see 4.3
        stumps = self.nodes.lazy().filter(pl.col("ref") == lamb).select("id")

        # see 4.4
        subst_range = self._get_subtree(id_subst)
        subst = self.nodes.lazy().filter(
            pl.col("id").is_between(subst_range.start, subst_range.stop, closed="left")
        )

        # we start duplicating
        new_trunks = subst.join(stumps, how="cross", suffix="_stump").select(
            "id",
            "id_stump",
            prev=pl.struct(stump="id_stump", local="id"),
            prev_arg=pl.struct(stump="id_stump", local="arg"),
            prev_ref=pl.struct(stump="id_stump", local="ref"),
            # 5.1.2
            prev_ref_unbound=pl.struct(stump=None, local="ref", schema=PREV_ID_SCHEMA),
        )

        new_nodes = (
            self.nodes.lazy()
            # remove the information about the previous iteration
            .select(pl.exclude("prev"))
            # See 5.2
            .filter(
                ~(pl.col("id").eq(redex) | pl.col("id").eq(lamb)),
            )
            # See 5.3
            .join(subst, left_on="id", right_on="id", how="anti")
            # See 5.4
            .with_columns(arg=pl.col("arg").replace(redex, a))
            .join(
                # 5.1.1 we check if the argument is a stump
                stumps.select("id", arg_is_stump=True),
                left_on="arg",
                right_on="id",
                how="left",
                maintain_order="left",
            )
            # See 5.1
            .join(
                new_trunks,
                left_on="id",
                right_on="id_stump",
                how="left",
                maintain_order="left",
            )
            .select(
                pl.col("prev").fill_null(pl.struct(stump=None, local="id")),
                pl.col("prev_ref").fill_null(pl.struct(stump=None, local="ref")),
                pl.col("prev_ref_unbound"),
                # 5.1.1
                pl.col("prev_arg").fill_null(
                    pl.when(pl.col("arg_is_stump"))
                    .then(pl.struct(stump="arg", local=id_subst))
                    .otherwise(pl.struct(stump=None, local="arg"))
                ),
            )
            .with_row_index("id")
        ).cache()

        # renumbering step: we replace each "previous location" with the new id
        out = (
            new_nodes.join(
                new_nodes.select(prev_ref="prev", ref="id"),
                on="prev_ref",
                how="left",
                maintain_order="left",
                nulls_equal=True,
            )
            .join(
                new_nodes.select(prev_ref_unbound="prev", ref_unbound="id"),
                on="prev_ref_unbound",
                how="left",
                maintain_order="left",
                nulls_equal=True,
            )
            .join(
                new_nodes.select(prev_arg="prev", arg="id"),
                on="prev_arg",
                how="left",
                maintain_order="left",
                nulls_equal=True,
            )
            .select(
                "id",
                ref=pl.coalesce("ref", "ref_unbound"),  # 5.1.2
                arg="arg",
                prev="prev",
            )
        ).collect()
        return AbstractTerm(out)

    @staticmethod
    def from_term(term: Term) -> AbstractTerm:
        """Convert a Term (De Bruijn) to AbstractTerm (DataFrame)."""
        rows = []
        visited = set()

        def build(t: Term, lambdas: List[int]) -> int:
            """Recursively build nodes. Returns the node id."""
            # Cycle detection
            term_id = id(t)
            if term_id in visited:
                raise ValueError(f"Cycle detected: term contains itself as a subterm")
            visited.add(term_id)

            node_id = len(rows)

            if isinstance(t, Var):
                # Variable: ref points to the lambda it's bound to
                ref = lambdas[-(t.index + 1)] if t.index < len(lambdas) else None
                rows.append({"id": node_id, "ref": ref, "arg": None, "prev": None})
            elif isinstance(t, Lam):
                # Lambda: ref=None, arg=None, child is implicitly at node_id+1
                rows.append({"id": node_id, "ref": None, "arg": None, "prev": None})
                build(t.body, lambdas + [node_id])
            elif isinstance(t, App):
                # Application: ref=None, arg points to argument, func is at node_id+1
                rows.append({"id": node_id, "ref": None, "arg": None, "prev": None})
                build(t.func, lambdas)
                arg_id = build(t.arg, lambdas)
                rows[node_id]["arg"] = arg_id

            visited.remove(term_id)
            return node_id

        build(term, [])
        return AbstractTerm(pl.DataFrame(rows, schema=SCHEMA))

    def to_term(self) -> Term:
        """Convert AbstractTerm (DataFrame) to Term (De Bruijn)."""

        def build(node_id: NodeId, lambdas: List[NodeId]) -> Term:
            """Recursively build term from node."""
            node = self.node(node_id)

            if node.ref is not None:
                # Variable: compute De Bruijn index from lambda list
                index = len(lambdas) - 1 - lambdas.index(node.ref)
                return Var(index)
            elif node.get_arg() is None:
                # Lambda: child is at node_id + 1
                return Lam(build(NodeId(node_id + 1), lambdas + [node_id]))
            else:
                # Application: func at node_id + 1, arg stored in node
                func = build(NodeId(node_id + 1), lambdas)
                arg_node = node.get_arg()
                assert arg_node is not None
                arg = build(arg_node, lambdas)
                return App(func, arg)

        return build(self.root(), [])
