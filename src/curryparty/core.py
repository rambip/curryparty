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
5.2 Beta-reduction remove the redex node, the lambda node and the substitute.
    All we have left is the trunk, and it is placed where the original redex was.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, List, NewType, Optional

import polars as pl
from polars import Schema, UInt32

__all__ = ["AbstractTerm", "NodeId"]

SCHEMA = Schema(
    {
        "id": UInt32,
        "ref": UInt32,
        "arg": UInt32,
        "bid": pl.Struct({"stump": UInt32, "local": UInt32}),
    },
)


def _shift(nodes: pl.DataFrame, offset: int):
    return nodes.with_columns(
        pl.col("id") + offset,
        pl.col("ref") + offset,
        pl.col("arg") + offset,
        bid=None,
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
    previous: NodeId
    previous_stump: NodeId

    def get_arg(self) -> Optional[NodeId]:
        if len(self.children) == 2:
            return self.children[1]
        return None

    def get_left(self) -> Optional[NodeId]:
        if len(self.children) == 0:
            return None
        return self.children[0]


def _generate_bi_identifier(
    stump_name: str, local_name: str, local_replacement=pl.lit(None)
):
    return pl.struct(
        stump=pl.col(stump_name).fill_null(pl.col(local_name)),
        local=local_replacement.fill_null(pl.col(local_name)),
    )


class AbstractTerm:
    nodes: pl.DataFrame

    def __init__(self, nodes: pl.DataFrame | pl.LazyFrame):
        nodes = nodes.match_to_schema(SCHEMA)
        if isinstance(nodes, pl.LazyFrame):
            self.nodes = nodes.collect()
        else:
            self.nodes = nodes

    def root(self) -> NodeId:
        return NodeId(0)

    def node(self, node_id: NodeId):
        """
        Get a node in the expression tree by id
        """

        id, ref, arg, bid = self.nodes.row(node_id)
        children = []
        if ref is None:
            children.append(id + 1)
        if arg is not None:
            children.append(arg)

        return Node(
            ref=ref,
            children=children,
            previous=bid["local"] if bid is not None else None,
            previous_stump=bid["stump"]
            # FIXME: use join on nulls
            if bid is not None and bid["stump"] != bid["local"]
            else None,
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

        stumps = (
            self.nodes.lazy().filter(pl.col("ref") == lamb).select("id", replaced=True)
        )

        b_subtree_range = self._get_subtree(id_subst)
        subst = self.nodes.lazy().filter(
            pl.col("id").is_between(
                b_subtree_range.start, b_subtree_range.stop, closed="left"
            )
        )

        subst_duplicated = subst.join(stumps, how="cross", suffix="_stump")
        rest_of_nodes = (
            self.nodes.lazy()
            .join(subst, on="id", how="anti")
            .with_columns(arg=pl.col("arg").replace(redex, a))
        )

        new_nodes = (
            pl.concat(
                [subst_duplicated, rest_of_nodes],
                how="diagonal",
            )
            .join(stumps, left_on="id", right_on="id", how="anti")
            .join(stumps, left_on="arg", right_on="id", how="left", suffix="_arg")
            .join(stumps, left_on="ref", right_on="id", how="left", suffix="_ref")
            .filter(
                ~(pl.col("id").eq(redex) | pl.col("id").eq(lamb)),
            )
            .select(
                bid=_generate_bi_identifier("id_stump", "id"),
                bid_ref=_generate_bi_identifier("id_stump", "ref"),
                bid_ref_fallback=pl.struct(stump="ref", local="ref"),
                bid_arg=_generate_bi_identifier(
                    # TODO: document and simplify to avoid "minor_replacement"
                    "id_stump",
                    "arg",
                    local_replacement=pl.when("replaced_arg").then(id_subst),
                ),
            )
            .sort("bid")
            .with_row_index("id")
        ).cache()

        out = (
            new_nodes.join(
                new_nodes.select(bid_ref="bid", ref="id"),
                on="bid_ref",
                how="left",
                maintain_order="left",
            )
            .join(
                new_nodes.select(bid_ref_fallback="bid", ref_fallback="id"),
                on="bid_ref_fallback",
                how="left",
                maintain_order="left",
            )
            .join(
                new_nodes.select(bid_arg="bid", arg="id"),
                on="bid_arg",
                how="left",
                maintain_order="left",
            )
            .select("id", ref=pl.coalesce("ref", "ref_fallback"), arg="arg", bid="bid")
        ).collect()
        return AbstractTerm(out)
