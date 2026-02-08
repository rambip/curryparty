from typing import Iterable, List, Optional, Tuple, TypeAlias

import svg

from .core import AbstractTerm, NodeId
from .utils import Interval, ShapeAnimFrame


def compute_height(term: AbstractTerm):
    _, y = compute_layout(term)
    return max(interval[1] for interval in y.values() if interval) + 1


def count_variables(term: AbstractTerm):
    return sum(1 for x in term.get_subtree(term.root()) if term.node(x).ref is not None)


Loc: TypeAlias = Tuple[NodeId, Optional[NodeId]]


def compute_layout(
    term: AbstractTerm, replaced_lambda: Optional[NodeId] = None, replaced_var_width=1
) -> tuple[dict[Loc, Interval], dict[Loc, Interval]]:
    """
    Compute the bounding-box of each node in the lambda-expression.

    Args:
        term: the term to compute the layout for
        replaced_lambda: a special lambda node. All variables that are bound to this lambda will have a special width
        replaced_var_width: the width of the variables that are bound to `replaced_lambda`
    """
    y: dict[Loc, Interval] = {(term.root(), None): Interval((0, 0))}
    x: dict[Loc, Interval] = {}
    nodes = list(term.get_subtree(term.root()))
    for node_id in nodes:
        node = term.node(node_id)
        ref = node.ref
        is_arg = node.get_arg() is not None
        for i, child in enumerate(node.children):
            offset = 1  # in most cases, the children is one step below the parent
            child_is_arg = term.node(child).get_arg() is not None
            if i == 1:
                # the right child (argument of application) is always put at the same lever
                offset = 0
            elif is_arg and not child_is_arg:
                # if child is a lambda or a variable, we can collapse
                offset = 0
            y[child, None] = y[node_id, None].shift(offset)

    next_var_x = count_variables(term) - 1

    for node_id in reversed(nodes):
        node = term.node(node_id)
        ref = node.ref
        arg = node.get_arg()
        children = node.children
        if len(children) == 0:
            # we place the variable in the next spot.
            # the variable is "replaced_var_width" units wide.
            width = replaced_var_width if ref == replaced_lambda else 1
            x[node_id, None] = Interval((next_var_x - width + 1, next_var_x))
            next_var_x -= width
        else:
            # we compute te bounding box of children
            for child in children:
                x[node_id, None] = x[child, None] | x.get(
                    (node_id, None), Interval(None)
                )

    # for applications, we use the bounding box of the left children:
    for node_id in reversed(nodes):
        node = term.node(node_id)
        arg = node.get_arg()
        if arg is not None:
            left = node.get_left()
            assert left is not None
            x[node_id, None] = x[left, None]

    return x, y


def draw_node(
    x: dict[Loc, Interval],
    y: dict[Loc, Interval],
    i_node: Loc,
    ref: Optional[Loc],
    arg: Optional[Loc],
    key: Loc,
    idx: int,
    replaced=False,
    removed=False,
) -> Iterable[ShapeAnimFrame]:
    """
    Generate shapes to display a node in the lambda-term.

    Args:
        x: the x bounding-boxes of all the nodes
        y: the y bounding-boxes of all the nodes
        i_node: the local identifier of the node to draw
        ref: if the node is a variable, the local identifier of the lambda it binds to
        arg: it the node is an application, the local identifier of its left argument
        key: the key that will be used for animation.  If a node is drawn in the next frame with the same key,
            the transition between the two will be displayed
        replaced: is the node going to disappear ?
        removed: has the node already disappear ?
    """
    x_node = x[i_node]
    y_node = y[i_node]
    if arg is not None or removed:
        color = "transparent"
    elif replaced or removed:
        color = "green"
    elif ref is not None:
        color = "red"
    else:
        color = "blue"

    r = svg.Rect(
        height=0.8,
        stroke_width=0.05,
        stroke="gray",
    )

    yield ShapeAnimFrame(
        element=r,
        key=("r", key),
        idx=idx,
        attrs={
            "x": 0.1 + x_node[0],
            "y": 0.1 + y_node[0] + (1 if replaced else 0),
            "width": 0.8 + x_node[1] - x_node[0],
            "fill_opacity": 1 if arg is None else 0,
            "fill": color,
        },
        zindex=0,
    )
    if arg is not None and not removed:
        r = svg.Rect(
            height=0.8,
            stroke_width=0.1,
            stroke="orange",
        )

        yield ShapeAnimFrame(
            element=r,
            key=("a", key),
            idx=idx,
            attrs={
                "x": 0.1 + x_node[0],
                "y": 0.1 + y_node[0],
                "width": 0.8 + x_node[1] - x_node[0],
                "fill_opacity": 1 if arg is None else 0,
                "fill": color,
            },
            zindex=1,
        )

    if ref is not None:
        y_ref = y[ref]
        e = svg.Line(
            stroke_width=0.2,
            stroke="gray",
        )
        yield ShapeAnimFrame(
            element=e,
            key=("l", key),
            idx=idx,
            attrs={
                "x1": x_node[0] + 0.5,
                "y1": y_ref[0] + 0.9,
                "x2": x_node[0] + 0.5,
                "y2": y_node[0] + 0.1 + (1 if replaced else 0),
                "stroke": "green" if replaced else "gray",
            },
            zindex=2,
        )

    if arg is not None:
        x_arg = x[arg]
        e1 = svg.Line(
            stroke="black",
            stroke_width=0.05,
        )
        e2 = svg.Circle(fill="black", r=0.1)
        if not removed:
            yield ShapeAnimFrame(
                element=e1,
                key=("b", key),
                idx=idx,
                attrs={
                    "x1": 0.5 + x_node[1],
                    "y1": 0.5 + y_node[0],
                    "x2": 0.5 + x_arg[0],
                    "y2": 0.5 + y_node[0],
                },
                zindex=3,
            )
            yield ShapeAnimFrame(
                element=e2,
                key=("c", key),
                idx=idx,
                attrs={
                    "cx": 0.5 + x_node[1],
                    "cy": 0.5 + y_node[0],
                },
                zindex=3,
            )


def compute_svg_frame_init(
    term: AbstractTerm, idx: int = 0
) -> Iterable[ShapeAnimFrame]:
    """
    Display the term before any reduction logic
    """
    x, y = compute_layout(term)
    for node_id in term.get_subtree(term.root()):
        node = term.node(node_id)
        ref = node.ref
        arg = node.get_arg()
        yield from draw_node(
            x,
            y,
            (node_id, None),
            (ref, None) if ref is not None else None,
            (arg, None) if arg is not None else None,
            key=(node_id, None),
            idx=idx,
        )


def compute_svg_frame_phase_a(
    term: AbstractTerm,
    redex: NodeId,
    b_subtree: List[NodeId],
    vars: List[NodeId],
    idx: int,
) -> Iterable[ShapeAnimFrame]:
    """
    Display the term with replaced variables (stumps) highlighted and the redex hidden.
    The blocks for variables are made wider to have room for replacement in the next step.
    """
    lamb = term.node(redex).get_left()
    assert lamb is not None
    b_width = sum(1 for x in b_subtree if term.node(x).ref is not None)
    x, y = compute_layout(term, replaced_lambda=lamb, replaced_var_width=b_width)
    for node_id in term.get_subtree(term.root()):
        node = term.node(node_id)
        ref = node.ref
        arg = node.get_arg()
        replaced = ref == lamb
        yield from draw_node(
            x,
            y,
            (node_id, None),
            (ref, None) if ref is not None else None,
            (arg, None) if arg is not None else None,
            key=(node_id, None),
            idx=idx,
            replaced=replaced,
            removed=(node_id == lamb or node_id == redex),
        )

    for stump in vars:
        for local_id in b_subtree:
            local_node = term.node(local_id)
            ref = local_node.ref
            arg = local_node.get_arg()
            key = (local_id, stump)
            yield from draw_node(
                x,
                y,
                (local_id, None),
                (ref, None) if ref is not None else None,
                (arg, None) if arg is not None else None,
                key=key,
                idx=idx,
            )


def compute_svg_frame_phase_b(
    term: AbstractTerm,
    redex: NodeId,
    b_subtree: List[NodeId],
    reduced: AbstractTerm,
    idx: int,
) -> Iterable[ShapeAnimFrame]:
    """
    Display the term with each stump replaced by the subtree.
    The layout is the same as the final layout, except it's translated.
    """
    lamb = term.node(redex).get_left()
    assert lamb is not None
    b_width = sum(1 for x in b_subtree if term.node(x).ref is not None)
    b = term.node(redex).get_arg()
    assert b is not None
    x, y = compute_layout(term, replaced_lambda=lamb, replaced_var_width=b_width)
    b_x = x[b, None][0]
    b_y = y[b, None][0]
    for node_id in reduced.get_subtree(reduced.root()):
        node = reduced.node(node_id)
        local = node.previous_local
        stump = node.previous_stump
        if stump is not None:
            delta_x = x[stump, None][0] - b_x
            delta_y = y[stump, None][0] - b_y + 1
            x[local, stump] = x[local, None].shift(delta_x)
            y[local, stump] = y[local, None].shift(delta_y)

    for node_id in reduced.get_subtree(reduced.root()):
        node = reduced.node(node_id)
        new_ref = node.ref
        new_arg = node.get_arg()

        key = node.previous()
        ref = reduced.node(new_ref).previous() if new_ref is not None else None
        arg = reduced.node(new_arg).previous() if new_arg is not None else None

        yield from draw_node(
            x,
            y,
            key,
            ref=ref,
            arg=arg,
            key=key,
            idx=idx,
        )


def compute_svg_frame_final(
    reduced: AbstractTerm, idx: int
) -> Iterable[ShapeAnimFrame]:
    """
    Display the reduced version of the term.
    """
    x, y = compute_layout(reduced)
    for node_id in reduced.get_subtree(reduced.root()):
        node = reduced.node(node_id)
        ref = node.ref
        arg = node.get_arg()
        key = node.previous()
        yield from draw_node(
            x,
            y,
            (node_id, None),
            (ref, None) if ref is not None else None,
            (arg, None) if arg is not None else None,
            key,
            idx=idx,
        )
