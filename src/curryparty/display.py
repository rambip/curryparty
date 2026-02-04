from typing import Any, Iterable, List, Optional

import svg

from .core import AbstractTerm, NodeId
from .utils import Interval, ShapeAnimFrame


def compute_height(term: AbstractTerm):
    _, y = compute_layout(term)
    return max(interval[1] for interval in y.values() if interval) + 1


def count_variables(term: AbstractTerm):
    # TODO: more efficient
    return sum(1 for x in term.get_subtree(term.root()) if term.node(x).ref is not None)


def compute_layout(
    term: AbstractTerm, lamb: Optional[NodeId] = None, replaced_var_width=1
) -> tuple[dict[NodeId, Interval], dict[NodeId, Interval]]:
    y = {term.root(): Interval((0, 0))}
    x = {}
    nodes = list(term.get_subtree(term.root()))
    for node_id in nodes:
        node = term.node(node_id)
        ref = node.ref
        arg = node.get_arg()
        if ref is not None:
            continue
        child = term.node(node_id).get_left()
        assert child is not None
        if arg is not None:
            y[child] = y[node_id].shift(0 if term.node(child).get_arg() is None else 1)
            y[arg] = y[node_id].shift(0)
        else:
            y[child] = y[node_id].shift(1)

    next_var_x = count_variables(term) - 1

    for node_id in reversed(nodes):
        node = term.node(node_id)
        ref = node.ref
        arg = node.get_arg()
        if ref is not None:
            width = replaced_var_width if ref == lamb else 1
            x[node_id] = Interval((next_var_x - width + 1, next_var_x))
            next_var_x -= width
            x[ref] = x[node_id] | x.get(ref, Interval(None))

        else:
            child = term.node(node_id).get_left()
            assert child is not None
            x[node_id] = x[child] | x.get(node_id, Interval(None))
            y[node_id] = y[child] | y[node_id]
    return x, y


def draw(
    x: dict[NodeId, Interval],
    y: dict[NodeId, Interval],
    i_node: NodeId,
    ref: Optional[NodeId],
    arg: Optional[NodeId],
    key: Any,
    idx: int,
    replaced=False,
    removed=False,
    hide_arg=False,
) -> Iterable[ShapeAnimFrame]:
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
    if arg is not None and not hide_arg:
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
    x, y = compute_layout(term)
    for node_id in term.get_subtree(term.root()):
        node = term.node(node_id)
        ref = node.ref
        arg = node.get_arg()
        yield from draw(x, y, node_id, ref, arg, key=node_id, idx=idx)


def compute_svg_frame_phase_a(
    term: AbstractTerm,
    redex: NodeId,
    b_subtree: List[NodeId],
    vars: List[NodeId],
    idx: int,
) -> Iterable[ShapeAnimFrame]:
    lamb = term.node(redex).get_left()
    assert lamb is not None
    b_width = sum(1 for x in b_subtree if term.node(x).ref is not None)
    x, y = compute_layout(term, lamb=lamb, replaced_var_width=b_width)
    for node_id in term.get_subtree(term.root()):
        if node_id in b_subtree:
            continue
        node = term.node(node_id)
        ref = node.ref
        arg = node.get_arg()
        replaced = ref == lamb
        yield from draw(
            x,
            y,
            node_id,
            ref,
            arg,
            key=node_id,
            idx=idx,
            replaced=replaced,
            removed=(node_id == lamb or node_id == redex),
        )

    for stump in vars:
        for local_id in b_subtree:
            local_node = term.node(local_id)
            ref = local_node.ref
            arg = local_node.get_arg()
            key = (stump, local_id)
            yield from draw(x, y, local_id, ref, arg, key=key, idx=idx)


def compute_svg_frame_phase_b(
    term: AbstractTerm,
    redex: NodeId,
    b_subtree: List[NodeId],
    reduced: AbstractTerm,
    idx: int,
) -> Iterable[ShapeAnimFrame]:
    lamb = term.node(redex).get_left()
    assert lamb is not None
    b_width = sum(1 for x in b_subtree if term.node(x).ref is not None)
    b = term.node(redex).get_arg()
    assert b is not None
    x, y = compute_layout(term, lamb=lamb, replaced_var_width=b_width)
    b_x = x[b][0]
    b_y = y[b][0]
    for node_id in reduced.get_subtree(reduced.root()):
        node = reduced.node(node_id)
        previous = node.previous
        stump = node.previous_stump
        if stump is not None:
            delta_x = x[stump][0] - b_x
            delta_y = y[stump][0] - b_y + 1
            x[(stump, previous)] = x[previous].shift(delta_x)
            y[(stump, previous)] = y[previous].shift(delta_y)

    for node_id in reduced.get_subtree(reduced.root()):
        node = reduced.node(node_id)
        new_ref = node.ref
        new_arg = node.get_arg()
        stump = node.previous_stump
        previous = node.previous

        key = (stump, previous) if stump is not None else previous

        if new_ref is None:
            ref = None
        else:
            node_ref = reduced.node(new_ref)
            stump_ref = node_ref.previous_stump
            previous_ref = node_ref.previous
            ref = (stump_ref, previous_ref) if stump_ref is not None else previous_ref
        if new_arg is None:
            arg = None
        else:
            node_arg = reduced.node(new_arg)
            stump_arg = node_arg.previous_stump
            previous_arg = node_arg.previous

            arg = (stump_arg, previous_arg) if stump_arg is not None else previous_arg
        yield from draw(
            x,
            y,
            key,
            ref,
            arg,
            key=key,
            idx=idx,
        )


def compute_svg_frame_final(
    reduced: AbstractTerm, idx: int
) -> Iterable[ShapeAnimFrame]:
    x, y = compute_layout(reduced)
    for node_id in reduced.get_subtree(reduced.root()):
        node = reduced.node(node_id)
        ref = node.ref
        arg = node.get_arg()
        stump = node.previous_stump
        previous = node.previous
        key = (stump, previous) if stump is not None else previous
        yield from draw(x, y, node_id, ref, arg, key, idx=idx)
