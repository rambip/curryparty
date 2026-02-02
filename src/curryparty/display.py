from typing import Any, Iterable, Optional, Union

import polars as pl
import svg

from .utils import Interval, ShapeAnimFrame


def compute_height(nodes: pl.DataFrame):
    _, y = compute_layout(nodes)
    return max(interval[1] for interval in y.values() if interval) + 1


def count_variables(nodes: pl.DataFrame):
    return nodes["ref"].count()


def compute_layout(
    nodes: pl.DataFrame, lamb=None, replaced_var_width=1
) -> tuple[dict[int, int], dict[int, int]]:
    y = {0: Interval((0, 0))}
    x = {}
    for node, ref, arg in nodes.select("id", "ref", "arg").iter_rows():
        if ref is not None:
            continue
        child = node + 1
        if arg is not None:
            y[child] = y[node].shift(0 if nodes["arg"][child] is None else 1)
            y[arg] = y[node].shift(0)
        else:
            y[child] = y[node].shift(1)

    next_var_x = count_variables(nodes) - 1

    for node, ref, arg in (
        nodes.sort("id", descending=True).select("id", "ref", "arg").iter_rows()
    ):
        if ref is not None:
            width = replaced_var_width if ref == lamb else 1
            x[node] = Interval((next_var_x - width + 1, next_var_x))
            next_var_x -= width
            x[ref] = x[node] | x.get(ref, Interval(None))

        else:
            child = node + 1
            x[node] = x[child] | x.get(node, Interval(None))
            y[node] = y[child] | y[node]
    return x, y


def draw(
    x: dict[Union[int, tuple[int, int]], Interval],
    y: dict[Union[int, tuple[int, int]], Interval],
    i_node: Union[int, tuple[int, int]],
    ref: Optional[int],
    arg: Optional[int],
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
    nodes: pl.DataFrame, idx: int = 0
) -> Iterable[ShapeAnimFrame]:
    x, y = compute_layout(nodes)
    for target_id, ref, arg in (
        nodes.select("id", "ref", "arg").sort("id", descending=True).iter_rows()
    ):
        yield from draw(x, y, target_id, ref, arg, key=target_id, idx=idx)


def compute_svg_frame_phase_a(
    nodes: pl.DataFrame, lamb: int, b_subtree: pl.DataFrame, vars: pl.Series, idx: int
) -> Iterable[ShapeAnimFrame]:
    redex = lamb - 1 if lamb is not None else None
    b_width = b_subtree.count()["ref"].item()
    x, y = compute_layout(nodes, lamb=lamb, replaced_var_width=b_width)
    for target_id, ref, arg in (
        nodes.select("id", "ref", "arg").sort("id", descending=True).iter_rows()
    ):
        replaced = ref is not None and ref == lamb
        yield from draw(
            x,
            y,
            target_id,
            ref,
            arg,
            key=target_id,
            idx=idx,
            replaced=replaced,
            removed=(target_id == lamb or target_id == redex),
        )

    for v in vars:
        for minor, ref, arg in (
            b_subtree.select("id", "ref", "arg").sort("id", descending=True).iter_rows()
        ):
            yield from draw(x, y, minor, ref, arg, key=(v, minor), idx=idx)


def compute_svg_frame_phase_b(
    nodes: pl.DataFrame,
    lamb: int,
    b_subtree: pl.DataFrame,
    new_nodes: pl.DataFrame,
    idx: int,
) -> Iterable[ShapeAnimFrame]:
    b_width = b_subtree.count()["ref"].item()
    b = b_subtree["id"][0]
    x, y = compute_layout(nodes, lamb=lamb, replaced_var_width=b_width)
    b_x = x[b][0]
    b_y = y[b][0]
    for bid, arg in new_nodes.select("bid", "arg").iter_rows():
        if bid["minor"] != bid["major"]:
            v = bid["major"]
            minor = bid["minor"]
            delta_x = x[v][0] - b_x
            delta_y = y[v][0] - b_y + 1
            x[(v, minor)] = x[minor].shift(delta_x)
            y[(v, minor)] = y[minor].shift(delta_y)

    for bid, new_ref, new_arg in new_nodes.select("bid", "ref", "arg").iter_rows():
        v = bid["major"]
        minor = bid["minor"]
        if new_ref is None:
            ref = None
        else:
            bid_ref = new_nodes["bid"][new_ref]
            ref = (
                (bid_ref["major"], bid_ref["minor"])
                if bid_ref["major"] != bid_ref["minor"]
                else bid_ref["minor"]
            )
        if new_arg is None:
            arg = None
        else:
            bid_arg = new_nodes["bid"][new_arg]
            arg = (
                (bid_arg["major"], bid_arg["minor"])
                if bid_arg["major"] != bid_arg["minor"]
                else bid_arg["minor"]
            )
        key = (v, minor) if minor != v else minor
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
    reduced: pl.DataFrame, idx: int
) -> Iterable[ShapeAnimFrame]:
    x, y = compute_layout(reduced)
    for target_id, bid, ref, arg in (
        reduced.select("id", "bid", "ref", "arg")
        .sort("id", descending=True)
        .iter_rows()
    ):
        minor = bid["minor"]
        major = bid["major"]
        key = (major, minor) if minor != major else minor
        yield from draw(x, y, target_id, ref, arg, key, idx=idx)
