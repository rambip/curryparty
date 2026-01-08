from typing import Any, Iterable, Optional, Union

import polars as pl
import svg

from .utils import Interval


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

    next_var_x = nodes.select(pl.col("ref").count()).item()

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

def compute_height(nodes: pl.DataFrame):
    _, y = compute_layout(nodes)
    return max(interval[1] for interval in y.values() if interval)


def draw(
    x: dict[Union[int, tuple[int, int]], Interval],
    y: dict[Union[int, tuple[int, int]], Interval],
    i_node: Union[int, tuple[int, int]],
    ref: Optional[int],
    arg: Optional[int],
    key: Any,
    replaced=False,
    removed=False,
) -> Iterable[tuple[Any, svg.Element, dict]]:
    x_node = x[i_node]
    y_node = y[i_node]
    if True:
        if arg is not None or removed:
            color = "transparent"
        elif replaced or removed:
            color = "green"
        elif ref is not None:
            color = "red"
        else:
            color = "blue"

        stroke_width = 0.05
        stroke = "gray"
        if arg is not None:
            stroke_width = 0.1
            stroke = "orange"
        r = svg.Rect(
            height=0.8,
            stroke_width=stroke_width,
            stroke=stroke,
        )

        yield (
            ("r", key),
            r,
            {
                "x": 0.1 + x_node[0],
                "y": 0.1 + y_node[0],
                "width": 0.8 + x_node[1] - x_node[0],
                "fill_opacity": 1 if arg is None else 0,
                "fill": color,
            },
        )

    if ref is not None:
        y_ref = y[ref]
        e = svg.Line(
            stroke_width=0.2,
            stroke="gray",
        )
        yield (
            ("l", key),
            e,
            {
                "x1": x_node[0] + 0.5,
                "y1": y_node[0] + 0.1,
                "x2": x_node[0] + 0.5,
                "y2": y_ref[0] + 0.9,
                "stroke": "green" if replaced else "gray",
            },
        )

    if arg is not None:
        x_arg = x[arg]
        y_arg = y[arg]
        e1 = svg.Line(
            stroke="black",
            stroke_width=0.05,
        )
        e2 = svg.Circle(fill="black", r=0.1)
        if not removed:
            yield (
                ("b", key),
                e1,
                {
                    "x1": 0.5 + x_node[1],
                    "y1": 0.5 + y_node[0],
                    "x2": 0.5 + x_arg[0],
                    "y2": 0.5 + y_arg[0],
                },
            )
            yield (
                ("c", key),
                e2,
                {
                    "cx": 0.5 + x_node[1],
                    "cy": 0.5 + y_node[0],
                },
            )


def compute_svg_frame_init(
    nodes: pl.DataFrame,
) -> Iterable[tuple[Any, svg.Element, dict[str, Any]]]:
    x, y = compute_layout(nodes)
    for target_id, ref, arg in (
        nodes.select("id", "ref", "arg").sort("id", descending=True).iter_rows()
    ):
        yield from draw(x, y, target_id, ref, arg, target_id)


def compute_svg_frame_phase_a(
    nodes: pl.DataFrame,
    lamb: int,
    b_subtree: pl.DataFrame,
    vars: pl.Series,
):
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
            target_id,
            replaced=replaced,
            removed=(target_id == lamb or target_id == redex),
        )

    for v in vars:
        for minor, ref, arg in (
            b_subtree.select("id", "ref", "arg").sort("id", descending=True).iter_rows()
        ):
            yield from draw(x, y, minor, ref, arg, key=(v, minor))


def compute_svg_frame_phase_b(
    nodes: pl.DataFrame,
    lamb: int,
    b_subtree: pl.DataFrame,
    new_nodes: pl.DataFrame,
):
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
            delta_y = y[v][0] - b_y
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
            if bid_arg["minor"] == b:
                arg = bid_arg["major"]
        key = (v, minor) if minor != v else minor
        yield from draw(x, y, key, ref, arg, key=key)


def compute_svg_frame_final(reduced: pl.DataFrame):
    x, y = compute_layout(reduced)
    for target_id, bid, ref, arg in (
        reduced.select("id", "bid", "ref", "arg")
        .sort("id", descending=True)
        .iter_rows()
    ):
        minor = bid["minor"]
        major = bid["major"]
        key = (major, minor) if minor != major else minor
        yield from draw(x, y, target_id, ref, arg, key)
