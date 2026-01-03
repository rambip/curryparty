from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterable, List, Optional, Union

import polars as pl
import svg
from polars import Schema, UInt32

__all__ = ["L", "V"]

SCHEMA = Schema(
    {
        "id": UInt32,
        "ref": UInt32,
        "arg": UInt32,
        "bid": pl.Struct({"major": UInt32, "minor": UInt32}),
    },
)


@dataclass
class Interval:
    values: Optional[tuple[int, int]]

    def __or__(self: "Interval", other: "Interval") -> "Interval":
        if self.values is None:
            if other.values is None:
                return Interval(None)
            else:
                return other
        else:
            if other.values is None:
                return self
        return Interval(
            (min(self.values[0], other.values[0]), max(self.values[1], other.values[1]))
        )

    def __getitem__(self, index):
        assert self.values is not None, "interval is empty"
        return self.values[index]

    def shift(self, offset: int) -> "Interval":
        if self.values is None:
            return Interval(None)
        else:
            return Interval((self.values[0] + offset, self.values[1] + offset))

    def copy(self):
        return Interval(self.values)


class L:
    def __init__(self, *lambda_names):
        self.n = len(lambda_names)
        self.lambdas = {x: i for (i, x) in enumerate(lambda_names)}
        self.refs: List[tuple[int, Union[int, str]]] = []
        self.args = []
        self.last_ = None

    def lamb(self, name: str) -> "L":
        self.n += 1
        self.lambdas[name] = self.n
        return self

    def _append_subtree_or_subexpression(self, t: Union[str, "L"]):
        if isinstance(t, L):
            offset = self.n
            for i, x in t.refs:
                self.refs.append((offset + i, t.lambdas.get(x, x)))

            for i, x in t.args:
                self.args.append((offset + i, offset + x))
            self.n += t.n
        else:
            assert isinstance(t, str)
            self.refs.append((self.n, t))
            self.n += 1

    def _(self, x: Union[str, "L"]) -> "L":
        self.last_ = self.n
        self._append_subtree_or_subexpression(x)
        return self

    def call(self, arg: Union[str, "L"]) -> "L":
        assert self.last_ is not None
        self.refs = [(i + 1, x) if i >= self.last_ else (i, x) for (i, x) in self.refs]
        self.args = [
            (i + 1, x + 1) if i >= self.last_ else (i, x) for (i, x) in self.args
        ]

        self.n += 1
        self.args.append((self.last_, self.n))
        self._append_subtree_or_subexpression(arg)

        return self

    def build(self) -> "Term":
        self.refs = [(i, self.lambdas.get(x, x)) for i, x in self.refs]
        ref = pl.from_records(
            self.refs, orient="row", schema={"id": pl.UInt32, "ref": pl.UInt32}
        )
        arg = pl.from_records(
            self.args, orient="row", schema={"id": pl.UInt32, "arg": pl.UInt32}
        )
        data = (
            pl.Series("id", range(self.n), dtype=pl.UInt32)
            .to_frame()
            .join(ref, on="id", how="left")
            .join(arg, on="id", how="left")
        ).with_columns(bid=pl.struct(major="id", minor="id"))
        return Term(data)


def V(name: str) -> L:
    return L()._(name)


class Term:
    def __init__(self, nodes: pl.DataFrame):
        assert nodes.schema == SCHEMA, (
            f"{nodes.schema} is different from expected {SCHEMA}"
        )
        self.nodes = nodes

    def _shift(self, offset=1):
        return self.nodes.with_columns(
            pl.col("id") + offset,
            pl.col("ref") + offset,
            pl.col("arg") + offset,
            bid=None,
        )

    def __call__(self, other: "Term") -> "Term":
        n = len(self.nodes)
        nodes = pl.concat(
            [
                pl.DataFrame(
                    [{"id": 0, "arg": n + 1}],
                    schema=SCHEMA,
                ),
                self._shift(1),
                other._shift(n + 1),
            ],
        )
        return Term(nodes)

    def subtree(self, node: int) -> pl.DataFrame:
        rightmost = node
        while True:
            ref = self.nodes["ref"][rightmost]
            arg = self.nodes["arg"][rightmost]
            if ref is not None:
                return self.nodes.filter(pl.col("id").is_between(node, rightmost))

            rightmost = arg if arg is not None else rightmost + 1

    def find_redexes(self):
        parents = self.nodes.filter(pl.col("ref").is_null())
        return (
            parents.join(
                self.nodes, left_on="id", right_on=pl.col("id") - 1, suffix="_child"
            )
            .filter(
                pl.col("arg").is_not_null(),
                pl.col("ref_child").is_null(),
                pl.col("arg_child").is_null(),
            )
            .select(redex="id", lamb="id_child", b="arg")
        )

    def _beta(self) -> tuple["Term", bool]:
        candidates = self.find_redexes()
        if len(candidates) == 0:
            return self, False
        redex, lamb, b = candidates.row(0)
        self.b = b
        a = lamb + 1

        vars = self.nodes.filter(pl.col("ref") == lamb).select("id", replaced=True)
        b_subtree = self.subtree(b)

        b_subtree_duplicated = b_subtree.join(vars, how="cross", suffix="_major")
        rest_of_nodes = self.nodes.join(b_subtree, on="id", how="anti").with_columns(
            arg=pl.col("arg").replace(redex, a)
        )

        def generate_bi_identifier(
            major_name: str, minor_name: str, minor_replacement=pl.lit(None)
        ):
            return pl.struct(
                major=pl.col(major_name).fill_null(pl.col(minor_name)),
                minor=minor_replacement.fill_null(pl.col(minor_name)),
            )

        new_nodes = (
            pl.concat(
                [b_subtree_duplicated, rest_of_nodes],
                how="diagonal_relaxed",
            )
            .join(vars, left_on="id", right_on="id", how="anti")
            .join(vars, left_on="arg", right_on="id", how="left", suffix="_arg")
            .filter(
                ~pl.col("id").is_between(redex, lamb),
            )
            .select(
                bid=generate_bi_identifier("id_major", "id"),
                bid_ref=generate_bi_identifier("id_major", "ref"),
                bid_arg=generate_bi_identifier(
                    "id_major", "arg", pl.when("replaced_arg").then(b)
                ),
            )
            .sort("bid")
            .with_row_index("id")
        )

        out = Term(
            new_nodes.join(
                new_nodes.select(bid_ref="bid", ref="id"),
                on="bid_ref",
                how="left",
            )
            .join(
                new_nodes.select(bid_arg="bid", arg="id"),
                on="bid_arg",
                how="left",
            )
            .select("id", "ref", "arg", "bid")
            .sort("id")
        )

        return out, True

    def _frame(
        self, t: int, final: Optional["Term"] = None
    ) -> Iterable[tuple[Any, svg.Element, dict[str, int]]]:
        y = {0: Interval((t, t))}
        x = {}
        for node, ref, arg in self.nodes.select("id", "ref", "arg").iter_rows():
            if ref is not None:
                continue
            child = node + 1
            if arg is not None:
                y[child] = y[node].shift(0 if self.nodes["arg"][child] is None else 1)
                y[arg] = y[node].shift(0)
            else:
                y[child] = y[node].shift(1)

        next_var = 0

        for node, ref, arg in (
            self.nodes.sort("id", descending=True)
            .select("id", "ref", "arg")
            .iter_rows()
        ):
            if ref is not None:
                x[node] = Interval((next_var, next_var))
                next_var -= 1
                x[ref] = x[node] | x.get(ref, Interval(None))

            else:
                child = node + 1
                x[node] = x[child] | x.get(node, Interval(None))
                y[node] = y[child] | y[node]

        def rect(is_arg: bool, is_ref: bool):
            stroke_width = 0.05
            stroke = "black"
            if is_arg:
                stroke_width = 0.1
                fill = "transparent"
                stroke = "orange"
            elif is_ref:
                fill = "red"
            else:
                fill = "blue"
            return svg.Rect(
                width=0.8,
                height=0.8,
                fill=fill,
                stroke_width=stroke_width,
                stroke=stroke,
            )

        for target_id, bid, ref, arg in (
            self.nodes.select("id", "bid", "ref", "arg")
            .sort("id", descending=True)
            .iter_rows()
        ):
            x_node = x[target_id]
            y_node = y[target_id]

            r = rect(arg is not None, ref is not None)
            yield (
                ("r", target_id),
                r,
                {
                    "x": 0.1 + x_node[0],
                    "y": 0.1 + y_node[0],
                    "width": 0.8 + x_node[1] - x_node[0],
                    "fill_opacity": 1 if arg is None else 0,
                },
            )

            if ref is not None:
                x_ref = x[ref]
                y_ref = y[ref]
                e = svg.Line(
                    stroke_width=0.1,
                    stroke="gray",
                )
                yield (
                    ("l", target_id),
                    e,
                    {
                        "x1": x_node[0] + 0.5,
                        "y1": y_node[0] + 0.1,
                        "x2": x_node[0] + 0.5,
                        "y2": y_ref[0] + 0.9,
                    },
                )
                continue

            if arg is not None:
                pass
                # traj_arg = trajectories[arg]
                # yield svg_left_arrow(
                #     [x[0] for x in traj_arg["x"]],
                #     [y[0] for y in traj["y"]],
                #     [x[1] for x in traj["x"]],
                #     [y[0] for y in traj["y"]],
                # )
                # continue

    def show_reduction(self):
        final = self._beta()[0]
        shapes: dict[int, ShapeAnim] = {}
        for t in range(5):
            for k, e, attributes in self._frame(t, final):
                if t == 0:
                    shapes[k] = ShapeAnim(e, attributes.items())
                else:
                    shapes[k].append_frame(attributes.items())

        elements = [x.to_element() for x in shapes.values()]
        return Html(
            svg.SVG(
                xmlns="http://www.w3.org/2000/svg",
                viewBox="-10 0 20 10",  # type: ignore
                height="400px",  # type: ignore
                elements=elements,
            ).as_str()
        )

    def _repr_html_(self):
        frame = self._frame(0)
        elements = []
        for _, e, attributes in frame:
            for name, v in attributes.items():
                e.__setattr__(name, v)
            elements.append(e)
        return svg.SVG(
            xmlns="http://www.w3.org/2000/svg",
            viewBox="-10 0 20 10",  # type: ignore
            height="400px",  # type: ignore
            elements=elements,
        ).as_str()


class Html:
    def __init__(self, content: str):
        self.content = content

    def _repr_html_(self):
        return self.content


@dataclass
class ShapeAnim:
    shape: svg.Element
    attributes: dict[str, list]

    def __init__(self, shape: svg.Element, attributes: Iterable[tuple[str, float]]):
        self.shape = shape
        self.attributes = {}
        for name, v in attributes:
            self.attributes[name] = [v]

    def append_frame(self, attributes: Iterable[tuple[str, float]]):
        for name, v in attributes:
            self.attributes[name].append(v)

    def to_element(self):
        self.shape.elements = [
            svg.Animate(
                attributeName=name,
                values=";".join(str(v) for v in values),
                dur=timedelta(seconds=4),
                repeatCount="indefinite",
            )
            for name, values in self.attributes.items()
        ]
        return self.shape
