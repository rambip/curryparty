from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, Optional, Union

import polars as pl
import svg
from polars import Schema, UInt32

SCHEMA = Schema(
    {
        "id": UInt32,
        "ref": UInt32,
        "arg": UInt32,
        "bid": pl.Struct({"major": UInt32, "minor": UInt32}),
    },
)


@dataclass
class BBox:
    x: Optional[tuple[int, int]]
    y: Optional[tuple[int, int]]

    def __or__(self: "BBox", other: "BBox") -> "BBox":
        def merge(a, b):
            if a is None:
                if b is None:
                    return None
                else:
                    return b
            else:
                if b is None:
                    return a
                else:
                    return (min(a[0], b[0]), max(a[1], b[1]))

        return BBox(merge(self.x, other.x), merge(self.y, other.y))

    def shift_x(self, offset: int) -> "BBox":
        if self.x is None:
            return BBox(self.x, self.y)
        else:
            return BBox((self.x[0] + offset, self.x[1] + offset), self.y)

    def shift_y(self, offset: int) -> "BBox":
        if self.y is None:
            return BBox(self.x, self.y)
        else:
            return BBox(self.x, (self.y[0] + offset, self.y[1] + offset))

    def copy(self):
        return BBox(self.x, self.y)


def V(name: str) -> "L":
    return L()._(name)


class L:
    def __init__(self, *lambda_names):
        self.n = len(lambda_names)
        self.lambdas = {x: i for (i, x) in enumerate(lambda_names)}
        self.refs = {}
        self.args = {}
        self.last_ = None

    def lamb(self, name: str) -> "L":
        self.n += 1
        self.lambdas[name] = self.n
        return self

    # TODO: rename
    def foo(self, args: dict, refs: dict, parent_lambdas: dict, offset: int):
        lambdas = parent_lambdas | self.lambdas
        for i, x in self.refs.items():
            if isinstance(x, str) and x in lambdas:
                refs[offset + i] = lambdas[x]
            else:
                refs[offset + i] = x

        for i, x in self.args.items():
            args[offset + i] = offset + x

    def _(self, x: Union[str, "L"]) -> "L":
        self.last_ = self.n
        if isinstance(x, L):
            x.foo(self.args, self.refs, self.lambdas, self.n)
            self.n += x.n
        else:
            assert isinstance(x, str)
            self.refs[self.n] = x
            self.n += 1
        return self

    def call(self, arg: Union[str, "L"]) -> "L":
        assert self.last_ is not None
        self.refs = {i + 1 if i >= self.last_ else i: x for (i, x) in self.refs.items()}
        self.args = {
            (i + 1 if i >= self.last_ else i): (x + 1 if i >= self.last_ else x)
            for (i, x) in self.args.items()
        }

        self.n += 1
        self.args[self.last_] = self.n
        if isinstance(arg, L):
            arg.foo(self.args, self.refs, self.lambdas, self.n)
            self.n += arg.n
        else:
            assert isinstance(arg, str), f"{type(arg)}"
            self.refs[self.n] = arg
            self.n += 1

        return self

    def build(self) -> "Term":
        nodes = [
            {
                "id": i,
                "ref": self.lambdas.get(self.refs.get(i, None), self.refs.get(i, None)),
                "arg": self.args.get(i, None),
            }
            for i in range(self.n)
        ]
        return Term(nodes)


class Term:
    def __init__(self, nodes):
        self.nodes = pl.DataFrame(nodes, schema=SCHEMA)
        if self.nodes["id"].count() == 0:
            self.nodes = self.nodes.select(pl.exclude("id")).with_row_index("id")

    def __call__(self, other: "Term") -> "Term":
        n = len(self.nodes)
        nodes = pl.concat(
            [
                pl.DataFrame([{"id": 0, "arg": n + 1}], schema=SCHEMA),
                self.nodes.with_columns(
                    pl.col("id") + 1,
                    pl.col("ref") + 1,
                    pl.col("arg") + 1,
                ),
                other.nodes.with_columns(
                    pl.col("id") + n + 1,
                    pl.col("ref") + n + 1,
                    pl.col("arg") + n + 1,
                ),
            ],
            how="vertical_relaxed",
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
        a = lamb + 1

        b_subtree = self.subtree(b)

        vars = self.nodes.filter(pl.col("ref") == lamb).select(
            subst_id="id", replaced=pl.lit(True)
        )

        b_subtree_duplicated = b_subtree.join(vars, how="cross", suffix="_var")
        rest_of_nodes = self.nodes.join(b_subtree, on="id", how="anti")

        new_nodes = (
            pl.concat(
                [b_subtree_duplicated, rest_of_nodes],
                how="diagonal_relaxed",
            )
            .join(vars, left_on="id", right_on="subst_id", how="anti")
            .join(vars, left_on="arg", right_on="subst_id", how="left", suffix="_arg")
            .filter(
                pl.col("id").ne(redex),
                pl.col("id").ne(lamb),
            )
            .select(
                bid=pl.struct(
                    major=pl.col("subst_id").fill_null(pl.col("id")), minor=pl.col("id")
                ),
                bid_ref=pl.struct(
                    major=pl.col("subst_id").fill_null(pl.col("ref")),
                    minor=pl.col("ref"),
                ),
                bid_arg=(
                    pl.struct(
                        major=pl.col("subst_id").fill_null(
                            pl.col("arg").replace(redex, a)
                        ),
                        minor=pl.when("replaced_arg")
                        .then(b)
                        .otherwise(pl.col("arg").replace(redex, a)),
                    )
                ),
            )
            .sort("bid")
            .with_row_index("id")
        )
        self.new_nodes = new_nodes

        new_ids = new_nodes.select("id", "bid")

        return (
            Term(
                new_nodes.join(
                    new_ids.rename({"id": "ref"}),
                    left_on="bid_ref",
                    right_on="bid",
                    how="left",
                )
                .join(
                    new_ids.rename({"id": "arg"}),
                    left_on="bid_arg",
                    right_on="bid",
                    how="left",
                )
                .select("id", "ref", "arg", "bid")
                .sort("id")
            )
        ), True

    def summary(self):
        try:
            bboxes = self.compute_bboxes()
            x = {i: b.x for (i, b) in bboxes.items()}
            y = {i: b.y for (i, b) in bboxes.items()}
            x_expr = pl.col("id").replace_strict(x, default=None)
            y_expr = pl.col("id").replace_strict(y, default=None)
        except (KeyError, AssertionError):
            x_expr = (None,)
            y_expr = None
        candidates = self.find_redexes()
        return candidates, self.nodes.with_columns(
            x=x_expr,
            y=y_expr,
        )

    def compute_bboxes(self) -> dict[int, BBox]:
        result = {0: BBox(None, (0, 0))}
        for row in self.nodes.iter_rows(named=True):
            if row["ref"] is not None:
                continue
            parent = row["id"]
            child = row["id"] + 1
            arg = row["arg"]
            if arg is not None:
                result[child] = result[parent].shift_y(
                    0 if self.nodes["arg"][child] is None else 1
                )
                result[arg] = result[parent].shift_y(0)
            else:
                result[child] = result[parent].shift_y(1)

        next_var = 0

        for row in self.nodes.sort("id", descending=True).iter_rows(named=True):
            parent = row["id"]
            ref = row["ref"]
            arg = row["arg"]
            if ref is not None:
                result[parent].x = (next_var, next_var)
                next_var -= 1
                result[ref] = result[parent] | result.get(ref, BBox(None, None))

            else:
                child = parent + 1
                result[parent].x = result[child].x
                result[parent] = result[child] | result[parent]

                if arg is not None:
                    result[parent] = result[arg] | result[parent]

        return result

    def _svg_blocks(self, last: Optional["Term"]) -> Iterable[svg.Element]:
        bboxes = self.compute_bboxes()
        if last is not None:
            last_bboxes = last.compute_bboxes()
        else:
            last_bboxes = None

        trajectories = {}

        for row in self.nodes.iter_rows(named=True):
            id = row["id"]
            src_id = row["bid"]["minor"] if row["bid"] else None
            if id is None:
                trajectories[id] = [bboxes[src_id], bboxes[src_id]]

            elif last_bboxes is None:
                trajectories[id] = [bboxes[id], bboxes[id]]
            else:
                trajectories[id] = [last_bboxes[src_id], bboxes[id]]

        for row in self.nodes.iter_rows(named=True):
            target_id = row["id"]
            src_id = row["bid"]["minor"] if row["bid"] else None
            arg = row["arg"]
            if target_id is None and last is not None:
                traj = [bboxes[src_id], bboxes[src_id]]
                fade = (1, 0)
            else:
                traj = trajectories[target_id]
                fade = (1, 1)

            ref = row["ref"]
            if ref is not None:
                yield svg.Rect(
                    width=0.8,
                    height=0.8,
                    fill="red",
                    elements=[
                        animate_bbox("x", traj),
                        animate_bbox("y", traj),
                        animate("opacity", fade),
                    ],
                )
                yield svg.Line(
                    stroke_width=0.05,
                    stroke="gray",
                    elements=[
                        animate("x1", [traj[0].x[0] + 0.5, traj[1].x[0] + 0.5]),
                        animate("y1", [traj[0].y[0] + 0.1, traj[1].y[0] + 0.1]),
                        animate("x2", [traj[0].x[0] + 0.5, traj[1].x[0] + 0.5]),
                        animate(
                            "y2",
                            [
                                trajectories[ref][0].y[0] + 0.9,
                                trajectories[ref][1].y[0] + 0.9,
                            ],
                        ),
                        animate("opacity", fade),
                    ],
                )
                continue

            if arg is not None:
                yield svg.Rect(
                    height=0.8,
                    fill_opacity=0.5,
                    stroke="orange",
                    stroke_width=0.1,
                    fill="none",
                    elements=[
                        animate_bbox("x", traj),
                        animate_bbox("y", traj),
                        animate_bbox("width", traj),
                        animate("opacity", fade),
                    ],
                )

                traj_arg = trajectories[arg]
                yield svg_left_arrow(
                    [t.x[0] for t in traj_arg],
                    [t.x[1] for t in traj],
                    [t.y[0] for t in traj_arg],
                    [t.y[0] for t in traj],
                )
                continue

            yield svg.Rect(
                height=0.8,
                fill="blue",
                elements=[
                    animate_bbox("x", traj),
                    animate_bbox("y", traj),
                    animate_bbox("width", traj),
                    animate("opacity", fade),
                ],
            )

    def _repr_html_(self):
        return self.display().as_str()

    def display(self, last: Optional["Term"] = None):
        return svg.SVG(
            xmlns="http://www.w3.org/2000/svg",
            viewBox="-10 0 20 10",  # type: ignore
            height="400px",  # type: ignore
            elements=list(self._svg_blocks(last)),
        )


def animate_bbox(name: str, bboxes: list, duration: int = 2, pad=0.1) -> svg.Element:
    """Create animation for a bounding box attribute (x, y, width, or height)"""
    bbox0, bbox1 = bboxes

    if name == "x":
        values = f"{pad + bbox0.x[0]};{pad + bbox1.x[0]}"
    elif name == "y":
        values = f"{pad + bbox0.y[0]};{pad + bbox1.y[0]}"
    elif name == "width":
        values = f"{1 - 2 * pad + bbox0.x[1] - bbox0.x[0]};{1 - 2 * pad + bbox1.x[1] - bbox1.x[0]}"
    elif name == "height":
        values = f"{1 - 2 * pad + bbox0.y[1] - bbox0.y[0]};{1 - 2 * pad + bbox1.y[1] - bbox1.y[0]}"
    else:
        raise ValueError(f"Unknown attribute: {name}")

    return svg.Animate(
        attributeName=name,
        values=values,
        dur=timedelta(seconds=duration),
        repeatCount="indefinite",
    )


def animate(name, values, duration=2):
    return svg.Animate(
        attributeName=name,
        values=f"{values[0]}; {values[1]}",
        dur=timedelta(seconds=duration),
        repeatCount="indefinite",
    )


def svg_left_arrow(x0_traj, x1_traj, y0_traj, y1_traj, s=0.1):
    line = svg.Line(
        stroke="black",
        stroke_width=0.05,
        x2=0.5,
        y2=0.5,
        elements=[
            animate("x1", [0.1 + a - b for a, b in zip(x0_traj, x1_traj)]),
            animate("y1", [0.5 + a - b for (a, b) in zip(y0_traj, y1_traj)]),
        ],
    )
    triangle = svg.Polygon(
        points=[
            0.5 + s,
            0.5 - s,
            0.5 - s,
            0.5,
            0.5 + s,
            0.5 + s,
        ],
        stroke_width=0,
        fill="black",
    )
    transform = svg.AnimateTransform(
        attributeName="transform",
        type="translate",
        values=" ; ".join([f"{x},{y}" for (x, y) in zip(x1_traj, y1_traj)]),
        dur=timedelta(seconds=2),
        repeatCount="indefinite",
    )
    return svg.G(elements=[line, triangle, transform])
