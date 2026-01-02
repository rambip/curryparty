from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable, Optional, Union

import polars as pl
import svg
from polars import Schema, String, UInt32

SCHEMA_NODES = Schema(
    {
        "id": UInt32,
        "ref": UInt32,
        "depth": UInt32,
        "bid": pl.Struct({"major": UInt32, "minor": UInt32}),
    },
)
SCHEMA_PARENT = Schema({"parent": UInt32, "id": UInt32, "type": String})


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
        self.parents = [(i, i + 1, "down") for i in range(len(lambda_names))]
        self.last_ = None

    def lamb(self, name: str) -> "L":
        self.parents.append((self.n, self.n + 1, "down"))
        self.lambdas[name] = self.n
        self.n += 1
        return self

    def foo(self, parent: list, refs: dict, parent_lambdas: dict, offset: int):
        lambdas = parent_lambdas | self.lambdas
        for a, b, c in self.parents:
            parent.append((a + offset, b + offset, c))

        for i, x in self.refs.items():
            if isinstance(x, str) and x in lambdas:
                refs[offset + i] = lambdas[x]
            else:
                refs[offset + i] = x

    def _(self, x: Union[str, "L"]) -> "L":
        self.last_ = self.n
        if isinstance(x, L):
            x.foo(self.parents, self.refs, self.lambdas, self.n)
            self.n += x.n
        else:
            assert isinstance(x, str)
            self.refs[self.n] = x
            self.n += 1
        return self

    def call(self, arg: Union[str, "L"]) -> "L":
        assert self.last_ is not None
        self.refs = {i + 1 if i >= self.last_ else i: x for (i, x) in self.refs.items()}
        self.parents = [
            (a + 1, b + 1, c) if a >= self.last_ else (a, b, c)
            for (a, b, c) in self.parents
        ]
        self.n += 1

        self.parents.append((self.last_, self.last_ + 1, "left"))
        self.parents.append((self.last_, self.n, "right"))
        if isinstance(arg, L):
            arg.foo(self.parents, self.refs, self.lambdas, self.n)
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
            }
            for i in range(self.n)
        ]
        parent = [{"parent": i, "id": c, "type": t} for (i, c, t) in self.parents]
        return Term(nodes, parent)


class Term:
    def __init__(self, nodes, parent):
        self.nodes = pl.DataFrame(nodes, schema=SCHEMA_NODES)
        if self.nodes["id"].count() == 0:
            self.nodes = self.nodes.select(pl.exclude("id")).with_row_index("id")

        self.parent = pl.DataFrame(parent, schema=SCHEMA_PARENT).sort("parent", "id")

        if self.nodes["depth"].count() == 0:
            self.nodes = self.with_depth()

    def __call__(self, other: "Term") -> "Term":
        n = len(self.nodes)
        nodes = pl.concat(
            [
                pl.DataFrame([{"id": 0, "depth": 0}], schema=SCHEMA_NODES),
                self.nodes.with_columns(
                    pl.col("id") + 1, pl.col("depth") + 1, pl.col("ref") + 1
                ),
                other.nodes.with_columns(
                    pl.col("id") + n + 1, pl.col("depth") + 1, pl.col("ref") + n + 1
                ),
            ],
            how="vertical_relaxed",
        )
        parent = pl.concat(
            [
                pl.DataFrame(
                    [
                        {"parent": 0, "id": 1, "type": "left"},
                        {"parent": 0, "id": n + 1, "type": "right"},
                    ]
                ),
                self.parent.with_columns(pl.col("parent") + 1, pl.col("id") + 1),
                other.parent.with_columns(
                    pl.col("parent") + n + 1, pl.col("id") + n + 1
                ),
            ],
            how="vertical_relaxed",
        )
        return Term(nodes, parent)

    def with_depth(self):
        d = {0: 0}
        for row in self.parent.sort("parent").iter_rows(named=True):
            parent = row["parent"]
            child = row["id"]
            d[child] = d[parent] + 1
        return self.nodes.with_columns(depth=pl.col("id").replace_strict(d))

    def find_redex(self):
        pivoted = self.parent.pivot("type", index="parent")
        applications = pivoted.filter(pl.col("left").is_not_null())
        lambdas = pivoted.filter(pl.col("down").is_not_null())

        candidates = (
            self.nodes.join(applications, left_on="id", right_on="parent")
            .join(lambdas, left_on="left", right_on="parent", suffix="_left")
            .select("id", "left", "right", "down_left", "depth")
        )

        return candidates.row(0) if len(candidates) > 0 else None

    def _beta(self) -> tuple["Term", bool]:
        nodes = self.nodes.drop_nulls("id")
        candidate = self.find_redex()
        if candidate is None:
            return self, False
        redex, lamb, b, a, redex_depth = candidate
        b_depth = redex_depth + 1

        b_subtree = (
            nodes.filter(pl.col("id") >= b)
            .filter(pl.col("depth").lt(b_depth).cum_sum().eq(0))
            .select(pl.exclude("bid"))
        )

        vars = nodes.filter(pl.col("ref") == lamb).select(
            subst_id="id", depth="depth", replaced=pl.lit(True)
        )

        b_subtree_duplicated = b_subtree.join(
            vars, how="cross", suffix="_var"
        ).with_columns(depth=pl.col("depth_var") + (pl.col("depth") - b_depth) - 2)

        rest_of_nodes = nodes.join(b_subtree, on="id", how="anti").with_columns()

        new_nodes = (
            pl.concat(
                [b_subtree_duplicated, rest_of_nodes],
                how="diagonal_relaxed",
            )
            .join(
                vars,
                left_on="id",
                right_on="subst_id",
                how="anti",
            )
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
                depth="depth",
            )
            .sort("bid")
            .with_row_index("id")
        )

        new_ids = new_nodes.select("id", "bid")

        df_out = (
            new_nodes.join(
                new_ids.rename({"id": "ref"}),
                left_on="bid_ref",
                right_on="bid",
                how="left",
            ).select("id", "ref", "depth", "bid")
        ).sort("id")

        parent2 = (
            self.parent.join(b_subtree, on="id", how="anti")
            .join(vars, left_on="id", right_on="subst_id", how="left")
            .select(
                "type",
                bid_parent=pl.struct(major=pl.col("parent"), minor=pl.col("parent")),
                bid=pl.struct(
                    major=pl.col("id").replace(redex, a),
                    minor=pl.when("replaced")
                    .then(b)
                    .otherwise(pl.col("id").replace(redex, a)),
                ),
            )
        )
        parent_duplicated = (
            self.parent.join(b_subtree, on="id")
            .join(vars, how="cross")
            .select(
                "type",
                bid_parent=pl.struct(major="subst_id", minor="parent"),
                bid=pl.struct(major="subst_id", minor="id"),
            )
        )
        parent_out = (
            pl.concat(
                [
                    parent_duplicated,
                    parent2,
                ],
                how="diagonal_relaxed",
            )
            .join(
                new_ids.rename({"id": "parent"}), left_on="bid_parent", right_on="bid"
            )
            .join(
                new_ids,
                on="bid",
            )
            .select("parent", "id", "type")
        )
        return Term(df_out, parent_out), True

    def compute_arity(self):
        return dict(self.parent.group_by(id="parent").len().iter_rows())

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
        arity = self.compute_arity()
        return self.nodes.join(self.parent, on="id", how="left").with_columns(
            x=x_expr,
            y=y_expr,
            arity=pl.col("id").replace_strict(arity, default=0),
        )

    def compute_bboxes(self) -> dict[int, BBox]:
        result = {0: BBox(None, (0, 0))}
        arities = self.compute_arity()
        for row in self.parent.sort("id").iter_rows(named=True):
            parent = row["parent"]
            child = row["id"]
            connection = row["type"]

            if connection == "right":
                result[child] = result[parent].copy()

            elif connection == "left" and arities.get(child, 0) != 2:
                result[child] = result[parent].copy()

            else:
                result[child] = result[parent].shift_y(1)

        next_var = 0

        for row in (
            self.parent.join(self.nodes, on="id")
            .sort(["id", "parent"], descending=True)
            .iter_rows(named=True)
        ):
            parent = row["parent"]
            child = row["id"]
            connection = row["type"]
            ref = row["ref"]
            if arities.get(child, 0) == 0:
                assert ref is not None
                result[child].x = (next_var, next_var)
                next_var -= 1

                result[ref] = result[child] | result.get(ref, BBox(None, None))

            # Update parent
            if connection == "left":
                if result[parent].x is None:
                    result[parent].x = result[child].x

            elif connection == "right":
                if result[parent].x is not None:
                    result[parent] = result[child] | result[parent]

            else:
                result[parent] = result[child] | result[parent]
        return result

    def _svg_blocks(self, last: Optional["Term"]) -> Iterable[svg.Element]:
        bboxes = self.compute_bboxes()
        if last is not None:
            arities_last = last.compute_arity()
            last_bboxes = last.compute_bboxes()
        else:
            arities_last = None
            last_bboxes = None

        arities = self.compute_arity()
        trajectories = {}

        for row in self.nodes.join(self.parent, on="id", how="left").iter_rows(
            named=True
        ):
            id = row["id"]
            src_id = row["bid"]["minor"] if row["bid"] else None
            if id is None:
                trajectories[id] = [bboxes[src_id], bboxes[src_id]]

            elif last_bboxes is None:
                trajectories[id] = [bboxes[id], bboxes[id]]
            else:
                trajectories[id] = [last_bboxes[src_id], bboxes[id]]

        for row in self.nodes.join(self.parent, on="id", how="left").iter_rows(
            named=True
        ):
            target_id = row["id"]
            connection = row["type"]
            parent = row["parent"]
            ref = row["ref"]
            src_id = row["bid"]["minor"] if row["bid"] else None

            if target_id is None and last is not None:
                traj = [bboxes[src_id], bboxes[src_id]]
                arity = arities_last.get(src_id, 0)
                fade = (1, 0)
            else:
                arity = arities.get(target_id, 0)
                traj = trajectories[target_id]
                fade = (1, 1)

            if connection == "right":
                traj_parent = trajectories[parent]
                assert parent is not None
                yield svg_left_arrow(
                    [t.x[0] for t in traj],
                    [t.x[1] for t in traj_parent],
                    [t.y[0] for t in traj],
                    [t.y[0] for t in traj_parent],
                )

            if arity == 1:  # Lambda
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
            elif arity == 0:  # Variable
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
            elif arity == 2:  # Application
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
