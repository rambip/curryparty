from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Iterable, Literal, Optional

import polars as pl
import svg
from polars import Schema, String, UInt32

SCHEMA_NODES = Schema(
    {"id": UInt32, "ref": UInt32, "depth": UInt32, "src_id": UInt32},
)
SCHEMA_CHILDREN = Schema({"id": UInt32, "child": UInt32, "type": String})


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


class Term:
    def __init__(self, nodes, children):
        self.nodes = pl.DataFrame(nodes, schema=SCHEMA_NODES)
        if self.nodes["id"].count() == 0:
            self.nodes = self.nodes.select(pl.exclude("id")).with_row_index("id")

        self.children = pl.DataFrame(children, schema=SCHEMA_CHILDREN)

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
        children = pl.concat(
            [
                pl.DataFrame(
                    [
                        {"id": 0, "child": 1, "type": "left"},
                        {"id": 0, "child": n + 1, "type": "right"},
                    ]
                ),
                self.children.with_columns(pl.col("id") + 1, pl.col("child") + 1),
                other.children.with_columns(
                    pl.col("id") + n + 1, pl.col("child") + n + 1
                ),
            ],
            how="vertical_relaxed",
        )
        return Term(nodes, children)

    def with_depth(self):
        d = {0: 0}
        for row in self.children.sort("id").iter_rows(named=True):
            parent = row["id"]
            child = row["child"]
            d[child] = d[parent] + 1
        return self.nodes.with_columns(depth=pl.col("id").replace_strict(d))

    def find_redex(self):
        pivoted = self.children.pivot("type", index="id")
        applications = pivoted.filter(pl.col("left").is_not_null())
        lambdas = pivoted.filter(pl.col("down").is_not_null())

        candidates = (
            self.nodes.join(applications, on="id")
            .join(lambdas, left_on="left", right_on="id", suffix="_left")
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
            .select(pl.exclude("src_id"))
        )

        b_subtree_duplicated = (
            b_subtree.with_columns(
                special=pl.col("id").eq(b).or_(None),
            )
            .join(
                nodes.filter(pl.col("ref") == lamb).select(
                    subst_id="id", var_depth="depth"
                ),
                how="cross",
            )
            .with_columns(depth=pl.col("var_depth") + (pl.col("depth") - b_depth) - 2)
        )

        df2 = (
            pl.concat(
                [b_subtree_duplicated, nodes.join(b_subtree, on="id", how="anti")],
                how="diagonal_relaxed",
            )
            .with_columns(
                deleted=(
                    pl.col("id").eq(redex)
                    | pl.col("id").eq(lamb)
                    | pl.col("ref").eq_missing(lamb)
                ),
            )
            .select(
                subst_id="subst_id",
                src_id="id",
                src_ref="ref",
                special="special",
                deleted="deleted",
                depth=pl.when(pl.col("deleted")).then(None).otherwise(pl.col("depth")),
            )
        )
        new_ids = (
            df2.filter(~pl.col("deleted"))
            .select(
                "subst_id",
                "src_id",
                src_child=pl.when(pl.col("special"))
                .then(pl.col("subst_id"))
                .otherwise(pl.col("src_id")),
                subst_child=pl.when(pl.col("special").is_null()).then(
                    pl.col("subst_id")
                ),
            )
            # FIXME: join with subst_id.fill_null(src_id) to avoid the null_equals=True
            .sort([pl.col("subst_id").fill_null(pl.col("src_id")), "src_id"])
            .with_row_index("id")
        )
        df_out = (
            df2.join(new_ids, on=["subst_id", "src_id"], how="left", nulls_equal=True)
            .join(
                new_ids.rename({"id": "ref"}),
                left_on=["subst_id", "src_ref"],
                right_on=["subst_id", "src_id"],
                how="left",
                nulls_equal=True,
            )
            .select("id", "ref", "depth", "src_id")
        ).sort("id")
        children_out = (
            self.children.join(
                b_subtree_duplicated, left_on="child", right_on="id", how="left"
            )
            .with_columns(pl.col("child").replace(redex, a))
            .rename({"id": "src_id", "child": "src_child"})
            .join(new_ids, on=["src_id", "subst_id"], nulls_equal=True)
            .join(
                new_ids.rename({"id": "child"}),
                left_on=["src_child", "subst_id"],
                right_on=["src_child", "subst_child"],
                nulls_equal=True,
            )
            .select("id", "child", "type")
        ).sort(["id", "child"])
        return Term(df_out, children_out), True

    def compute_arity(self):
        return dict(self.children.group_by("id").len().iter_rows())

    def summary(self):
        try:
            bboxes = self.compute_bboxes()
            x = {i: b.x for (i, b) in bboxes.items()}
            y = {i: b.y for (i, b) in bboxes.items()}
            x_expr = pl.col("id").replace_strict(x)
            y_expr = pl.col("id").replace_strict(y)
        except (KeyError, AssertionError):
            x_expr = (None,)
            y_expr = None
        arity = self.compute_arity()
        return self.nodes.join(
            self.children, left_on="id", right_on="child", suffix="_parent", how="left"
        ).with_columns(
            x=x_expr,
            y=y_expr,
            arity=pl.col("id").replace_strict(arity, default=0),
        )

    def compute_bboxes(self) -> dict[int, BBox]:
        result = {0: BBox(None, (0, 0))}
        arities = self.compute_arity()
        for row in self.children.sort("id").iter_rows(named=True):
            parent = row["id"]
            child = row["child"]
            connection = row["type"]

            if connection == "right":
                result[child] = result[parent].copy()

            elif connection == "left" and arities.get(child, 0) != 2:
                result[child] = result[parent].copy()

            else:
                result[child] = result[parent].shift_y(1)

        next_var = 0

        for row in (
            self.children.join(self.nodes, left_on="child", right_on="id")
            .sort(["child", "id"], descending=True)
            .iter_rows(named=True)
        ):
            parent = row["id"]
            child = row["child"]
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

        for row in self.nodes.join(
            self.children, left_on="id", right_on="child", suffix="_parent", how="left"
        ).iter_rows(named=True):
            id = row["id"]
            src_id = row["src_id"]
            if id is None:
                trajectories[id] = [bboxes[src_id], bboxes[src_id]]

            elif last_bboxes is None:
                trajectories[id] = [bboxes[id], bboxes[id]]
            else:
                trajectories[id] = [last_bboxes[src_id], bboxes[id]]

        for row in self.nodes.join(
            self.children, left_on="id", right_on="child", suffix="_parent", how="left"
        ).iter_rows(named=True):
            target_id = row["id"]
            connection = row["type"]
            parent = row["id_parent"]
            ref = row["ref"]
            src_id = row["src_id"]

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
