from datetime import timedelta
from typing import Callable, Iterable, Optional

import polars as pl
import svg
from polars import Schema, String, UInt32

SCHEMA_NODES = Schema(
    {"id": UInt32, "ref": UInt32, "depth": UInt32, "src_id": UInt32},
)
SCHEMA_CHILDREN = Schema({"id": UInt32, "child": UInt32, "type": String})


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

    def clear(self):
        self.nodes = self.nodes.drop_nulls("id")

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
        self.clear()
        candidate = self.find_redex()
        if candidate is None:
            return self, False
        redex, lamb, b, a, redex_depth = candidate
        b_depth = redex_depth + 1

        b_subtree = (
            self.nodes.filter(pl.col("id") >= b)
            .filter(pl.col("depth").lt(b_depth).cum_sum().eq(0))
            .select(pl.exclude("src_id"))
        )

        b_subtree_duplicated = (
            b_subtree.with_columns(
                special=pl.col("id").eq(b).or_(None),
            )
            .join(
                self.nodes.filter(pl.col("ref") == lamb).select(
                    subst_id="id", var_depth="depth"
                ),
                how="cross",
            )
            .with_columns(depth=pl.col("var_depth") + (pl.col("depth") - b_depth) - 2)
        )

        df2 = (
            pl.concat(
                [b_subtree_duplicated, self.nodes.join(b_subtree, on="id", how="anti")],
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
            x_min, x_max = self.compute_x_bounds()
            x_min_expr = pl.col("id").replace_strict(x_min, default=None)
            x_max_expr = pl.col("id").replace_strict(x_max, default=None)
        except (KeyError, AssertionError):
            x_min_expr = None
            x_max_expr = None
        try:
            y = self.compute_y_position()
            y_expr = pl.col("id").replace_strict(y, default=None)
        except KeyError:
            y_expr = None
        arity = self.compute_arity()
        return self.nodes.join(
            self.children, left_on="id", right_on="child", suffix="_parent", how="left"
        ).with_columns(
            y=y_expr,
            x_min=x_min_expr,
            x_max=x_max_expr,
            arity=pl.col("id").replace_strict(arity, default=0),
        )

    def compute_y_position(self):
        """
        TODO: bounds of children
        """
        y = {0: 0}
        arities = self.compute_arity()

        for row in self.children.sort("id").iter_rows(named=True):
            parent = row["id"]
            child = row["child"]
            connection = row["type"]

            if connection == "right":
                y[child] = y[parent]

            elif connection == "left" and arities.get(child, 0) != 2:
                y[child] = y[parent]

            else:
                y[child] = y[parent] + 1

        return y

    def compute_x_bounds(self):
        x_min = {}
        x_max = {}
        next_var = 0
        arities = self.compute_arity()
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
                x_min[child] = next_var
                x_max[child] = next_var
                next_var -= 1

                assert ref is not None
                if ref not in x_min:
                    x_min[ref] = x_min[child]
                    x_max[ref] = x_max[child]
                else:
                    x_min[ref] = min(x_min[ref], x_min[child])
                    x_max[ref] = max(x_max[ref], x_max[child])

            # Update parent
            if connection == "left":
                if parent not in x_min:
                    x_min[parent] = x_min[child]
                    x_max[parent] = x_max[child]

            elif connection == "right":
                if parent in x_min:
                    x_min[parent] = min(x_min[parent], x_min[child])
                    x_max[parent] = max(x_max[parent], x_max[child])

            else:
                if parent in x_min:
                    x_min[parent] = min(x_min[parent], x_min[child])
                    x_max[parent] = max(x_max[parent], x_max[child])
                else:
                    x_min[parent] = x_min[child]
                    x_max[parent] = x_max[child]
        return x_min, x_max

    def position(
        self,
        node_id,
        arity,
        time,
        y_last,
        x_min_last,
        x_max_last,
        y_now,
        x_min_now,
        x_max_now,
        src_id,
    ):
        """Returns bounding box [[x_min, x_max], [y_min, y_max]] for a node at given time (0 or 1)"""
        if src_id is None and node_id is None:
            raise ValueError("unkown node")

        ENABLED = False
        try:
            if ENABLED and time == 0 and src_id is not None:
                y = y_last[src_id]
                x1 = x_min_last[src_id]
                x2 = x_max_last[src_id]
            else:
                y = y_now[node_id]
                x1 = x_min_now[node_id]
                x2 = x_max_now[node_id]
        except KeyError:
            y = 0
            x1 = 0
            x2 = 0

        if arity == 1:  # Lambda
            return [
                [x1, x2],
                [y, y + 1],
            ]
        elif arity == 0:  # Variable
            return [
                [x1, x1 + 1],
                [y, y + 1],
            ]
        else:  # Application (arity == 2)
            return [
                [x1, x2],
                [y, y + 1],
            ]

    def _svg_blocks(self, last: Optional["Term"]) -> Iterable[svg.Element]:
        y_now = self.compute_y_position()
        x_min_now, x_max_now = self.compute_x_bounds()

        if last is None:
            y_last = y_now
            x_min_last, x_max_last = x_min_now, x_max_now
        else:
            y_last = last.compute_y_position()
            x_min_last, x_max_last = last.compute_x_bounds()

        arities = self.compute_arity()

        for row in self.nodes.join(
            self.children, left_on="id", right_on="child", suffix="_parent", how="left"
        ).iter_rows(named=True):
            target_id = row["id"]
            connection = row["type"]
            parent = row["id_parent"]
            ref = row["ref"]
            src_id = row["src_id"]

            if target_id is None:
                continue

            arity = arities.get(target_id, 0)

            if connection == "right":
                assert parent is not None
                yield svg_left_arrow(
                    x_min_now[target_id],
                    x_max_now[parent],
                    y_now[target_id],
                    y_now[parent],
                )

            bboxes = [
                self.position(
                    target_id,
                    arity,
                    0,
                    y_last,
                    x_min_last,
                    x_max_last,
                    y_now,
                    x_min_now,
                    x_max_now,
                    src_id,
                ),
                self.position(
                    target_id,
                    arity,
                    1,
                    y_last,
                    x_min_last,
                    x_max_last,
                    y_now,
                    x_min_now,
                    x_max_now,
                    src_id,
                ),
            ]

            if arity == 1:  # Lambda
                yield svg.Rect(
                    height=0.8,
                    fill="blue",
                    elements=[
                        animate_bbox("x", bboxes),
                        animate_bbox("y", bboxes),
                        animate_bbox("width", bboxes),
                    ],
                )
            elif arity == 0:  # Variable
                yield svg.Rect(
                    width=0.8,
                    height=0.8,
                    fill="red",
                    elements=[
                        animate_bbox("x", bboxes),
                        animate_bbox("y", bboxes),
                    ],
                )
                yield svg.Line(
                    x1=x_min_now[target_id] + 0.5,
                    y1=y_now[target_id] + 0.1,
                    x2=x_min_now[target_id] + 0.5,
                    y2=y_now[ref] + 0.9,
                    stroke_width=0.05,
                    stroke="gray",
                )
            elif arity == 2:  # Application
                yield svg.Rect(
                    height=0.8,
                    fill_opacity=0.5,
                    stroke="orange",
                    stroke_width=0.1,
                    fill="none",
                    elements=[
                        animate_bbox("x", bboxes),
                        animate_bbox("y", bboxes),
                        animate_bbox("width", bboxes),
                    ],
                )

    def display(self, last: Optional["Term"] = None):
        return svg.SVG(
            xmlns="http://www.w3.org/2000/svg",
            viewBox="-10 0 20 10",  # type: ignore
            height="400px",  # type: ignore
            elements=list(self._svg_blocks(last)),
        )


def animate_bbox(name: str, bboxes: list, duration: int = 2) -> svg.Element:
    """Create animation for a bounding box attribute (x, y, width, or height)"""
    bbox0, bbox1 = bboxes

    if name == "x":
        values = f"{0.1 + bbox0[0][0]};{0.1 + bbox1[0][0]}"
    elif name == "y":
        values = f"{0.1 + bbox0[1][0]};{0.1 + bbox1[1][0]}"
    elif name == "width":
        values = f"{0.8 + bbox0[0][1] - bbox0[0][0]};{0.8 + bbox1[0][1] - bbox1[0][0]}"
    elif name == "height":
        values = f"{0.8 + bbox0[1][1] - bbox0[1][0]};{0.8 + bbox1[1][1] - bbox1[1][0]}"
    else:
        raise ValueError(f"Unknown attribute: {name}")

    return svg.Animate(
        attributeName=name,
        values=values,
        dur=timedelta(seconds=duration),
        repeatCount="indefinite",
    )


def animate(name: str, f: Callable[[int], float], duration: int = 2) -> svg.Element:
    return svg.Animate(
        attributeName=name,
        values=f"{f(0)};{f(1)}",
        dur=timedelta(seconds=duration),
        repeatCount="indefinite",
    )


def svg_left_arrow(x0, x1, y0, y1, s=0.1):
    line = svg.Line(
        x1=0.1 + x0,
        y1=0.5 + y0,
        x2=0.5 + x1,
        y2=0.5 + y1,
        stroke="black",
        stroke_width=0.05,
    )
    triangle = svg.Polygon(
        points=[
            0.5 + x1 + s,
            0.5 + y1 - s,
            0.5 + x1 - s,
            0.5 + y1,
            0.5 + x1 + s,
            0.5 + y1 + s,
        ],
        stroke_width=0,
        fill="black",
        # fill="black",
    )
    return svg.G(elements=[line, triangle])
