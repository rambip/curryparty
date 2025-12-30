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
            .join(lambdas, left_on="left", right_on="id")
            .select("id", "left", "right", "depth")
        )

        return candidates.row(0) if len(candidates) > 0 else None

    def _beta(self) -> tuple["Term", bool]:
        self.clear()
        candidate = self.find_redex()
        if candidate is None:
            return self, False
        redex, lamb, b, redex_depth = candidate
        b_depth = redex_depth + 1

        b_subtree = self.nodes.filter(pl.col("id") >= b).filter(
            pl.col("depth").lt(b_depth).cum_sum().eq(0)
        )

        b_subtree_duplicated = (
            b_subtree.select(
                pl.col("id"),
                special=pl.col("id").eq(b).or_(None),
                src_depth=pl.col("depth"),
            )
            .join(
                self.nodes.filter(pl.col("ref") == lamb).select(
                    subst_id="id", var_depth="depth"
                ),
                how="cross",
            )
            .with_columns(
                new_depth=pl.col("var_depth") + (pl.col("src_depth") - b_depth)
            )
        )

        df2 = (
            self.nodes.join(
                b_subtree_duplicated, left_on="id", right_on="id", how="left"
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
                depth=pl.when(pl.col("deleted"))
                .then(None)
                .otherwise(
                    pl.coalesce(
                        pl.col("new_depth"),  # Use new depth for duplicated nodes
                        pl.col("depth"),  # Keep original depth for all other nodes
                    )
                    - 2
                ),
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
            .sort(["subst_id", "src_id"])
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

    def arities(self):
        return dict(self.children.group_by("id").len().iter_rows())

    def compute_y_position(self):
        """
        TODO: bounds of children
        """
        y = {0: 0}
        arities = self.arities()

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
        arities = self.arities()
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

    def _svg_blocks(self, last: Optional["Term"]) -> Iterable[svg.Element]:
        y_now = self.compute_y_position()
        x_min_now, x_max_now = self.compute_x_bounds()
        if last is None:
            y_last = y_now
            x_min_last, x_max_last = x_min_now, x_max_now
        else:
            y_last = last.compute_y_position()
            x_min_last, x_max_last = last.compute_x_bounds()
        y = [y_last, y_now]
        x_min = [x_min_last, x_min_now]
        x_max = [x_max_last, x_max_now]

        arities = self.arities()
        for row in self.nodes.join(
            self.children, left_on="id", right_on="child", suffix="_parent", how="left"
        ).iter_rows(named=True):
            target_id = row["id"]
            connection = row["type"]
            parent = row["id_parent"]
            ref = row["ref"]
            src_id = row["src_id"]
            if target_id is None:
                # freshly deleted node
                continue
            if connection == "right":
                assert parent is not None
                yield svg_left_arrow(
                    x_min[target_id], x_max[parent], y[target_id], y[parent]
                )
            if arities.get(target_id, 0) == 1:
                yield (
                    svg.Rect(
                        height=0.8,
                        fill="blue",
                        elements=[
                            animate(
                                "x", lambda i: 0.1 + x_min[i][[src_id, target_id][i]]
                            ),
                            animate("y", lambda i: 0.1 + y[i][[src_id, target_id][i]]),
                            animate(
                                "width",
                                lambda i: 0.8
                                + x_max[i][[src_id, target_id][i]]
                                - x_min[i][[src_id, target_id][i]],
                            ),
                        ],
                    )
                )
            elif arities.get(target_id, 0) == 0:
                yield svg.Rect(
                    width=0.8,
                    height=0.8,
                    fill="red",
                    elements=[
                        animate("x", lambda i: 0.1 + x_min[i][[src_id, target_id][i]]),
                        animate("y", lambda i: 0.1 + y[i][[src_id, target_id][i]]),
                    ],
                )
                yield (
                    svg.Line(
                        x1=x_min[target_id] + 0.5,
                        y1=y[target_id] + 0.1,
                        x2=x_min[target_id] + 0.5,
                        y2=y[ref] + 0.9,
                        stroke_width=0.05,
                        stroke="gray",
                        elements=[
                            animate(
                                "x1", lambda i: 0.5 + x_min[i][[src_id, target_id][i]]
                            ),
                            animate("y1", lambda i: 0.1 + y[i][[src_id, target_id][i]]),
                            animate(
                                "x2", lambda i: 0.5 + x_min[i][[src_id, target_id][i]]
                            ),
                            # fixme: old and new ref
                            animate("y2", lambda i: 0.9 + y[i][ref]),
                        ],
                    )
                )
            elif arities.get(target_id, 0) == 2:
                yield (
                    svg.Rect(
                        height=0.8,
                        fill_opacity=0.5,
                        stroke="orange",
                        stroke_width=0.1,
                        fill="none",
                        elements=[
                            animate(
                                "x",
                                lambda i: 0.1 + x_min[i][[src_id, target_id][i]],
                            ),
                            animate("y", lambda i: 0.1 + y[i][[src_id, target_id][i]]),
                            animate(
                                "width",
                                lambda i: 0.8
                                + x_max[i][[src_id, target_id][i]]
                                - x_min[i][[src_id, target_id][i]],
                            ),
                        ],
                    )
                )

    def display(self, last: Optional["Term"] = None):
        return svg.SVG(
            xmlns="http://www.w3.org/2000/svg",
            viewBox="-10 0 20 10",  # type: ignore
            height="400px",  # type: ignore
            elements=list(self._svg_blocks(last)),
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
