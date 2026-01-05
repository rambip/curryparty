import polars as pl
from polars import Schema, UInt32
from polars.functions.lazy import coalesce

SCHEMA = Schema(
    {
        "id": UInt32,
        "ref": UInt32,
        "arg": UInt32,
        "bid": pl.Struct({"major": UInt32, "minor": UInt32}),
    },
)


def _shift(nodes: pl.DataFrame, offset: int):
    return nodes.with_columns(
        pl.col("id") + offset,
        pl.col("ref") + offset,
        pl.col("arg") + offset,
        bid=None,
    )


def compose(f: pl.DataFrame, x: pl.DataFrame):
    n = len(f)
    return pl.concat(
        [
            pl.DataFrame(
                [{"id": 0, "arg": n + 1}],
                schema=SCHEMA,
            ),
            _shift(f, 1),
            _shift(x, n + 1),
        ],
    )


def find_redexes(nodes: pl.DataFrame):
    parents = nodes.filter(pl.col("ref").is_null())
    return (
        parents.join(nodes, left_on="id", right_on=pl.col("id") - 1, suffix="_child")
        .filter(
            pl.col("arg").is_not_null(),
            pl.col("ref_child").is_null(),
            pl.col("arg_child").is_null(),
        )
        .select(redex="id", lamb="id_child", b="arg")
    )


def find_variables(nodes: pl.DataFrame, lamb: int):
    return nodes.filter(pl.col("ref") == lamb).select("id", replaced=True)


def subtree(nodes: pl.DataFrame, root: int) -> pl.DataFrame:
    rightmost = root
    while True:
        ref = nodes["ref"][rightmost]
        arg = nodes["arg"][rightmost]
        if ref is not None:
            return nodes.filter(pl.col("id").is_between(root, rightmost))

        rightmost = arg if arg is not None else rightmost + 1


def _generate_bi_identifier(
    major_name: str, minor_name: str, minor_replacement=pl.lit(None)
):
    return pl.struct(
        major=pl.col(major_name).fill_null(pl.col(minor_name)),
        minor=minor_replacement.fill_null(pl.col(minor_name)),
    )


def beta_reduce(nodes: pl.DataFrame, lamb: int, b: int) -> pl.DataFrame:
    redex = lamb - 1
    a = lamb + 1

    vars = find_variables(nodes, lamb)
    b_subtree = subtree(nodes, b)

    b_subtree_duplicated = b_subtree.join(vars, how="cross", suffix="_major")
    rest_of_nodes = nodes.join(b_subtree, on="id", how="anti").with_columns(
        arg=pl.col("arg").replace(redex, a)
    )

    new_nodes = (
        pl.concat(
            [b_subtree_duplicated, rest_of_nodes],
            how="diagonal_relaxed",
        )
        .join(vars, left_on="id", right_on="id", how="anti")
        .join(vars, left_on="arg", right_on="id", how="left", suffix="_arg")
        .join(vars, left_on="ref", right_on="id", how="left", suffix="_ref")
        .filter(
            ~pl.col("id").is_between(redex, lamb),
        )
        .select(
            bid=_generate_bi_identifier("id_major", "id"),
            bid_ref=_generate_bi_identifier("id_major", "ref"),
            bid_ref_fallback=pl.struct(major="ref", minor="ref"),
            bid_arg=_generate_bi_identifier(
                "id_major", "arg", minor_replacement=pl.when("replaced_arg").then(b)
            ),
        )
        .sort("bid")
        .with_row_index("id")
    )

    return (
        new_nodes.join(
            new_nodes.select(bid_ref="bid", ref="id"),
            on="bid_ref",
            how="left",
        )
        .join(
            new_nodes.select(bid_ref_fallback="bid", ref_fallback="id"),
            on="bid_ref_fallback",
            how="left",
        )
        .join(
            new_nodes.select(bid_arg="bid", arg="id"),
            on="bid_arg",
            how="left",
        )
        .select("id", ref=pl.coalesce("ref", "ref_fallback"), arg="arg", bid="bid")
        .sort("id")
    )
