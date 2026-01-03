import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import polars as pl

    from lambda_calc_py import L, V


@app.cell
def _():
    zero = L("f", "x")._("x").build()
    omega = L("x")._("x").call("x").build()
    return omega, zero


@app.cell
def _(zero):
    zero
    return


@app.cell
def _():
    s = (
        L("n", "f", "x")
        ._("f")
        .call(
            V("n").call("f").call("x")
        )
    ).build()
    s
    return (s,)


@app.cell
def _(omega):
    omega(omega).show_reduction()
    return


@app.cell
def _(x0):
    x0.show_reduction()
    return


@app.cell
def _(x0):
    for ((v,), nodes) in x0.new_nodes.group_by(pl.col("bid").struct.field("major")):
        print(nodes)
    x0.b
    return


@app.cell
def _(omega):
    omega(omega).show_reduction().content
    return


@app.cell
def _(s, zero):
    t = s(zero)
    t
    return (t,)


@app.cell
def _(t):
    step1, _ = t._beta()
    step1.show_reduction()
    return (step1,)


@app.cell
def _(step1):
    step2, _ = step1._beta()
    step2
    return


@app.cell
def _(omega):
    omega(omega).show_reduction()
    return


@app.cell
def _(omega, zero):
    x0 = omega(zero)
    x0._beta()
    x0.new_nodes
    return (x0,)


@app.cell
def _(omega, zero):
    omega(zero).show_reduction()
    return


@app.cell
def _(omega, zero):
    x = omega(zero)
    out = x._beta()[0]
    out
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
