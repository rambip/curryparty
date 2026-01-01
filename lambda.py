import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import polars as pl

    from lambda_calc_py.dataframe import Term, L, V


@app.cell
def _():
    zero = L("f", "x")._("x").build()
    omega = L("x")._("x").call("x").build()
    return omega, zero


@app.cell
def _():
    s = (
        L("n", "f", "x")
        ._("f")
        .call(
            V("n").call("f").call("x")
        )
    ).build()
    return (s,)


@app.cell
def _(zero):
    mo.Html(zero.display().as_str())
    return


@app.cell
def _(omega):
    mo.Html(omega(omega).display().as_str())
    return


@app.cell
def _(s, zero):
    t = s(zero)
    mo.Html(t.display().as_str())
    return (t,)


@app.cell
def _(t):
    step1, _ = t._beta()
    return (step1,)


@app.cell
def _(step1):
    mo.Html(step1.display().as_str())
    return


@app.cell
def _(step1):
    step2, _ = step1._beta()
    mo.Html(step2.display(step1).as_str())
    return (step2,)


@app.cell
def _(step2):
    mo.Html(step2.display().as_str())
    return


if __name__ == "__main__":
    app.run()
