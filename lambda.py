import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import polars as pl

    from curryparty import L, V


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
def _(omega):
    omega
    return


@app.cell
def _():
    s = (L("n", "f", "x")._("f").call(V("n").call("f").call("x"))).build()
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
def _(omega):
    omega(omega).show_reduction().content
    return


@app.cell
def _(s, zero):
    mo.vstack([x.show_reduction() for x in s(s(s(zero))).reduction_chain()])
    return


@app.cell
def _(s, zero):
    s(zero).beta().beta()
    return


@app.cell
def _(s, zero):
    s(zero).beta().beta().beta()
    return


@app.cell
def _(omega):
    omega(omega).show_reduction()
    return


@app.cell
def _(omega, zero):
    x0 = omega(zero)
    x0.beta()
    return (x0,)


@app.cell
def _(omega, zero):
    omega(zero).show_reduction()
    return


@app.cell
def _(omega, zero):
    x = omega(zero)
    out = x.beta()
    out
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
