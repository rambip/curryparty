import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import polars as pl

    from curryparty import L, V


@app.function
def carousel(elements):
    return  mo.Html(
    f"""
    <script>

    .mo-slide-content {{
        width: 100%
    }}
    </script>
    {mo.carousel(elements)}
    """
)


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
    omega(omega).show_beta()
    return


@app.cell
def _(omega):
    omega(omega).show_beta().content
    return


@app.cell
def _(s, zero):
    s(s(s(zero))).show_beta()
    return


@app.cell
def _(s, zero):
    s(zero).reduce()
    return


@app.cell
def _(omega):
    omega(omega).show_beta(10)
    return


@app.cell
def _(omega, zero):
    x0 = omega(zero)
    x0.beta()
    return


@app.cell
def _(omega, zero):
    omega(zero).show_beta()
    return


@app.cell
def _(omega, zero):
    x = omega(zero)
    out = x.beta()
    out
    return


@app.cell
def _():
    y = (
        L("f")
        ._(L("g")._("f").call(V("g").call("g")))
        .call(L("g")._("f").call(V("g").call("g")))
    ).build()
    y
    return (y,)


@app.cell
def _(y):
    succ = L("n", "f", "x")._("f").call(V("n").call("f").call("x")).build()
    y(succ).beta().beta().beta().beta().beta()
    return (succ,)


@app.cell
def _():
    fact = (
        L("n", "f")
        ._("n")
        .call(
            L("f", "n")
            ._("n")
            .call(V("f").call(L("f", "x")._("n").call("f").call(V("f").call("x"))))
        )
        .call(L("x")._("f"))
        .call(L("x")._("x").build())
    ).build()
    fact
    return (fact,)


@app.cell
def _(fact, three):
    carousel(fact(three).reduction_chain())
    return


@app.cell
def _(succ, zero):
    succ(zero).beta().beta()
    return


@app.cell
def _(succ, zero):
    foo = succ(zero).beta().beta()
    mo.hstack([foo, foo.data.nodes])
    return (foo,)


@app.cell
def _(foo):
    mo.hstack([foo.beta(), foo.beta().data.nodes])
    return


@app.cell
def _(succ, zero):
    numbers = []
    term = zero
    for i in range(10):
        term = succ(term).reduce()
        numbers.append(term.reduce())
    bar = mo.vstack(numbers)
    return bar, term


@app.cell
def _(succ, term):
    term0 = term
    n0 = []
    for _ in range(30):
        term0 = succ(term0)
        n0.append(term0.reduce())
    baz = mo.vstack(n0)
    return (baz,)


@app.cell
def _(bar, baz):
    mo.hstack([bar, baz])
    return


@app.cell
def _(fact, succ, zero):
    fact(succ(succ(succ(succ(succ(zero)))))).reduce()
    return


@app.cell
def _():
    pred = L("n", "f", "x")._(
        V("n")
        .call(L("g", "h")._("h").call(V("g").call("f")))
        .call(L("u")._("x"))
        .call(L("u")._("u"))
    ).build()
    return (pred,)


@app.cell
def _(pred, zero):
    list(pred(zero).reduction_chain())
    return


@app.cell
def _():
    l_true = L("a", "b")._("a").build()
    l_false = L("a", "b")._("b").build()
    l_not = L("b")._("b").call(l_false).call(l_true).build()
    return (l_not,)


@app.cell
def _(l_not):
    L("x")._(l_not).call(V(l_not).call("x")).build().reduce()
    return


@app.cell
def _(succ):
    succ.nodes.filter(
        pl.col("arg").is_not_null(),
        pl.col("arg").shift().is_null(),
        pl.col("ref").shift().is_null()
    ).select("id", "arg")
    return


@app.cell
def _(succ):
    succ.nodes.shift(-1)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
