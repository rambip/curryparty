import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import polars as pl

    from lambda_calc_py import Lambda, Variable, display
    from lambda_calc_py.dataframe import Term


@app.function
def var(i: int):
    return pl.from_dicts([
        {"type": "variable", }
    ])


@app.cell
def _():
    w = Lambda(Variable(0)(Variable(0)))
    return (w,)


@app.cell
def _(s):
    s.compute_y_position()
    return


@app.cell
def _():
    zero = Term(
        [None, None, {"ref": 1}], 
        [
            {"id": 0, "child": 1, "type": "down"},
            {"id": 1, "child": 2, "type": "down"},
        ]
    )
    return (zero,)


@app.cell
def _():
    s = Term(
        [
            None, None, None, None, {"ref": 1}, None, None, {"ref": 0}, {"ref": 1}, {"ref": 2}
        ], 
        [
            {"id": 0, "child": 1, "type": "down"},
            {"id": 1, "child": 2, "type": "down"},
            {"id": 2, "child": 3, "type": "down"},
            {"id": 3, "child": 4, "type": "left"},
            {"id": 3, "child": 5, "type": "right"},
            {"id": 5, "child": 6, "type": "left"},
            {"id": 5, "child": 9, "type": "right"},
            {"id": 6, "child": 7, "type": "left"},
            {"id": 6, "child": 8, "type": "right"},
        ]
    )
    mo.Html(s.display().as_str())
    return (s,)


@app.cell
def _():
    omega = Term(
        [
            None,
            None,
            {
                "ref": 0,
            },
            {
                "ref": 0,
            },
        ],
        [
            {"id": 0, "child": 1, "type": "down"},
            {"id": 1, "child": 2, "type": "left"},
            {"id": 1, "child": 3, "type": "right"},
        ],
    )
    return


@app.cell
def _(s, zero):
    t = s(zero)
    t.nodes
    return (t,)


@app.cell
def _(t):
    step, _ = t._beta()
    step.nodes, step.children
    return (step,)


@app.cell
def _(step):
    step.compute_x_bounds()
    return


@app.cell
def _(t):
    _b = t._beta()[0]
    _b.nodes.drop_nulls("id"), _b.children
    return


@app.cell
def _():
    import svg

    svg.Rect(elements=[svg.Animate(attributeName="x")]).as_str()
    return


@app.cell
def _(t):
    mo.Html(t._beta()[0].display().as_str())
    return


@app.cell
def _(w):
    display(w(w))
    return


@app.cell
def _():
    _s = Lambda(Lambda(Lambda(Variable(1)(Variable(2)(Variable(1))(Variable(0))))))
    display(_s)
    return


@app.cell
def _(VVariable):
    y = Lambda(Lambda(VVariable(0)(Variable(0))()))
    return (y,)


@app.cell
def _():
    two = Lambda(Lambda(Variable(1)(Variable(1)(Variable(0)))))
    display(two)
    return (two,)


@app.cell
def _(s, two):
    s(two)
    return


@app.cell
def _(s, two):
    s(two)._beta()
    return


@app.cell
def _():
    _zero = Lambda(Lambda(Variable(0)))
    display(_zero)
    return


@app.cell
def _(s, y, zero):
    y(s)(zero)
    return


@app.cell
def _(s, y, zero):
    term = y(s)(zero)
    result = [display(term)]
    for _ in range(20):
        term, changed = term._beta()
        if not changed:
            break
        result.append(display(term))
    mo.vstack(result)
    return (term,)


@app.cell
def _(s, two):
    s(two)._beta()[0].body.body#._beta()[0]
    return


@app.cell
def _(s, two):
    s(two)._beta()[0]._beta()[0].body.body
    return


@app.cell
def _(term):
    term
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
