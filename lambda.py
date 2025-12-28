import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo

    from lambda_calc_py import Lambda, Variable, display


@app.cell
def _():
    w = Lambda(Variable(0)(Variable(0)))
    return (w,)


@app.cell
def _(w):
    display(w(w))
    return


@app.cell
def _():
    s = Lambda(Lambda(Lambda(Variable(1)(Variable(2)(Variable(1))(Variable(0))))))
    display(s)
    return (s,)


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
def _(s, two):
    term = s(two)
    mo.output.append(display(term))
    for _ in range(10):
        term, changed = term._beta()
        if not changed:
            break
        mo.output.append(display(term))

    return (term,)


@app.cell
def _(term):
    term
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
