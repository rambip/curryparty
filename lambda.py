# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "curryparty==0.4.3",
#     "marimo",
#     "polars==1.38.0",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    from curryparty import L, o


@app.cell(hide_code=True)
def _():
    with open("logo.svg") as file:
        logo = mo.Html(file.read())
    logo.center()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Lambda-calculus demo

    This notebook showcases the `curryparty` library, a library created to illustrate lambda-calculus in a playful way.

    If you like it, go give a star on github :star:

    https://github.com/rambip/curryparty

    If you want a more thourough introduction, you might like this page:

    https://rambip.github.io/lambda_calc.py.html
    """)
    return


@app.cell
def _():
    # Basic use
    # first, create an expression
    identity = L("x").o("x")

    # you can print the classical representation:
    print(identity)

    # or you can display it as a svg
    identity
    return


@app.cell
def _():
    # if there are free variables, you can't do much with it:
    L("x").o("y")
    return


@app.cell
def _():
    # you can check if a term has free variables:
    L("x").o("y").check()
    return


@app.cell
def _():
    # church booleans
    false = L("a", "b").o("a").check()
    true = L("a", "b").o("b").check()
    return false, true


@app.cell
def _(false):
    print(false)
    false
    return


@app.cell
def _(true):
    print(true)
    true
    return


@app.cell
def _():
    # church numerals
    # a single argument in `o` is a value
    zero = L("f", "x").o("x")
    # two arguments is a function call
    one = L("f", "x").o("f", "x")
    # you can use `o` as a function:
    two = L("f", "x").o("f", o("f", "x"))
    # and so on
    three = L("f", "x").o("f", o("f", o("f", "x")))
    return one, three, two, zero


@app.cell
def _(one, three, two, zero):
    print(zero, one, two, three, sep="  |  ")
    mo.hstack([zero, one, two, three])
    return


@app.cell
def _():
    # successor function
    succ = L("n", "f", "x").o("f", o("n", "f", "x")).check()
    succ
    return (succ,)


@app.cell
def _(false, true):
    not_ = L("b").o("b", true, false).check()
    return (not_,)


@app.cell
def _(false, not_):
    # compose terms
    print(not_(false))
    not_(false)
    return


@app.cell
def _(false, not_):
    # simplify expressions
    not_(false).beta()
    return


@app.cell
def _(false, not_):
    not_(false).beta().beta()
    return


@app.cell
def _(false, not_):
    not_(false).beta().beta().beta()
    return


@app.cell
def _(false, not_):
    # the term cannot be reduced anymore
    print(not_(false).beta().beta().beta().beta())
    return


@app.cell
def _(false, not_):
    # show all steps
    steps = list(not_(false).reduction_chain())
    mo.vstack(steps)
    return


@app.cell
def _(false, not_):
    # jump to the result
    not_(false).reduce()
    return


@app.cell
def _(succ, zero):
    # another example
    succ(zero)
    return


@app.cell
def _(one, succ, zero):
    should_be_one = succ(zero).reduce()
    assert should_be_one == one
    should_be_one
    return


@app.cell
def _(succ, zero):
    mo.carousel(succ(succ(zero)).reduction_chain())
    return


@app.cell
def _(succ, zero):
    # show animations
    succ(zero).show_beta()
    return


@app.cell
def _(succ, zero):
    # show all reductions
    mo.carousel(x.show_beta() or x for x in succ(zero).reduction_chain())
    return


@app.cell
def _():
    omega = L("x").o("x", "x").check()
    print(omega)
    omega
    return (omega,)


@app.cell
def _(omega):
    omega(omega).show_beta()
    return


@app.cell
def _(omega):
    # faster
    omega(omega).show_beta(duration=2)
    return


@app.cell
def _():
    # favorite combinator
    y = L("f").o(
        L("g").o("f", o("g", "g")),
        L("g").o("f", o("g", "g")),
    )  # .check()
    y
    return


@app.cell
def _():
    # factorial function without recursion
    fact = (
        L("n", "f").o(
            "n",
            L("f", "n").o("n", o("f", L("f", "x").o("n", "f", o("f", "x")))),
            L("x").o("f"),
            L("x").o("x"),
        )
    ).check()
    fact
    return (fact,)


@app.cell
def _(fact, succ, three):
    five = succ(succ(three)).check().reduce()
    fact(five).reduce()
    return


@app.cell
def _(succ, zero):
    # illustration of the smart layout logic:
    numbers = []
    term = zero
    for i in range(30):
        term = succ(term).reduce()
        numbers.append(term.reduce())
    mo.hstack(
        [
            mo.vstack(numbers[:10]),
            mo.vstack(numbers[10:]),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
