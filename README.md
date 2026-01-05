# Curry party

`curryparty` is a library created to explore, visualize and teach lambda-calculus concepts.

# Install

Run `pip install curryparty` or `uv add curryparty` depending on your package manager.

# How to use

```python
from curryparty import L, V

# build expressions idiomatically
identity = L("x")._("x").build()

zero = L("f", "x")._("x").build()

omega = L("x")._("x").call("x").build()

# you can create more complex terms:

succ = L("n", "f", "x")._("f").call(
    V("n").call("f").call("x")
)

# If you try to combine them, nothing will happen:
bomb = omega(omega)
one = s(zero)

# You need to beta-reduce them:
bomb.beta()
one_with_first_reduction = one.beta()

# if you want the final form:
term = s(zero)
while term is not None:
    term = term.beta()
    print(term)

# this is equivalent to:
for x in term.reduction_chain():
    print(x)
```

But the main point of this library is the svg-based display system.
If you use a notebook such as jupyternotebook or marimo, you will see something like this:

<img width="1140" height="440" alt="notebook_display" src="https://github.com/user-attachments/assets/6ac48738-eb58-4f7c-908f-567a855f37f5" />



You can also use `term.show_reduction` to get an animated version.

# Tutorial

`lambda.py` shows a tutorial in marimo format. Click on the button to try it out:

[![Open with marimo](https://marimo.io/shield.svg)](https://marimo.app/github.com/rambip/curryparty/blob/main/lambda.py)

# How it works

Under the wood, all the terms are converted into a list of nodes, that can either be a `lambda`, an `application` (with 2 arguments) or a `variable` (with 0 arguments).

For efficiency, they are converted to a [polars](https://github.com/pola-rs/polars) dataframe in prefix traversal order.
If you want to understand how it works, start by looking at the `.nodes` attribute of any term.
