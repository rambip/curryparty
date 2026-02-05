<p align="center">
  <img src="https://github.com/rambip/curryparty/blob/main/logo.svg?raw=true" width="500px"/>
</p>


# Curry party

`curryparty` is a library created to explore, visualize and teach lambda-calculus concepts.

# Install

Run `pip install curryparty` or `uv add curryparty` depending on your package manager.

# How to use

```python
from curryparty import L, o

# Build classic lambda calculus expressions

# Identity: λx. x
identity = L("x").o("x").build()

# Const (K combinator): λx. λy. x
const = L("x", "y").o("x").build()

# Omega: λx. x x
omega = L("x").o("x", "x").build()
print(omega(omega))  # (λ0 x0(x0))((λ1 x1(x1)))

# Church numerals
zero = L("f", "x").o("x").build()
one = L("f", "x").o("f", "x").build()
two = L("f", "x").o("f", o("f", "x")).build()

succ = L("n", "f", "x").o("f", o("n", "f", "x")).build()

# S combinator: λf. λg. λx. f x (g x)
s_comb = L("f", "g", "x").o("f", "x", o("g", "x")).build()

# Beta reduction
# Nothing happens until you reduce:
app = identity(identity)
print(app)  # (λ0 x0)((λ1 x1))

# Perform one reduction step:
reduced = app.beta()
print(reduced)  # λ0 x0

# Multiple reductions:
term = succ(zero)
print(f"Steps: {len(list(term.reduction_chain()))}")  # Steps: 3

# Print each step:
for step in term.reduction_chain():
    print(step)

# Or get the final result directly:
result = succ(zero).reduce()
print(result)  # λ0 λ1 x0(x1) (i.e, one)
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
