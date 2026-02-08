"""The `curryparty` library, a playground to learn lambda-calculus.

This library is intended to be used in an interactive
"""

from typing import Iterable, Optional, Union

try:
    from importlib.util import find_spec

    find_spec("polars")
except ImportError:
    raise ImportError(
        "curryparty needs the `polars` library. \n Please install it, typically with `pip install polars`"
    )
import uuid

from svg import SVG, Length, Rect, ViewBoxSpec

from .core import AbstractTerm
from .display import (
    compute_height,
    compute_svg_frame_final,
    compute_svg_frame_init,
    compute_svg_frame_phase_a,
    compute_svg_frame_phase_b,
    count_variables,
)
from .term import App, Lam, Term, Var
from .utils import ShapeAnim, ShapeAnimFrame

__all__ = ["L", "o"]


def log2(n):
    if n <= 0:
        raise ValueError(f"log2 of negative number {n}")
    elif n == 1:
        return 0
    return 1 + log2(n // 2)


class LambdaTerm:
    def __init__(self, data: Union[Term, AbstractTerm]):
        """
        The main exported class of the library.

        It is not meant to be used directly. To build a term, use the `L` class or the `o` function instead.

        a `LambdaTerm` can be either:
            - complete when there is no free variable left in it
            - uncomplete when there are still free variable in it

        To assert that a lambda term is complete, use `term.check()`
        """
        if isinstance(data, AbstractTerm):
            self.data = data
            return
        if data.is_fully_bound():
            self.data = AbstractTerm.from_term(data)
        else:
            self.data: Union[Term, AbstractTerm] = data

    def as_term(self) -> "Term":
        """
        Converts to the shared term representation.
        """
        if isinstance(self.data, Term):
            return self.data
        return self.data.to_term()

    def build(self) -> "LambdaTerm":
        print(
            "Warning: `build` is deprecated. You don't need to build terms anymore.\n"
            "To check if there is no free variable in your term, use `check` instead"
        )
        assert isinstance(self.data, AbstractTerm), (
            f"Lambda term has free variables: {list(self.data.free_vars())}"
        )
        return self

    def check(self) -> "LambdaTerm":
        """
        Checks that there is no freevar in the expression, and return itself
        """
        assert isinstance(self.data, AbstractTerm), (
            f"Lambda term has free variables: {list(self.data.free_vars())}"
        )
        return self

    def __call__(self, other: "LambdaTerm") -> "LambdaTerm":
        return LambdaTerm((self.as_term()(other.as_term())))

    def beta(self) -> Optional["LambdaTerm"]:
        """
        Operate a lambda-reduction, one single evaluation step.
        If you want to get the final form, use `reduce`.
        If you want to get all intermediate step before the final form, use `reduction_chain`.
        """
        assert isinstance(self.data, AbstractTerm), (
            "Lambda term has free variables, it cannot be reduced"
        )
        candidates = self.data.find_redexes()
        redex = next(candidates, None)
        if redex is None:
            return None
        reduced = self.data.beta_reduce(redex)
        return LambdaTerm(reduced)

    def reduce(self):
        """
        Reduce the term to its final form.
        Note that this function is not guaranteed to terminate, for example:
        ```py
        omega = L("x").o("x", "x")
        omega(omega).reduce()
        ```

        will hang forever.

        If you want to get all the intermediate reduction steps, use `reduction_chain`
        """
        last_non_reduced = self
        for term in self.reduction_chain():
            last_non_reduced = term
        return last_non_reduced

    def reduction_chain(self) -> Iterable["LambdaTerm"]:
        """
        Yield every intermediate step in the reduction of the term to a final form.
        """
        term = self
        while term is not None:
            yield term
            term = term.beta()

    def __eq__(self, other: object):
        """
        Check if 2 lambda-terms are equal, up to renaming the variables
        """
        self.check()
        if not isinstance(other, LambdaTerm):
            return False
        return self.data == other.data

    def __str__(self) -> str:
        """Convert to string representation using lambda notation."""
        lambda_counter = [0]  # Global counter for lambda indices

        def pretty(t, lambda_stack=[], paren=False):
            if isinstance(t, Var):
                # Get the lambda index this variable refers to
                if t.is_bound:
                    if t.index < len(lambda_stack):
                        lam_idx = lambda_stack[-(t.index + 1)]
                        return f"x{lam_idx}"
                    return f"x{t.index}"
                else:
                    # Free variable - use its name
                    return str(t.name)
            elif isinstance(t, Lam):
                idx = lambda_counter[0]
                lambda_counter[0] += 1
                result = f"λ{idx} {pretty(t.body, lambda_stack + [idx])}"
                return f"({result})" if paren else result
            elif isinstance(t, App):
                func = pretty(t.func, lambda_stack, paren=isinstance(t.func, Lam))
                arg = pretty(t.arg, lambda_stack, paren=not isinstance(t.arg, Var))
                return f"{func}({arg})"
            return str(t)

        return pretty(self.as_term())

    def show_beta(self, duration=7):
        """
        Show one reduction step as an SVG animation.
        Returns None if the term can't be reduced.

        Tip: use `term.show_beta() or term` to fallback to the static figure if the term can't be reduced.
        """
        assert isinstance(self.data, AbstractTerm), (
            "Lambda term has free variables, it cannot be reduced"
        )

        candidates = self.data.find_redexes()
        redex = next(candidates, None)
        if redex is None:
            return None

        lamb = self.data.node(redex).get_left()
        assert lamb is not None
        b = self.data.node(redex).get_arg()
        assert b is not None
        new_nodes = self.data.beta_reduce(redex)
        vars = list(self.data.find_variables(lamb))
        b_subtree = list(self.data.get_subtree(b))
        height = min(compute_height(self.data), compute_height(new_nodes)) * 2
        raw_width = max(count_variables(self.data), count_variables(new_nodes))
        width = 1 << (1 + log2(raw_width))
        frame_data: list[ShapeAnimFrame] = []
        N_STEPS = 6

        for t in range(N_STEPS):
            if t == 0:
                items = compute_svg_frame_init(self.data, t)
            elif t == 1 or t == 2:
                items = compute_svg_frame_phase_a(self.data, redex, b_subtree, vars, t)
            elif t == 3 or t == 4:
                items = compute_svg_frame_phase_b(
                    self.data, redex, b_subtree, new_nodes, t
                )
            else:
                items = compute_svg_frame_final(new_nodes, t)
            frame_data.extend(items)

        figure_id = uuid.uuid4()
        box_id = f"lambda_box_{figure_id}".replace("-", "")
        grouped = ShapeAnim.group_by_key(frame_data)
        anims = [ShapeAnim.from_frames(frames, duration) for frames in grouped.values()]
        anims.sort(key=lambda a: a.zindex)
        anim_elements = [
            x.to_element(N_STEPS, begin=f"{box_id}.click", reset=f"{box_id}.mouseover")
            for x in anims
        ]

        anim_elements.append(
            Rect(
                id=box_id,
                x=-width,
                y=0,
                width=Length(100, "%"),
                height=Length(100, "%"),
                fill="transparent",
            )
        )

        # prefered size in pixels
        H = height * 40
        anim_svg = SVG(
            xmlns="http://www.w3.org/2000/svg",
            viewBox=ViewBoxSpec(-width, 0, 2 * width, height),
            style=f"max-height:{H}px",
            elements=anim_elements,
        ).as_str()

        return Html(
            '<div style="width:100%">'
            '<div style="margin-bottom:30px">'
            "click to animate, move away and back to reset"
            "</div>"
            f"{anim_svg}"
            "</div>"
        )

    def _repr_html_(self):
        """
        Display the term as a static svg.
        """
        if not isinstance(self.data, AbstractTerm):
            mention = ""
            free_vars = list(set(self.data.free_vars()))
            if not self.data.is_valid(0):
                mention = " [invalid]"
            elif free_vars:
                mention = f" [free: {', '.join(free_vars)}]"
            return self.data.__repr__() + mention

        frame = sorted(compute_svg_frame_init(self.data), key=lambda x: x.zindex)

        width = (1 << (1 + log2(count_variables(self.data)))) + 4
        height = compute_height(self.data) + 1

        elements = [ShapeAnim.from_single_frame(x) for x in frame]

        # prefered size in pixels
        H = height * 40
        W = width * 40

        return SVG(
            xmlns="http://www.w3.org/2000/svg",
            viewBox=ViewBoxSpec(-1, 0, width, height),
            elements=elements,
            style=f"max-height:{H}px; margin-left: clamp(0px, calc(100% - {W}px), 100px)",
        ).as_str()


# type of the argument used to create lambda-terms
BuilderArg = Union[str, "Term", "LambdaTerm"]


def o(*args: BuilderArg) -> Term:
    """
    Combine multiple terms with function application.

    Note that the arguments are combined in a left-associative way, in the same way as in Lisp, Ocaml or Haskell.
    For example, `o(f, g, h)` would be written `f(g)(h)` in python, or `Apply(Apply(f, g), h)`
    """
    if len(args) == 0:
        raise ValueError("o() requires at least one argument")

    result = _to_term(args[0])
    for f in args[1:]:
        result = App(result, _to_term(f))

    return result


class L:
    def __init__(self, *names: Union[str, Var]):
        self.names = list(names)

    def o(self, *args: BuilderArg) -> LambdaTerm:
        """
        Provide the body of the lambda-expression.

        You can provide either a single term, or multiple terms that will be compined with function application.

        Note that the arguments are combined in a left-associative way, in the same way as in Lisp, Ocaml or Haskell.
        For example, `L(x).o(f, g, h)` would be written `lambda x: f(g)(h)` in python, or `Apply(Apply(f, g), h)`
        """
        # Convert body to term with given context
        term = _to_term(o(*args))

        # Wrap in lambdas for each name
        for v in reversed(self.names):
            name = v if isinstance(v, str) else v.name
            assert isinstance(name, str)
            term = Lam.from_unbound_term(term, name)

        return LambdaTerm(term)


def _to_term(arg: BuilderArg) -> Term:
    """
    Convert a BuilderArg to a Term in the given context.

    Args:
        arg: Variable name, Application, Term, or LambdaTerm
        context: Stack of bound variable names (innermost last)

    Returns:
        A Term object (may contain free variables)
    """
    if isinstance(arg, str):
        # Variable reference - find its De Bruijn index or keep as free variable
        return Var(arg)
    elif issubclass(arg.__class__, Term):
        return arg  # type: ignore
    elif isinstance(arg, LambdaTerm):
        return arg.as_term()
    else:
        raise ValueError(f"Unknown type for lambda builder: {type(arg)}")


class Html:
    def __init__(self, content: str):
        self.content = content

    def _repr_html_(self):
        return self.content
