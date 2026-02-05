"""The `curryparty` library, a playground to learn lambda-calculus.

This library is intended to be used in an interactive
"""

from typing import Iterable, Optional, Union

try:
    import polars as pl
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
from .term import Term, app, lam, var
from .utils import ShapeAnim, ShapeAnimFrame

__all__ = ["L", "o"]


def log2(n):
    if n <= 0:
        raise ValueError(f"log2 of negative number {n}")
    elif n == 1:
        return 0
    return 1 + log2(n // 2)


class LambdaTerm:
    def __init__(self, data: AbstractTerm):
        self.data = data

    def __call__(self, other: "LambdaTerm") -> "LambdaTerm":
        return LambdaTerm((self.data)(other.data))

    def beta(self) -> Optional["LambdaTerm"]:
        candidates = self.data.find_redexes()
        redex = next(candidates, None)
        if redex is None:
            return None
        reduced = self.data.beta_reduce(redex)
        return LambdaTerm(reduced)

    def reduce(self):
        last_non_reduced = self
        for term in self.reduction_chain():
            last_non_reduced = term
        return last_non_reduced

    def reduction_chain(self) -> Iterable["LambdaTerm"]:
        term = self
        while term is not None:
            yield term
            term = term.beta()

    def __str__(self) -> str:
        """Convert to string representation using lambda notation."""
        from .term import App, Lam, Var

        lambda_counter = [0]  # Global counter for lambda indices

        def pretty(t, lambda_stack=[], paren=False):
            if isinstance(t, Var):
                # Get the lambda index this variable refers to
                if t.index < len(lambda_stack):
                    lam_idx = lambda_stack[-(t.index + 1)]
                    return f"x{lam_idx}"
                return f"x{t.index}"
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

        return pretty(self.data.to_term())

    def show_beta(self, duration=7):
        """
        Generates an HTML representation that toggles visibility between
        a static state and a SMIL animation on hover using pure CSS.
        """

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
        if count_variables(self.data) == 0:
            return "no width"
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


# Type alias for clarity
BuilderArg = Union[str, "L", "Application", Term, LambdaTerm]


class Application:
    """
    Represents a pending application that will be resolved in context.

    This is returned by o() when used standalone, and allows writing:
        L("f").o("x", o("y", "z"))
    where o("y", "z") creates an Application that gets resolved in f's context.
    """

    def __init__(self, *args: BuilderArg):
        self.args = args


def o(*args: BuilderArg) -> Application:
    """
    Create an application expression.

    This is meant to be used inside L(...).o(...) contexts:
        L("f", "x").o("f", o("f", "x"))  # λf. λx. f (f x)

    The o("f", "x") creates an Application that will be resolved
    in the context of the enclosing lambda.

    Args:
        *args: Variable names, lambda builders, nested applications, or Terms

    Returns:
        An Application object that will be resolved in context
    """
    if len(args) == 0:
        raise ValueError("o() requires at least one argument")

    return Application(*args)


class L:
    """
    Builder for lambda calculus terms with deferred evaluation.

    The key insight: We don't resolve variable names to De Bruijn indices
    until build() is called. This allows nested lambdas to reference outer
    variables correctly.

    Supports:
    - Multiple lambda bindings: L("x", "y", "z")
    - Variable references: L("x").o("x")
    - Applications: L("f").o("x", "y") applies f to x, then to y
    - Nested applications: L("f").o("x", o("y", "z"))
    - Nested lambdas: L("x").o(L("y").o("x"))
    - Shorthand for body: L("x")._("x") same as L("x").o("x")
    """

    def __init__(self, *names: str):
        """
        Create a lambda builder with bound variable names.

        Args:
            *names: Variable names to bind (creates nested lambdas)
                   L("x", "y") creates λx. λy. ...
        """
        self.names = list(names)
        # Store the body as raw BuilderArgs, not resolved Terms
        self.body_args: list[BuilderArg] = []

    def o(self, *args: BuilderArg) -> "L":
        """
        Set the body or apply arguments.

        This method handles:
        1. Setting the lambda body: L("x").o("x")
        2. Function application: L("f").o("x", "y") means f(x)(y)
        3. Nested applications: L("f").o("x", o("y", "z"))

        Args:
            *args: Variable names (strings), lambda builders (L),
                  applications (Application from o()), or Terms

        Returns:
            Self for chaining
        """
        if len(args) == 0:
            raise ValueError("o() requires at least one argument")

        # Store the arguments without resolving them yet
        self.body_args.extend(args)
        return self

    def _to_term(self, arg: BuilderArg, context: list[str]) -> Term:
        """
        Convert an argument to a Term in the given context.

        Args:
            arg: Variable name, lambda builder, application, or Term
            context: Stack of variable names (innermost last)

        Returns:
            A Term object
        """
        if isinstance(arg, str):
            # Variable reference - find its De Bruijn index
            return self._var_to_term(arg, context)
        elif isinstance(arg, L):
            # Nested lambda - build it with extended context
            return arg._build_with_context(context)
        elif isinstance(arg, Application):
            # Nested application - resolve in current context
            return self._application_to_term(arg, context)
        elif isinstance(arg, Term):
            return arg
        elif isinstance(arg, LambdaTerm):
            return arg.data.to_term()
        else:
            raise ValueError(f"unknown type for lambda builder: {type(arg)}")

    def _application_to_term(self, appl: Application, context: list[str]) -> Term:
        """
        Convert an Application to a Term in the given context.

        Args:
            appl: Application object from o()
            context: Stack of variable names

        Returns:
            A Term representing the application
        """
        result: Term | None = None
        for arg in appl.args:
            term = self._to_term(arg, context)
            if result is None:
                result = term
            else:
                result = app(result, term)

        if result is None:
            raise ValueError("Application produced no result")

        return result

    def _var_to_term(self, name: str, context: list[str]) -> Term:
        """
        Convert a variable name to a Var term with De Bruijn index.

        Searches from innermost to outermost lambda to find the binding.

        Args:
            name: Variable name to look up
            context: Stack of variable names (innermost last)

        Returns:
            Var term with appropriate De Bruijn index
        """
        # Search from the end (innermost lambda) backwards
        for i in range(len(context) - 1, -1, -1):
            if context[i] == name:
                # Found it! Calculate De Bruijn index
                # Distance from end of list
                index = len(context) - 1 - i
                return var(index)

        # Not found in current scope
        raise ValueError(
            f"Variable '{name}' not bound in current lambda scope: {context}"
        )

    def _build_with_context(self, outer_context: list[str]) -> Term:
        """
        Build this lambda term with outer variable context.

        This is the KEY method that fixes the scoping issue.
        We pass down the outer context so nested lambdas can reference
        outer variables.

        Args:
            outer_context: Variable names from outer lambdas

        Returns:
            Complete Term with correct De Bruijn indices
        """
        # Create extended context: outer + our names
        full_context = outer_context + self.names

        # Build body with full context
        if len(self.body_args) == 0:
            raise ValueError(
                "Cannot build lambda without a body. Use .o(...) to set the body."
            )

        # Convert body args to terms
        result: Term | None = None
        for arg in self.body_args:
            term = self._to_term(arg, full_context)
            if result is None:
                result = term
            else:
                result = app(result, term)

        if result is None:
            raise ValueError("Body produced no result")

        # Wrap in lambdas for our names
        for _ in self.names:
            result = lam(result)

        return result

    def build(self) -> LambdaTerm:
        """
        Build the actual Term from this builder.

        This is when variable resolution happens!

        Returns:
            The complete lambda term wrapped in LambdaTerm
        """
        result = self._build_with_context([])
        return LambdaTerm(AbstractTerm.from_term(result))


class Html:
    def __init__(self, content: str):
        self.content = content

    def _repr_html_(self):
        return self.content
