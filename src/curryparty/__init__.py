from typing import Iterable, List, Optional, Union

try:
    import polars as pl
except ImportError:
    raise ImportError(
        "curryparty needs the `polars` library. \n Please install it, typically with `pip install polars`"
    )
from svg import SVG, Rect
import uuid

from .core import SCHEMA, beta_reduce, compose, find_redexes, find_variables, subtree
from .display import (
    compute_height,
    compute_svg_frame_final,
    compute_svg_frame_init,
    compute_svg_frame_phase_a,
    compute_svg_frame_phase_b,
)
from .utils import ShapeAnim

__all__ = ["L", "V"]


class Term:
    def __init__(self, nodes: pl.DataFrame):
        assert nodes.schema == SCHEMA, (
            f"{nodes.schema} is different from expected {SCHEMA}"
        )
        self.nodes = nodes
        self.lamb = None

    def __call__(self, other: "Term") -> "Term":
        return Term(compose(self.nodes, other.nodes))

    def beta(self) -> Optional["Term"]:
        candidates = find_redexes(self.nodes)
        if len(candidates) == 0:
            return None
        _redex, lamb, b = candidates.row(0)
        self.lamb = lamb
        self.b = b
        reduced = beta_reduce(self.nodes, lamb, b)
        return Term(reduced)

    def reduce(self):
        last_non_reduced = self
        for term in self.reduction_chain():
            last_non_reduced = term
        return last_non_reduced

    def reduction_chain(self) -> Iterable["Term"]:
        term = self
        while True:
            yield term
            term = term.beta()
            if term is None:
                break

    def show_beta(self, x0=-10, width=30):
        """
        Generates an HTML representation that toggles visibility between
        a static state and a SMIL animation on hover using pure CSS.
        """
        candidates = find_redexes(self.nodes)
        if len(candidates) == 0:
            return self._repr_html_()

        _redex, lamb, b = candidates.row(0)
        new_nodes = beta_reduce(self.nodes, lamb, b)
        vars = find_variables(self.nodes, lamb)["id"]
        b_subtree = subtree(self.nodes, b)
        height = compute_height(self.nodes) + 3
        shapes: dict[int, ShapeAnim] = {}
        N_STEPS = 6

        for t in range(N_STEPS):
            if t == 0:
                items = compute_svg_frame_init(self.nodes)
            elif t == 1 or t == 2:
                items = compute_svg_frame_phase_a(self.nodes, lamb, b_subtree, vars)
            elif t == 3 or t == 4:
                items = compute_svg_frame_phase_b(
                    self.nodes, lamb, b_subtree, new_nodes
                )
            else:
                items = compute_svg_frame_final(new_nodes)
            for k, e, attributes in items:
                if k not in shapes:
                    shapes[k] = ShapeAnim(e)
                shapes[k].append_frame(t, attributes.items())

        figure_id = uuid.uuid4()
        box_id = f"lambda_box_{figure_id}".replace("-", "")
        anim_elements = [
            x.to_element(N_STEPS, begin=f"{box_id}.click", reset=f"{box_id}.mouseover")
            for x in shapes.values()
        ]

        anim_elements.append(
            Rect(
                id=box_id,
                x=str(x0),
                y="0",
                width="100%",
                height="100%",
                fill="transparent",
            )
        )

        anim_svg = SVG(
            xmlns="http://www.w3.org/2000/svg",
            viewBox=f"{x0} 0 {width} {height}",
            height=f"{35 * height}px",
            elements=anim_elements,
        ).as_str()

        return Html(
            f"<div><div style=\"margin:5px\">click to animate, move away and back to reset</div>{anim_svg}</div>"
        )

    def _repr_html_(self, x0=-10, width=30):
        frame = compute_svg_frame_init(self.nodes)
        elements = []

        height = compute_height(self.nodes) + 1
        for _, e, attributes in frame:
            for name, v in attributes.items():
                e.__setattr__(name, v)
            elements.append(e)
        return SVG(
            xmlns="http://www.w3.org/2000/svg",
            viewBox=f"{x0} 0 {width} {height}",  # type: ignore
            height=f"{35 * height}px",  # type: ignore
            elements=elements,
        ).as_str()


class L:
    def __init__(self, *lambda_names):
        self.n = len(lambda_names)
        self.lambdas = {x: i for (i, x) in enumerate(lambda_names)}
        self.refs: List[tuple[int, Union[int, str]]] = []
        self.args = []
        self.last_ = None

    def lamb(self, name: str) -> "L":
        self.n += 1
        self.lambdas[name] = self.n
        return self

    def _append_subtree_or_subexpression(self, t: Union[str, "L"]):
        if isinstance(t, L):
            offset = self.n
            for i, x in t.refs:
                self.refs.append((offset + i, t.lambdas.get(x, x)))

            for i, x in t.args:
                self.args.append((offset + i, offset + x))
            self.n += t.n
        else:
            assert isinstance(t, str)
            self.refs.append((self.n, t))
            self.n += 1

    def _(self, x: Union[str, "L"]) -> "L":
        self.last_ = self.n
        self._append_subtree_or_subexpression(x)
        return self

    def call(self, arg: Union[str, "L"]) -> "L":
        assert self.last_ is not None
        self.refs = [(i + 1, x) if i >= self.last_ else (i, x) for (i, x) in self.refs]
        self.args = [
            (i + 1, x + 1) if i >= self.last_ else (i, x) for (i, x) in self.args
        ]

        self.n += 1
        self.args.append((self.last_, self.n))
        self._append_subtree_or_subexpression(arg)

        return self

    def build(self) -> "Term":
        self.refs = [(i, self.lambdas.get(x, x)) for i, x in self.refs]
        ref = pl.from_records(
            self.refs, orient="row", schema={"id": pl.UInt32, "ref": pl.UInt32}
        )
        arg = pl.from_records(
            self.args, orient="row", schema={"id": pl.UInt32, "arg": pl.UInt32}
        )
        data = (
            pl.Series("id", range(self.n), dtype=pl.UInt32)
            .to_frame()
            .join(ref, on="id", how="left")
            .join(arg, on="id", how="left")
        ).with_columns(bid=pl.struct(major="id", minor="id"))
        return Term(data)


def V(name: str) -> L:
    return L()._(name)


class Html:
    def __init__(self, content: str):
        self.content = content

    def _repr_html_(self):
        return self.content
