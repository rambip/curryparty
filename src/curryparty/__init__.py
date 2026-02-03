from typing import Iterable, List, Optional, Union

try:
    import polars as pl
except ImportError:
    raise ImportError(
        "curryparty needs the `polars` library. \n Please install it, typically with `pip install polars`"
    )
import uuid

from svg import SVG, Rect

from .core import SCHEMA, beta_reduce, compose, find_redexes, find_variables, subtree
from .display import (
    compute_height,
    compute_svg_frame_final,
    compute_svg_frame_init,
    compute_svg_frame_phase_a,
    compute_svg_frame_phase_b,
    count_variables,
)
from .utils import ShapeAnim, ShapeAnimFrame

__all__ = ["L", "V"]


def log2(n):
    if n <= 0:
        raise ValueError(f"log2 of negative number {n}")
    elif n == 1:
        return 0
    return 1 + log2(n // 2)


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
        candidates = find_redexes(self.nodes).first().collect()
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

    def show_beta(self, duration=7):
        """
        Generates an HTML representation that toggles visibility between
        a static state and a SMIL animation on hover using pure CSS.
        """
        candidates = find_redexes(self.nodes).first().collect()
        if len(candidates) == 0:
            return self._repr_html_()

        _redex, lamb, b = candidates.row(0)
        new_nodes = beta_reduce(self.nodes, lamb, b)
        vars = find_variables(self.nodes, lamb).collect()["id"]
        b_subtree = subtree(self.nodes, b).collect()
        height = min(compute_height(self.nodes), compute_height(new_nodes)) * 2
        if count_variables(self.nodes) == 0:
            return "no width"
        raw_width = max(count_variables(self.nodes), count_variables(new_nodes))
        width = 1 << (1 + log2(raw_width))
        frame_data: list[ShapeAnimFrame] = []
        N_STEPS = 6

        for t in range(N_STEPS):
            if t == 0:
                items = compute_svg_frame_init(self.nodes, t)
            elif t == 1 or t == 2:
                items = compute_svg_frame_phase_a(self.nodes, lamb, b_subtree, vars, t)
            elif t == 3 or t == 4:
                items = compute_svg_frame_phase_b(
                    self.nodes, lamb, b_subtree, new_nodes, t
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
                x=f"{-width}",
                y="0",
                width="100%",
                height="100%",
                fill="transparent",
            )
        )

        # prefered size in pixels
        H = height * 40
        anim_svg = SVG(
            xmlns="http://www.w3.org/2000/svg",
            viewBox=f"{-width} 0 {2 * width} {height}",
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

    def _repr_html_(self, x0=-10):
        frame = sorted(compute_svg_frame_init(self.nodes), key=lambda x: x.zindex)

        width = (1 << (1 + log2(count_variables(self.nodes)))) + 4
        height = compute_height(self.nodes) + 1

        elements = [ShapeAnim.from_single_frame(x) for x in frame]

        # prefered size in pixels
        H = height * 40
        W = width * 40

        svg_element = SVG(
            xmlns="http://www.w3.org/2000/svg",
            viewBox=f"{-1} 0 {width} {height}",  # type: ignore
            elements=elements,
            style=f"max-height:{H}px; margin-left: clamp(0px, calc(100% - {W}px), 100px)",
        ).as_str()
        return f"<div>{svg_element}</div>"


def offset_var(x: Union[int, str], offset: int) -> Union[int, str]:
    if isinstance(x, int):
        return x + offset
    return x


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

    def _append_subtree_or_subexpression(self, t: Union[str, "L", Term]):
        offset = self.n
        if isinstance(t, L):
            for i, x in t.refs:
                self.refs.append((offset + i, offset_var(t.lambdas.get(x, x), offset)))

            for i, x in t.args:
                self.args.append((offset + i, offset + x))
            self.n += t.n
        elif isinstance(t, Term):
            for i, x in t.nodes.select("id", "ref").drop_nulls().iter_rows():
                self.refs.append((offset + i, offset + x))
            for i, x in t.nodes.select("id", "arg").drop_nulls().iter_rows():
                self.args.append((offset + i, offset + x))
            self.n += len(t.nodes)
        else:
            assert isinstance(t, str)
            self.refs.append((self.n, t))
            self.n += 1

    def _(self, x: Union[str, "L", Term]) -> "L":
        self.last_ = self.n
        self._append_subtree_or_subexpression(x)
        return self

    def call(self, arg: Union[str, "L", Term]) -> "L":
        assert self.last_ is not None
        self.refs = [
            (i + 1, offset_var(x, 1)) if i >= self.last_ else (i, x)
            for (i, x) in self.refs
        ]
        self.args = [
            (i + 1, x + 1) if i >= self.last_ else (i, x) for (i, x) in self.args
        ]

        self.n += 1
        self.args.append((self.last_, self.n))
        self._append_subtree_or_subexpression(arg)

        return self

    def build(self) -> "Term":
        def bind_var(x: Union[str, int]) -> int:
            # check that all remaining unbound variables are bound to this lambda
            if isinstance(x, int):
                return x
            if x not in self.lambdas:
                raise ValueError(f"variable {x} is not bound to any lambda")
            return self.lambdas[x]

        self.refs = [(i, bind_var(x)) for i, x in self.refs]
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
