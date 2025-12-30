from typing import Iterable, List, NamedTuple, Optional

import svg


class NodeInfo(NamedTuple):
    id: int
    type: str
    ref: Optional[int]  # For variables: which lambda they reference
    parent_lambda: Optional[int]  # Enclosing lambda id
    parent_id: Optional[int]  # Immediate parent node id
    is_left_child: bool
    is_right_child: bool


class Term:
    def __call__(self, arg: "Term") -> "Term":
        return Application(self, arg)

    def replace(self, d: int, v: "Term") -> "Term": ...

    def _beta(self) -> tuple["Term", bool]: ...

    def _eta(self) -> "Term":
        return self

    def bump_free_vars(self, cutoff: int = 0) -> "Term": ...

    def nodes(
        self,
        next_id: list[int],
        context: List[int],
        parent_id: Optional[int],
        is_left_child: bool,
        is_right_child: bool,
    ) -> Iterable[NodeInfo]: ...


class Lambda(Term):
    def __init__(self, body: Term):
        self.body = body

    def __repr__(self):
        return f"\\ {self.body}"

    def _beta(self) -> tuple[Term, bool]:
        result, changed = self.body._beta()
        return Lambda(result), changed

    def replace(self, d: int, v: Term):
        return Lambda(self.body.replace(d + 1, v.bump_free_vars(cutoff=0)))

    def bump_free_vars(self, cutoff: int = 0) -> Term:
        return Lambda(self.body.bump_free_vars(cutoff + 1))

    def _eta(self) -> Term:
        # FIXME: There must be no variable that refers to this lambda.
        if isinstance(self.body, Application):
            if self.body.right == Variable(0):
                return self.body.left
        return self

    def nodes(
        self,
        next_id: list[int],
        context: List[int],
        parent_id: Optional[int],
        is_left_child: bool,
        is_right_child: bool,
    ) -> Iterable[NodeInfo]:
        my_id = next_id[0]
        next_id[0] += 1
        parent_lambda = context[-1] if context else None

        yield NodeInfo(
            id=my_id,
            type="lambda",
            ref=None,
            parent_lambda=parent_lambda,
            parent_id=parent_id,
            is_left_child=is_left_child,
            is_right_child=is_right_child,
        )
        yield from self.body.nodes(next_id, context + [my_id], my_id, False, False)


class Variable(Term):
    def __init__(self, depth: int):
        self.depth = depth

    def _beta(self) -> tuple[Term, bool]:
        return self, False

    def replace(self, d: int, v: Term):
        if self.depth == d:
            return v
        elif self.depth > d:
            return Variable(self.depth - 1)  # Decrement free variables
        else:
            return self

    def __repr__(self):
        return f"{self.depth}"

    def __eq__(self, other):
        return isinstance(other, Variable) and self.depth == other.depth

    def nodes(
        self,
        next_id: list[int],
        context: List[int],
        parent_id: Optional[int],
        is_left_child: bool,
        is_right_child: bool,
    ) -> List[NodeInfo]:
        my_id = next_id[0]
        next_id[0] += 1
        parent_lambda = context[-1] if context else None
        ref = context[-1 - self.depth] if self.depth < len(context) else None

        node = NodeInfo(
            id=my_id,
            type="variable",
            ref=ref,
            parent_lambda=parent_lambda,
            parent_id=parent_id,
            is_left_child=is_left_child,
            is_right_child=is_right_child,
        )
        return [node]

    def bump_free_vars(self, cutoff: int = 0) -> Term:
        if self.depth >= cutoff:
            return Variable(self.depth + 1)
        return self


class Application(Term):
    def __init__(self, left: Term, right: Term):
        self.left = left
        self.right = right

    def replace(self, d: int, v: Term):
        return Application(self.left.replace(d, v), self.right.replace(d, v))

    def bump_free_vars(self, cutoff: int = 0) -> Term:
        return Application(
            self.left.bump_free_vars(cutoff), self.right.bump_free_vars(cutoff)
        )

    def _beta(self) -> tuple[Term, bool]:
        # If I'm a redex, reduce me
        if isinstance(self.left, Lambda):
            result = self.left.body.replace(0, self.right)
            return result, True

        # Otherwise try left subtree
        new_left, left_changed = self.left._beta()
        if left_changed:
            return Application(new_left, self.right), True

        # Then try right subtree
        new_right, right_changed = self.right._beta()
        if right_changed:
            return Application(self.left, new_right), True

        return self, False

    def __repr__(self):
        return f"({self.left} {self.right})"

    def __call__(self, arg: Term) -> Term:
        return Application(self, arg)

    def nodes(
        self,
        next_id: list[int],
        context: List[int],
        parent_id: Optional[int],
        is_left_child: bool,
        is_right_child: bool,
    ) -> Iterable[NodeInfo]:
        my_id = next_id[0]
        next_id[0] += 1
        parent_lambda = context[-1] if context else None

        yield NodeInfo(
            id=my_id,
            type="application",
            ref=None,
            parent_lambda=parent_lambda,
            parent_id=parent_id,
            is_left_child=is_left_child,
            is_right_child=is_right_child,
        )

        # Left child
        yield from self.left.nodes(next_id, context, my_id, True, False)

        # Right child
        yield from self.right.nodes(next_id, context, my_id, False, True)


def compute_y_positions(nodes: List[NodeInfo]) -> dict[int, int]:
    """Forward pass: compute y positions and currying depth top-down"""
    y = {}

    for node in nodes:
        if node.parent_id is None:
            y[node.id] = 0
            continue

        if node.is_right_child:
            y[node.id] = y[node.parent_id]
            continue
        if node.is_left_child and node.type != "application":
            y[node.id] = y[node.parent_id]
            continue

        y[node.id] = y[node.parent_id] + 1

    return y


def compute_x_extents(nodes: List[NodeInfo]) -> tuple[dict[int, int], dict[int, int]]:
    """Backward pass: compute x extents bottom-up"""
    x_min = {}
    x_max = {}
    next_var = 0

    for node in reversed(nodes):
        if node.type == "variable":
            # Assign variable index
            x_min[node.id] = next_var
            x_max[node.id] = next_var
            next_var -= 1

            # Update the lambda this variable references
            if node.ref is not None:
                if node.ref not in x_min:
                    x_min[node.ref] = x_min[node.id]
                    x_max[node.ref] = x_max[node.id]
                else:
                    x_min[node.ref] = min(x_min[node.ref], x_min[node.id])
                    x_max[node.ref] = max(x_max[node.ref], x_max[node.id])

        # Update parent
        if node.is_left_child:
            if node.parent_id not in x_min:
                x_min[node.parent_id] = x_min[node.id]
                x_max[node.parent_id] = x_max[node.id]

        elif node.is_right_child:
            if node.parent_id in x_min:
                x_min[node.parent_id] = min(x_min[node.parent_id], x_min[node.id])
                x_max[node.parent_id] = max(x_max[node.parent_id], x_max[node.id])

        elif node.parent_id is not None:
            if node.parent_id in x_min:
                x_min[node.parent_id] = min(x_min[node.parent_id], x_min[node.id])
                x_max[node.parent_id] = max(x_max[node.parent_id], x_max[node.id])
            else:
                x_min[node.parent_id] = x_min[node.id]
                x_max[node.parent_id] = x_max[node.id]

    return x_min, x_max


def display(term: Term) -> svg.SVG:
    nodes = list(term.nodes([0], [], None, False, False))
    y = compute_y_positions(nodes)
    x_min, x_max = compute_x_extents(nodes)
    elements = []

    for node in nodes:
        if node.is_right_child:
            assert node.parent_id is not None
            target_x = x_max[node.parent_id]
            target_y = y[node.parent_id]
            elements.append(
                svg.Line(
                    x1=0.1 + x_min[node.id],
                    y1=0.5 + y[node.id],
                    x2=0.5 + target_x,
                    y2=0.5 + target_y,
                    stroke="black",
                    stroke_width=0.05,
                )
            )
            s = 0.1
            elements.append(
                svg.Polygon(
                    points=[
                        0.5 + target_x + s,
                        0.5 + target_y - s,
                        0.5 + target_x - s,
                        0.5 + target_y,
                        0.5 + target_x + s,
                        0.5 + target_y + s,
                    ],
                    stroke_width=0,
                    fill="black",
                    # fill="black",
                )
            )
        if node.type == "lambda":
            elements.append(
                svg.Rect(
                    x=x_min[node.id] + 0.1,
                    y=y[node.id] + 0.1,
                    width=1 + x_max[node.id] - x_min[node.id] - 0.2,
                    height=0.8,
                    fill="blue",
                )
            )
        elif node.type == "variable":
            elements.append(
                svg.Rect(
                    x=x_min[node.id] + 0.1,
                    y=y[node.id] + 0.1,
                    width=0.8,
                    height=0.8,
                    fill="red",
                )
            )
            if node.ref is not None:
                elements.append(
                    svg.Line(
                        x1=x_min[node.id] + 0.5,
                        y1=y[node.id] + 0.1,
                        x2=x_min[node.id] + 0.5,
                        y2=y[node.ref] + 0.9,
                        stroke_width=0.05,
                        stroke="gray",
                    )
                )
        elif node.type == "application":
            elements.append(
                svg.Rect(
                    x=x_min[node.id] + 0.1,
                    y=y[node.id] + 0.1,
                    width=1 + x_max[node.id] - x_min[node.id] - 0.2,
                    height=0.8,
                    fill_opacity=0.5,
                    stroke="orange",
                    stroke_width=0.1,
                    fill="none",
                )
            )

    return svg.SVG(
        viewBox="-10 0 20 10",  # type: ignore
        height="400px",  # type: ignore
        elements=elements,
    )
