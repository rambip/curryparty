from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Iterable, Optional

from svg import Animate, Element


@dataclass
class Interval:
    values: Optional[tuple[int, int]]

    def __or__(self: "Interval", other: "Interval") -> "Interval":
        if self.values is None:
            if other.values is None:
                return Interval(None)
            else:
                return other
        else:
            if other.values is None:
                return self
        return Interval(
            (min(self.values[0], other.values[0]), max(self.values[1], other.values[1]))
        )

    def __getitem__(self, index):
        assert self.values is not None, "interval is empty"
        return self.values[index]

    def shift(self, offset: int) -> "Interval":
        if self.values is None:
            return Interval(None)
        else:
            return Interval((self.values[0] + offset, self.values[1] + offset))


@dataclass
class ShapeAnim:
    shape: Element
    attributes: set[str]
    values: dict[tuple[int, str], Any]
    n: int
    duration: int

    def __init__(
        self,
        shape: Element,
        duration: int = 7,
    ):
        self.shape = shape
        self.attributes = set()
        self.values = {}
        self.duration = duration

    def append_frame(self, i: int, attributes: Iterable[tuple[str, Any]]):
        for name, v in attributes:
            self.attributes.add(name)
            self.values[i, name] = v

    def to_element(self, n: int, begin: str, reset: str):
        elements = []

        visible = [
            all((i, name) in self.values for name in self.attributes) for i in range(n)
        ]

        for name in self.attributes:
            non_nulls = [
                self.values[i, name] for i in range(n) if (i, name) in self.values
            ]
            for i in reversed(range(n)):
                if (i, name) in self.values:
                    break
                self.values[i, name] = non_nulls[-1]
            for i in range(n):
                if (i, name) not in self.values:
                    self.values[i, name] = non_nulls[0]
            self.shape.__setattr__(name, non_nulls[0])

            elements.append(
                Animate(
                    attributeName=name,
                    values=";".join(str(self.values[i, name]) for i in range(n)),
                    dur=timedelta(seconds=self.duration),
                    begin=begin,
                    fill="freeze",
                    repeatCount="1",
                )
            )
            elements.append(
                Animate(
                    attributeName=name,
                    values=f"{non_nulls[0]}",
                    dur=0,
                    begin=reset,
                )
            )
        elements.append(
            Animate(
                attributeName="opacity",
                values=";".join("1" if v else "0" for v in visible),
                dur=timedelta(seconds=self.duration),
                begin=begin,
                fill="freeze",
                repeatCount="1",
            )
        )
        elements.append(
            Animate(
                attributeName="opacity",
                values="1" if visible[0] else "0",
                dur=0,
                begin=reset,
            )
        )

        assert not self.shape.elements
        self.shape
        self.shape.elements = elements
        return self.shape
