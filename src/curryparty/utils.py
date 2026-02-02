from dataclasses import dataclass, field
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
class ShapeAnimFrame:
    element: Element
    key: Any
    idx: int
    attrs: dict[str, Any]
    zindex: int = 0

    def apply_attributes(self):
        for name, v in self.attrs.items():
            self.element.__setattr__(name, v)


@dataclass
class ShapeAnim:
    key: Any
    element: Element
    zindex: int
    duration: int
    attributes: set[str] = field(default_factory=set)
    values: dict[tuple[int, str], Any] = field(default_factory=dict)

    @staticmethod
    def from_single_frame(frame: ShapeAnimFrame) -> Element:
        frame.apply_attributes()
        return frame.element

    @staticmethod
    def from_frames(frames: list[ShapeAnimFrame], duration: int = 7) -> "ShapeAnim":
        if not frames:
            raise ValueError("frames list cannot be empty")

        frames.sort(key=lambda f: f.idx)

        zindex = frames[0].zindex
        for f in frames[1:]:
            if f.zindex != zindex:
                raise ValueError(
                    f"zindex mismatch for key {frames[0].key}: got {f.zindex}, expected {zindex}"
                )

        anim = ShapeAnim(
            key=frames[0].key,
            element=frames[0].element,
            zindex=zindex,
            duration=duration,
        )

        for f in frames:
            for name, v in f.attrs.items():
                anim.attributes.add(name)
                anim.values[f.idx, name] = v

        return anim

    @staticmethod
    def group_by_key(
        frames: Iterable[ShapeAnimFrame],
    ) -> dict[Any, list[ShapeAnimFrame]]:
        groups: dict[Any, list[ShapeAnimFrame]] = {}
        for frame in frames:
            if frame.key not in groups:
                groups[frame.key] = []
            groups[frame.key].append(frame)
        return groups

    def append_frame(self, i: int, attributes: Iterable[tuple[str, Any]]):
        for name, v in attributes:
            self.attributes.add(name)
            self.values[i, name] = v

    def to_element(self, n: int, begin: str, reset: str) -> Element:
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
            self.element.__setattr__(name, non_nulls[0])

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

        assert not self.element.elements
        self.element.elements = elements
        return self.element
