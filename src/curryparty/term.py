"""
Lambda calculus term representation using De Bruijn indices.

This module provides a clean, shared interface for representing lambda calculus terms
that can be used by different evaluation engines.

The representation uses De Bruijn indices for variable binding, which eliminates
the need for alpha-conversion and makes substitution straightforward.

A variable can be either:
- A bound variable (De Bruijn index as int)
- A free variable (string name)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator, Union

__all__ = ["Term", "Var", "Lam", "App"]


class Term(ABC):
    """
    Base class for lambda calculus terms.

    This is an immutable representation that can be easily shared
    and converted to different internal representations for evaluation.
    """

    def __call__(self, arg: Term) -> Term:
        """Apply this term to an argument"""
        return App(self, arg)

    @abstractmethod
    def bind_var(self, var_name: str, depth: int = 0) -> Term: ...

    @abstractmethod
    def free_vars(self) -> Generator[str]: ...

    @abstractmethod
    def is_valid(self, depth: int) -> bool:
        """
        Check if the De-Bruijn indices are all valid.
        Not that a term can be valid and contain free variables.

        ```
        Lam(Var(0)) # valid

        Lam(Var(1)) # invalid

        Lam(Var("x")) # valid, but with a free variable
        ```
        """
        ...

    def is_fully_bound(self) -> bool:
        """
        Check if all variables in this term are bound (no free variables).

        Returns:
            True if all variables have De Bruijn indices (bound),
            False if any variable has a string name (free).
        """
        return len(list(self.free_vars())) == 0


@dataclass(frozen=True)
class Var(Term):
    """
    A variable reference.

    Can be either:
    - A bound variable with De Bruijn index (int)
    - A free variable with a name (str)

    For bound variables, the index indicates how many lambda binders to traverse upward
    to find the binding lambda:
    - index=0: bound by the immediately enclosing lambda
    - index=1: bound by the next outer lambda
    - etc.

    Example:
        λx. λy. x  =>  Lam(Lam(Var(1)))  # x is 1 level up from inner lambda
        λx. λy. y  =>  Lam(Lam(Var(0)))  # y is bound by immediate lambda
        λx. y      =>  Lam(Var("y"))     # y is a free variable

    Attributes:
        name: Either a De Bruijn index (int) for bound variables,
              or a string name for free variables
    """

    name: Union[int, str]

    def bind_var(self, var_name: str, depth: int = 0) -> Term:
        if self.name == var_name:
            return Var(depth)
        return self

    def free_vars(self) -> Generator[str]:
        if isinstance(self.name, str):
            yield self.name

    def is_valid(self, depth) -> bool:
        if isinstance(self.name, int):
            return self.name <= depth
        return True

    def __post_init__(self):
        if isinstance(self.name, int) and self.name < 0:
            raise ValueError(f"Variable index must be non-negative, got {self.name}")

    @property
    def is_bound(self) -> bool:
        """Check if this variable is bound (has De Bruijn index)."""
        return isinstance(self.name, int)

    @property
    def is_free(self) -> bool:
        """Check if this variable is free (has string name)."""
        return isinstance(self.name, str)

    @property
    def index(self) -> int:
        """Get the De Bruijn index. Raises if this is a free variable."""
        if isinstance(self.name, str):
            raise ValueError(f"Cannot get index of free variable '{self.name}'")
        return self.name


@dataclass(frozen=True)
class Lam(Term):
    """
    Lambda abstraction.

    Represents an anonymous function that binds one variable.
    The body may reference this variable using Var(0).

    Example:
        λx. x      =>  Lam(Var(0))           # identity function
        λx. λy. x  =>  Lam(Lam(Var(1)))      # const function

    Attributes:
        body: The body of the lambda abstraction
    """

    body: Term

    def free_vars(self) -> Generator[str]:
        return self.body.free_vars()

    def bind_var(self, var_name: str, depth: int = 0) -> Term:
        return Lam(self.body.bind_var(var_name, depth + 1))

    def is_valid(self, depth: int) -> bool:
        return self.body.is_valid(depth + 1)

    @staticmethod
    def from_unbound_term(term: Term, var_name: str) -> Term:
        return Lam(term.bind_var(var_name, 0))


@dataclass(frozen=True)
class App(Term):
    """
    Function application.

    Represents applying a function to an argument.

    Example:
        (λx. x) y  =>  App(Lam(Var(0)), Var(0))  # applying identity to y

    Attributes:
        func: The function being applied
        arg: The argument to apply
    """

    func: Term
    arg: Term

    def bind_var(self, var_name: str, depth: int = 0) -> Term:
        return App(
            self.func.bind_var(var_name, depth), self.arg.bind_var(var_name, depth)
        )

    def free_vars(self) -> Generator[str]:
        yield from self.func.free_vars()
        yield from self.arg.free_vars()

    def is_valid(self, depth: int) -> bool:
        return self.func.is_valid(depth) and self.arg.is_valid(depth)
