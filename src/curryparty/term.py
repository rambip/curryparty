"""
Lambda calculus term representation using De Bruijn indices.

This module provides a clean, shared interface for representing lambda calculus terms
that can be used by different evaluation engines.

The representation uses De Bruijn indices for variable binding, which eliminates
the need for alpha-conversion and makes substitution straightforward.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Term", "Var", "Lam", "App"]


@dataclass(frozen=True)
class Term:
    """
    Base class for lambda calculus terms.

    This is an immutable representation that can be easily shared
    and converted to different internal representations for evaluation.
    """

    def __call__(self, arg: Term) -> Term:
        """Apply this term to an argument"""
        return App(self, arg)


@dataclass(frozen=True)
class Var(Term):
    """
    A variable reference using De Bruijn index.

    The index indicates how many lambda binders to traverse upward
    to find the binding lambda:
    - index=0: bound by the immediately enclosing lambda
    - index=1: bound by the next outer lambda
    - etc.

    Example:
        λx. λy. x  =>  Lam(Lam(Var(1)))  # x is 1 level up from inner lambda
        λx. λy. y  =>  Lam(Lam(Var(0)))  # y is bound by immediate lambda

    Attributes:
        index: De Bruijn index (0-based, counting from innermost lambda)
    """

    index: int

    def __post_init__(self):
        if self.index < 0:
            raise ValueError(f"Variable index must be non-negative, got {self.index}")


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


# Convenience functions for building terms


def var(index: int = 0) -> Var:
    """Create a variable with given De Bruijn index"""
    return Var(index)


def lam(body: Term) -> Lam:
    """Create a lambda abstraction"""
    return Lam(body)


def app(func: Term, arg: Term) -> App:
    """Create an application"""
    return App(func, arg)
