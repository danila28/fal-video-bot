from typing import Dict, Type, TypeVar

T = TypeVar("T")
container: Dict[Type[T], T] = {}


def inject(key: Type[T]) -> T:
    return container[key]


def instance(value: T) -> None:
    container[type(value)] = value
