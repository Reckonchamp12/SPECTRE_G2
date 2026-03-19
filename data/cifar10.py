from .synthetic import make_synthetic
from .adult     import make_adult
from .cifar10   import make_cifar10
from .gridworld import make_gridworld

__all__ = ["make_synthetic", "make_adult", "make_cifar10", "make_gridworld"]
