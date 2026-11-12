"""How big a picture pixport agrees to decode. A header is a few bytes, and a hostile one can claim
a picture of any size: every reader checks the claimed size against this budget before it
allocates anything for it.

The default, 2^28 pixels (268 million, a 16384 x 16384 picture), is above Pillow's own refusal
(about 179 million) and well above any camera. ``pixport.read(path, max_pixels=n)`` changes it for
one call; setting ``pixport.limits.MAX_PIXELS`` changes it for the process.
"""

from __future__ import annotations

import contextvars

from .errors import PixportError

MAX_PIXELS = 1 << 28

_override: contextvars.ContextVar[int | None] = contextvars.ContextVar("pixport_max_pixels", default=None)


class TooLargeError(PixportError):
    """A picture larger than the pixel budget (``max_pixels``)."""


def max_pixels() -> int:
    """The pixel budget in force: the one passed to ``read``, else ``MAX_PIXELS``."""
    n = _override.get()
    return int(MAX_PIXELS if n is None else n)


def check(width: int, height: int, what: str = "picture") -> None:
    """Refuse a ``width`` x ``height`` picture above the pixel budget."""
    budget = max_pixels()
    if width * height > budget:
        raise TooLargeError(f"The {what} ({width:,} x {height:,} pixels) is larger than pixport decodes "
                            f"({budget:,} pixels; pass max_pixels= to allow more)")


class budget:
    """``with budget(n):`` reads pictures up to ``n`` pixels (None keeps the current budget)."""

    def __init__(self, n: int | None):
        self.n = n
        self.token = None

    def __enter__(self):
        if self.n is not None:
            if int(self.n) < 1:
                raise ValueError("max_pixels must be at least 1")
            self.token = _override.set(int(self.n))
        return self

    def __exit__(self, *exc):
        if self.token is not None:
            _override.reset(self.token)
        return False
