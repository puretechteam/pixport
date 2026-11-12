"""pixport's compiled decoders (the optional pixport-native package), when they're installed.

Each module keeps its NumPy code and takes the compiled one in its place when ``native`` has it:
the same results, faster. ``PIXPORT_PURE=1`` keeps pixport on its own code (the tests run both
ways); a pixport-native of another version is left alone, since its decoders may not match.
JPEG XL is the exception: it is only read and written through pixport-native."""

from __future__ import annotations

import os

VERSION = "0.2.0"  # the pixport-native this pixport is made for (it ships with each release)

native = None
if not os.environ.get("PIXPORT_PURE"):
    try:
        import pixport_native as _native
    except ImportError:
        pass
    else:
        if getattr(_native, "__version__", None) == VERSION:
            native = _native


def accelerated() -> bool:
    """Whether the compiled decoders are in use."""
    return native is not None


def has(name: str) -> bool:
    return native is not None and hasattr(native, name)
