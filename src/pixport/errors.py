"""pixport's errors: every damaged or hostile file ends in one of these, never in a crash."""


class PixportError(Exception):
    """A file pixport can't read or write: damaged, cut short, or not what it claims to be."""


class UnsupportedError(PixportError):
    """A valid file using something pixport doesn't read yet (a camera's compression, an EXR codec)."""
