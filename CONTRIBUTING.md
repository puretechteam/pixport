# Contributing

Thanks for helping. Bug reports, new cameras and fixes are all welcome.

## Reporting a bug

Open an issue with the file that fails, if you can share it (for a RAW photo, the camera it came from), what you expected, what you got, and whether pixport-native is installed (`pip show pixport-native`).

Security problems go through [SECURITY.md](SECURITY.md) instead, not a public issue.

## Making a change

```
python -m venv .venv
.venv\Scripts\pip install -e .[dev]
.venv\Scripts\python -m pytest
.venv\Scripts\ruff check .
```

Set `PIXPORT_PURE=1` to run the suite in pure Python: the tests that need pixport-native skip. GitHub Actions runs the tests (Python 3.10, 3.11 and 3.13, on Windows and Linux) and `ruff check` on every pull request.

- **Write from published descriptions.** No GPL or LGPL code goes into pixport. LibRaw, OpenEXR and djxl are for checking results, as black boxes.
- **The same results both ways.** Whatever pixport-native computes, the NumPy code has to compute too, except JPEG XL, and a test has to compare the two.
- **Hostile files:** readers must raise `PixportError` and nothing else, and never allocate what a file's header merely claims.
- **Check against real tools.** New output should be read back by an independent reader (Pillow, OpenCV), new input checked against one (LibRaw, OpenEXR, djxl).
- **Changelog:** add a dated entry to [CHANGELOG.md](CHANGELOG.md) for anything a user would notice.

## License

By contributing, you agree that your contribution is licensed under the [MIT License](LICENSE).
