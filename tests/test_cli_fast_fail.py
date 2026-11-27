"""Tests for CLI fast-fail validation before decoding source images."""

from __future__ import annotations

import pytest

import pixport.__main__ as cli


@pytest.mark.parametrize("ext", ["exr", "hdr", "jxl"])
@pytest.mark.parametrize("flag,value,opt_name", [("--exposure", "1", "--exposure"), ("--bits", "8", "--bits")])
def test_cli_rejects_unsupported_options_before_reading(monkeypatch, capsys, tmp_path, ext, flag, value, opt_name):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("read() should not be called before validating output flags")

    monkeypatch.setattr(cli, "read", fail_if_called)

    rc = cli.main([str(tmp_path / "big.hdr"), str(tmp_path / f"out.{ext}"), flag, value])
    assert rc == 1
    captured = capsys.readouterr()
    assert f"{opt_name} is not supported for {ext.upper()} output" in captured.err


def test_cli_missing_source_reports_unsupported_option_not_missing_file(capsys, tmp_path):
    missing_src = tmp_path / "does_not_exist.hdr"
    rc = cli.main([str(missing_src), str(tmp_path / "out.exr"), "--exposure", "1"])
    assert rc == 1
    captured = capsys.readouterr()
    assert "--exposure is not supported for EXR output" in captured.err
