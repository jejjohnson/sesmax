"""The command line for the data of the basins, on a stand-in for DVC."""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import pytest
import yaml

from sesmax_ocean._src.basins.commands import REMOTE, BasinName, app, bundle_path
from sesmax_ocean.basins import BASINS


PACKAGE = Path(__file__).resolve().parents[1]
#: Writes its arguments down, and knows a remote once one is added.
DVC = """#!/bin/sh
echo "$@" >> calls.txt
case "$1 $2" in
  "remote add") [ -e remote ] && exit 251; shift $(($# - 1)); echo "$1" > remote;;
  "config remote."*) cat remote 2>/dev/null || exit 251;;
  "root "*) [ ! -e no_repository ] || exit 253;;
  "pull "*) [ ! -e no_network ] || exit 1;;
esac
"""


@pytest.fixture
def here(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty directory, and the stand-in on the path."""
    (tmp_path / "bin").mkdir()
    dvc = tmp_path / "bin" / "dvc"
    dvc.write_text(DVC)
    dvc.chmod(0o755)
    (tmp_path / "work").mkdir()
    monkeypatch.setenv("PATH", f"{dvc.parent}:/usr/bin:/bin")
    monkeypatch.chdir(tmp_path / "work")
    return tmp_path / "work"


def _run(*args: object) -> None:
    app(["data", *(str(a) for a in args)], result_action="return_value")


def _calls(here: Path) -> list[str]:
    """What DVC was asked to do; asking for the remote does not count."""
    calls = (here / "calls.txt").read_text().splitlines()
    return [call for call in calls if call.split()[0] not in ("root", "config")]


def test_the_command_line_knows_the_basins() -> None:
    assert set(get_args(BasinName)) == set(BASINS)


def test_the_stages_are_the_commands() -> None:
    stage = yaml.safe_load((PACKAGE / "dvc.yaml").read_text("utf-8"))["stages"][
        "build-basin"
    ]
    assert set(stage["foreach"]) == set(BASINS)
    written = str(bundle_path("${item}"))
    assert stage["do"]["outs"] == [written]
    assert stage["do"]["cmd"] == f"sesmax-ocean data write ${{item}} --out {written}"
    for source in stage["do"]["deps"]:
        assert (PACKAGE / source).exists()


@pytest.mark.parametrize(
    "url",
    ["gdrive://1a2b", "s3://bucket/key", "azure://container/a", "gs://b/k", "/data/a"],
)
def test_a_remote_is_set_for_the_checkout_alone(url: str, here: Path) -> None:
    _run("init", url)
    assert _calls(here) == [f"remote add --local --default {REMOTE} {url}"]


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("https://drive.google.com/a", "unknown scheme 'https'"),
        ("gcs://bucket", "unknown scheme 'gcs'"),
        ("bucket/key", "neither a URL nor an absolute path"),
    ],
)
def test_a_wrong_url_is_refused(url: str, message: str, here: Path) -> None:
    with pytest.raises(SystemExit, match=message):
        _run("init", url)
    assert not (here / "calls.txt").exists()


def test_a_remote_is_not_replaced_unasked(here: Path) -> None:
    _run("init", "/data/a")
    with pytest.raises(SystemExit, match="exit 251"):
        _run("init", "/data/b")


def test_fetch_and_push_need_a_remote(here: Path) -> None:
    with pytest.raises(SystemExit, match="sesmax-ocean data init"):
        _run("fetch", "med_sea")
    with pytest.raises(SystemExit, match="sesmax-ocean data init"):
        _run("build", "med_sea", "--push")
    assert _calls(here) == []
    _run("init", "/data/a")
    _run("fetch", "med_sea")
    _run("build", "med_sea", "--push", "--force")
    assert _calls(here)[1:] == [
        "pull data/basin/med_sea.zarr",
        "repro --force build-basin@med_sea",
        "push data/basin/med_sea.zarr",
    ]


def test_a_failure_of_dvc_stops_the_command(here: Path) -> None:
    _run("init", "/data/a")
    (here / "no_network").touch()
    with pytest.raises(SystemExit, match=r"dvc pull .* failed"):
        _run("fetch", "med_sea")


def test_a_bundle_is_written_without_dvc(
    here: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run("write", "med_sea", "--nx", 8, "--ny", 8)
    assert (here / "data/basin/med_sea.zarr/zarr.json").is_file()
    assert not (here / "calls.txt").exists()
    _run("status")
    out = capsys.readouterr().out
    assert "remote: none" in out
    assert "[x] med_sea" in out
    assert "[ ] gulf_stream" in out


def test_outside_a_repository(here: Path) -> None:
    """Not the same as a repository without a remote."""
    (here / "no_repository").touch()
    with pytest.raises(SystemExit, match="not in a DVC repository"):
        _run("status")


def test_without_dvc(here: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(here))
    with pytest.raises(SystemExit, match="pip install dvc"):
        _run("fetch", "med_sea")
