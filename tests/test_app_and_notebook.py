"""The web demo and the Kaggle / Colab notebook."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image
from test_generate import write_checkpoint

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
NOTEBOOK = ROOT / "notebooks" / "train_colab_kaggle.ipynb"


# --- notebook ---------------------------------------------------------------------


def build_notebook(out: Path) -> "object":
    nbformat = pytest.importorskip("nbformat")
    subprocess.run(
        [sys.executable, str(ROOT / "notebooks" / "build_notebook.py"), str(ROOT), str(out)],
        check=True,
        capture_output=True,
    )
    return nbformat.read(out, as_version=4)


def test_notebook_embeds_the_current_package_code(tmp_path: Path) -> None:
    notebook = build_notebook(tmp_path / "nb.ipynb")
    written = {
        cell.source.split("\n", 1)[0].split()[-1]: cell.source.split("\n", 1)[1]
        for cell in notebook.cells
        if cell.cell_type == "code" and cell.source.startswith("%%writefile")
    }
    package = ROOT / "src" / "bia_captioner"
    expected = {f"src/bia_captioner/{p.name}": p.read_text() for p in package.glob("*.py")}
    assert set(written) == set(expected), "a module is missing from the notebook"
    for name, source in expected.items():
        # (cells are stored stripped, so compare without the final newline)
        assert written[name] == source.strip(), f"{name} in the notebook differs from src/"


def test_committed_notebook_is_up_to_date(tmp_path: Path) -> None:
    # Fails when src/ changed but `python notebooks/build_notebook.py . notebooks/...`
    # was not re-run.
    fresh = build_notebook(tmp_path / "nb.ipynb")
    nbformat = pytest.importorskip("nbformat")
    committed = nbformat.read(NOTEBOOK, as_version=4)
    assert [c.source for c in committed.cells] == [c.source for c in fresh.cells]


def test_notebook_settings_cell_is_valid_python(tmp_path: Path) -> None:
    import ast

    notebook = build_notebook(tmp_path / "nb.ipynb")
    for cell in notebook.cells:
        if cell.cell_type == "code" and not cell.source.startswith("%%"):
            ast.parse(cell.source)  # raises on a syntax error


# --- web app ----------------------------------------------------------------------


def load_app() -> "object":
    pytest.importorskip("streamlit")
    spec = importlib.util.spec_from_file_location("streamlit_app", APP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("mode", ["Beam search", "Greedy", "Sampling"])
def test_app_captions_an_image(tmp_path: Path, mode: str) -> None:
    app = load_app()
    path, _ = write_checkpoint(tmp_path / "run")
    settings = {
        "checkpoint": str(path), "device": "cpu", "mode": mode, "beam_size": 3,
        "length_penalty": 0.6, "temperature": 0.8, "top_k": 5, "top_p": 0.9,
    }  # fmt: skip
    results = app.captions_for(Image.new("RGB", (90, 60), "white"), settings)
    assert results and isinstance(results[0][0], str)
    assert (len(results) > 1) == (mode == "Beam search")  # beams are listed with scores


def test_app_page_renders_and_reports_a_missing_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setattr(sys, "argv", ["app", "--checkpoint", "does/not/exist.pt"])
    page = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not page.exception
    assert page.title[0].value == "BIA Image Captioner"
    assert "Checkpoint not found" in page.error[0].value
