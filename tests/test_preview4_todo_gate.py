from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_preview4_todo.py"


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_preview4_todo", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


todo_gate = _load_script()


def test_todo_gate_reports_every_unchecked_item_with_location(tmp_path: Path) -> None:
    (tmp_path / "01-contract.md").write_text(
        "# Contract\n\n- [x] Frozen\n- [ ] Implement the runtime.\n",
        encoding="utf-8",
    )
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "02-validation.md").write_text("- [ ] Run the matrix.\n", encoding="utf-8")

    items = todo_gate.find_unchecked_items(tmp_path)

    assert [(str(item.path), item.line, item.text) for item in items] == [
        ("01-contract.md", 4, "Implement the runtime."),
        (str(Path("nested") / "02-validation.md"), 1, "Run the matrix."),
    ]


def test_todo_gate_accepts_complete_documents_and_ignores_prose(tmp_path: Path) -> None:
    (tmp_path / "done.md").write_text(
        "# Done\n\n- [x] Implemented\nThe text [ ] is not a checklist item.\n",
        encoding="utf-8",
    )

    assert todo_gate.find_unchecked_items(tmp_path) == ()


def test_todo_gate_cli_rejects_missing_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["check_preview4_todo.py", "--root", "missing"])

    with pytest.raises(SystemExit, match="does not exist"):
        todo_gate.main()
