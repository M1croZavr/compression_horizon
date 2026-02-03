"""Check that all Python scripts under scripts/ compile successfully."""

import ast
from pathlib import Path


def _scripts_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "scripts"


def _collect_script_paths() -> list[Path]:
    root = _scripts_dir()
    if not root.exists():
        return []
    return sorted(root.rglob("*.py"))


def test_all_scripts_compile():
    """Every .py file under scripts/ (recursively) must parse as valid Python."""
    scripts_dir = _scripts_dir()
    assert scripts_dir.exists(), f"scripts directory not found: {scripts_dir}"
    paths = _collect_script_paths()
    assert paths, "No .py files found under scripts/"

    errors: list[tuple[Path, str]] = []
    for path in paths:
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            ast.parse(source, filename=str(path))
        except SyntaxError as e:
            errors.append((path, str(e)))

    if errors:
        lines = ["Scripts that failed to compile:"]
        for path, msg in errors:
            rel = path.relative_to(scripts_dir)
            lines.append(f"  {rel}: {msg}")
        raise AssertionError("\n".join(lines))
