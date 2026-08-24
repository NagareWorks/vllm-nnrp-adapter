from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

UNCHECKED_ITEM = re.compile(r"^\s*- \[ \] (?P<text>.+)$")


@dataclass(frozen=True, slots=True)
class UncheckedItem:
    path: Path
    line: int
    text: str


def find_unchecked_items(root: Path) -> tuple[UncheckedItem, ...]:
    items: list[UncheckedItem] = []
    for path in sorted(root.rglob("*.md")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            match = UNCHECKED_ITEM.match(line)
            if match is not None:
                items.append(
                    UncheckedItem(
                        path=path.relative_to(root),
                        line=line_number,
                        text=match.group("text"),
                    )
                )
    return tuple(items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Require complete Preview4 TODO closure before release.")
    parser.add_argument("--root", type=Path, default=Path("doc/todo/v1-preview4"))
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Preview4 TODO directory does not exist: {root}")
    items = find_unchecked_items(root)
    if items:
        details = "\n".join(f"{item.path}:{item.line}: {item.text}" for item in items)
        raise SystemExit(f"Preview4 TODO is not complete:\n{details}")
    print(f"Preview4 TODO is complete: {root}")


if __name__ == "__main__":
    main()
