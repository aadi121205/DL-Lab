"""
For each Exp*/ExpN.ipynb:
  1. Strip markdown cells and inline comments from code cells
  2. Execute the cleaned notebook in-place (outputs saved back to the .ipynb)
"""

import re
import sys
import json
import copy
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def strip_comments(source: str) -> str:
    """Remove # comments from Python source, preserving blank lines."""
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        # remove inline comment but keep string literals safe (simple heuristic)
        cleaned = re.sub(r'\s+#[^"\']*$', '', line)
        lines.append(cleaned)
    return "\n".join(lines)


def clean_notebook(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    nb = copy.deepcopy(nb)
    cleaned_cells = []
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            continue
        if cell.cell_type == "code":
            cell.source = strip_comments(cell.source)
            cell.outputs = []
            cell.execution_count = None
        cleaned_cells.append(cell)
    nb.cells = cleaned_cells
    return nb


def run_notebook(nb_path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"Processing: {nb_path}")

    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)

    cleaned = clean_notebook(nb)

    client = NotebookClient(
        cleaned,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent)}},
    )

    try:
        client.execute()
        print(f"  Executed successfully.")
    except Exception as e:
        print(f"  Execution error (saving partial outputs): {e}")

    with open(nb_path, "w") as f:
        nbformat.write(cleaned, f)

    print(f"  Saved: {nb_path}")


def main():
    root = Path(__file__).parent
    notebooks = sorted(root.glob("Exp*/Exp*.ipynb"))

    if not notebooks:
        print("No notebooks found.")
        sys.exit(1)

    print(f"Found {len(notebooks)} notebook(s).")
    for nb_path in notebooks:
        run_notebook(nb_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
