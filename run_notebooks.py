"""
For each Exp*/ExpN.ipynb:
  1. Strip markdown cells and inline comments, execute, then:
     - Save text output to output.txt in the same folder
     - Save image outputs as img_<cell>_<n>.png in the same folder
  2. Export a clean .py file (no markdown, no comments) alongside the notebook
"""

import re
import sys
import base64
import copy
from pathlib import Path
import nbformat
from nbclient import NotebookClient


def strip_comments(source: str) -> str:
    lines = []
    for line in source.splitlines():
        if line.lstrip().startswith("#"):
            continue
        cleaned = re.sub(r'\s+#[^"\']*$', '', line)
        lines.append(cleaned)
    result = "\n".join(lines)
    # collapse runs of 3+ blank lines down to 2
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result.strip()


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
            if not cell.source:
                continue
        cleaned_cells.append(cell)
    nb.cells = cleaned_cells
    return nb


def save_outputs(nb: nbformat.NotebookNode, folder: Path) -> None:
    txt_lines = []
    img_count = 0

    for cell_idx, cell in enumerate(nb.cells):
        if cell.cell_type != "code" or not cell.outputs:
            continue

        cell_has_text = False
        for out in cell.outputs:
            # --- text output ---
            text = None
            if out.output_type in ("stream", "display_data", "execute_result"):
                if out.output_type == "stream":
                    text = "".join(out.get("text", []))
                else:
                    data = out.get("data", {})
                    if "text/plain" in data:
                        text = "".join(data["text/plain"])

            if text:
                if not cell_has_text:
                    txt_lines.append(f"=== Cell {cell_idx + 1} ===")
                    cell_has_text = True
                txt_lines.append(text.rstrip())

            # --- image output ---
            if out.output_type in ("display_data", "execute_result"):
                data = out.get("data", {})
                for mime in ("image/png", "image/jpeg", "image/svg+xml"):
                    if mime in data:
                        ext = mime.split("/")[1].replace("svg+xml", "svg")
                        img_path = folder / f"img_cell{cell_idx + 1}_{img_count}.{ext}"
                        raw = data[mime]
                        if isinstance(raw, str):
                            img_path.write_bytes(base64.b64decode(raw))
                        else:
                            img_path.write_bytes(raw)
                        img_count += 1
                        if not cell_has_text:
                            txt_lines.append(f"=== Cell {cell_idx + 1} ===")
                            cell_has_text = True
                        txt_lines.append(f"[image saved: {img_path.name}]")

    out_txt = folder / "output.txt"
    out_txt.write_text("\n".join(txt_lines), encoding="utf-8")
    print(f"  Output text : {out_txt}")
    if img_count:
        print(f"  Images saved: {img_count}")


def export_py(nb: nbformat.NotebookNode, nb_path: Path) -> None:
    py_path = nb_path.with_suffix(".py")
    blocks = []
    for cell in nb.cells:
        if cell.cell_type == "code" and cell.source.strip():
            blocks.append(cell.source.strip())
    py_path.write_text("\n\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"  Python file : {py_path}")


def run_notebook(nb_path: Path) -> None:
    print(f"\n{'='*60}")
    print(f"Processing: {nb_path}")

    with open(nb_path) as f:
        nb = nbformat.read(f, as_version=4)

    cleaned = clean_notebook(nb)

    # export clean .py before execution (no outputs needed)
    export_py(cleaned, nb_path)

    client = NotebookClient(
        cleaned,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(nb_path.parent)}},
    )

    try:
        client.execute()
        print("  Executed successfully.")
    except Exception as e:
        print(f"  Execution error (saving partial outputs): {e}")

    save_outputs(cleaned, nb_path.parent)


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
