"""Regenerate the three revised paper figures from the repository data."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("make_revision_figures.py")), run_name="__main__")
