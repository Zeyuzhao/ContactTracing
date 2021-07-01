from pathlib import Path

# Assume folder containing project is root
PROJECT_ROOT = Path(__file__).parent.parent


def set_project_root(path: Path):
    """Set root of project"""
    global PROJECT_ROOT
    PROJECT_ROOT = path
