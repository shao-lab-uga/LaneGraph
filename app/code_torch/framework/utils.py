from pathlib import Path


def get_git_report() -> Path:
    """
    Utility function to search up file tree for a .git folder

    :return: Pathlib path to git repository root
    :rtype: Path
    """
    current_path = Path.cwd()
    while current_path != current_path.parent:
        if (current_path / ".git").is_dir():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError("no .git directory found.")
