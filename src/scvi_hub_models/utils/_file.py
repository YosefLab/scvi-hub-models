def make_parents(*paths) -> None:
    """Make parent directories of a file path if they do not exist."""
    from pathlib import Path

    for p in paths:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
