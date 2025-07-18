import logging
from pathlib import Path


def find(
    glob: str,
    filter: str = "",
    parent: Path = None,
) -> list:
    """Search for a list of matches to the given glob pattern, optionally filtering results that contain the given filter."""
    if glob.startswith("/"):
        # Search from an absolute path.
        matches = Path("/").glob(glob[1:])
    else:
        # Search from the given or current working directory.
        if parent is None:
            parent = Path()
        matches = parent.glob(glob)
    matching_paths = [match for match in matches if filter in match.as_posix()]
    logging.info(f"Found {len(matching_paths)} matches with filter '{filter}' for pattern: {glob}")
    return matching_paths


def find_one(
    glob: str,
    filter: str = "",
    default: Path = None,
    none_ok: bool = False,
    parent: Path = None,
) -> Path:
    """Search for a single match to the given glob pattern, optionally filtering duplicate glob matches using the given filter."""
    matching_paths = find(glob, filter, parent)
    if len(matching_paths) == 1:
        first_match = matching_paths[0]
        logging.info(f"Found one matching file: {first_match}")
        return first_match
    elif len(matching_paths) > 1:
        logging.error(f"Found multiple matching files, please remove all but one: {matching_paths}")
        raise ValueError(f"Too many matches found.")
    elif default is not None or none_ok:
        return default
    else:
        raise ValueError(f"No match found.")
