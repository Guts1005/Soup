#!/usr/bin/env python3
"""Check whether a release tag is the highest vX.Y.Z tag of the repository.

Used by .github/workflows/docker.yml to decide whether to push the `latest`
Docker image tag (#2195). Releasing a security patch on an older line (such as
v0.76.2 while v0.77.0 is out) must not move `latest` back to older runtime
dependencies.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

_SEMVER_TAG_RE = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)$")


def parse_semver(tag: str) -> tuple[int, int, int] | None:
    """Parse a 'vX.Y.Z' or 'X.Y.Z' tag into an integer tuple (major, minor, patch)."""
    match = _SEMVER_TAG_RE.match(tag.strip())
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def get_repo_tags() -> list[str]:
    """Return all git tags from the repository via `git tag --sort=-v:refname`."""
    try:
        proc = subprocess.run(
            ["git", "tag", "--sort=-v:refname"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [t.strip() for t in proc.stdout.splitlines() if t.strip()]
    except Exception as e:
        print(f"Warning: could not read git tags ({e})", file=sys.stderr)
        return []


def is_latest_release(candidate_tag: str, all_tags: list[str] | None = None) -> bool:
    """Return True if candidate_tag is greater than or equal to the highest release tag."""
    candidate_ver = parse_semver(candidate_tag)
    if candidate_ver is None:
        raise ValueError(f"Candidate tag '{candidate_tag}' is not a valid vX.Y.Z semver tag")

    if all_tags is None:
        all_tags = get_repo_tags()

    parsed_tags = [parse_semver(t) for t in all_tags]
    valid_vers = [v for v in parsed_tags if v is not None]

    if not valid_vers:
        return True

    highest_ver = max(valid_vers)
    return candidate_ver >= highest_ver


def main() -> int:
    tag = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("REF_NAME", "")
    if not tag:
        print("Usage: check_docker_latest.py <tag>", file=sys.stderr)
        return 1

    try:
        is_latest = is_latest_release(tag)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    result_str = "true" if is_latest else "false"
    print(f"Tag: {tag} | is_latest: {result_str}")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as f:
            f.write(f"is_latest={result_str}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
