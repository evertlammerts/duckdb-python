"""
setuptools_scm integration for DuckDB Python versioning.

This module provides the setuptools_scm version scheme and handles environment variable overrides
to match the exact behavior of the original DuckDB Python package.
"""

import os
import re
from typing import Any

# Import from our own versioning module to avoid duplication
from ._versioning import parse_version, format_version

# MAIN_BRANCH_VERSIONING default should be 'True' for main branch and feature branches
MAIN_BRANCH_VERSIONING = True
def _main_branch_versioning():
    from_env = os.getenv('MAIN_BRANCH_VERSIONING')
    return from_env == "1" if from_env is not None else MAIN_BRANCH_VERSIONING

def version_scheme(version: Any) -> str:
    """
    setuptools_scm version scheme that matches DuckDB's original behavior.
    
    Args:
        version: setuptools_scm version object
        
    Returns:
        PEP440 compliant version string
    """
    return bump_version(str(version.tag), version.distance, version.dirty)


def bump_version(base_version: str, distance: int, dirty: bool = False) -> str:
    """Bump the version if needed."""
    # Validate the base version (this should never include anything else than X.Y.Z or X.Y.Z.postN)
    try:
        major, minor, patch, post = parse_version(base_version)
    except ValueError as e:
        raise ValueError(f"Incorrect version format: {base_version} (expected X.Y.Z or X.Y.Z.postN)") from e

    # If we're exactly on a tag (distance = 0, dirty=False)
    distance = int(distance or 0)
    if distance == 0 and not dirty:
        return format_version(major, minor, patch, post)

    # Otherwise we're at a distance and / or dirty, and need to bump
    if post is not None:
        # We're developing on top of a post-release
        return f"{format_version(major, minor, patch, post)}.dev{distance}"
    elif _main_branch_versioning():
        return f"{format_version(major, minor+1, 0)}.dev{distance}"
    return f"{format_version(major, minor, patch+1)}.dev{distance}"


# Here we handle getting versions from env vars. We only support a single way of
# manually overriding the version, which is through OVERRIDE_GIT_DESCRIBE. If
# SETUPTOOLS_SCM_PRETEND_VERSION* is set we unset it.
SCM_PRETEND_ENV_VAR = "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DUCKDB"
OVERRIDE_GIT_DESCRIBE_ENV_VAR = "OVERRIDE_GIT_DESCRIBE"
OVERRIDE = os.getenv(OVERRIDE_GIT_DESCRIBE_ENV_VAR)

if OVERRIDE:
    # OVERRIDE_GIT_DESCRIBE_ENV_VAR is set, we'll put it in SCM_PRETEND_ENV_VAR_FOR_DUCKDB
    print(f"[versioning] Found {OVERRIDE_GIT_DESCRIBE_ENV_VAR}={OVERRIDE}")
    DESCRIBE_RE = re.compile(
        r"""
        ^v(?P<tag>\d+\.\d+\.\d+(?:-post\d+)?)   # vX.Y.Z or vX.Y.Z-postN
        (?:-(?P<distance>\d+))?                  # optional -N
        (?:-g(?P<hash>[0-9a-fA-F]+))?            # optional -g<sha>
        $""",
        re.VERBOSE,
    )
    match = DESCRIBE_RE.match(OVERRIDE)
    if not match:
        raise ValueError(f"Invalid {OVERRIDE_GIT_DESCRIBE_ENV_VAR}: {OVERRIDE}")

    tag = match["tag"]
    distance = match["distance"]
    commit = match["hash"] and match["hash"].lower()

    # Convert git tag format to PEP440 format (v1.3.1-post1 -> 1.3.1.post1)
    if "-post" in tag:
        tag = tag.replace("-post", ".post")

    # If we get an override we do need to bump
    pep440 = bump_version(tag, int(distance or 0))
    if commit:
        pep440 += f"+g{commit}"

    os.environ[SCM_PRETEND_ENV_VAR] = pep440
    print(f"[versioning] Injected {SCM_PRETEND_ENV_VAR}={pep440}")
elif SCM_PRETEND_ENV_VAR in os.environ:
    # SCM_PRETEND_ENV_VAR is already set, but we don't allow that
    print(f"[versioning] WARNING: We do not support {SCM_PRETEND_ENV_VAR}! Removing.")
    del os.environ[SCM_PRETEND_ENV_VAR]


if "SETUPTOOLS_SCM_PRETEND_VERSION" in os.environ:
    print(f"[versioning] WARNING: We do not support SETUPTOOLS_SCM_PRETEND_VERSION! Removing.")
    del os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"]
########################################################################
# END VERSIONING LOGIC
########################################################################