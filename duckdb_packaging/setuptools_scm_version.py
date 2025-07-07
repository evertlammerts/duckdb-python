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

SCM_PRETEND_ENV_VAR = "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DUCKDB"
SCM_GLOBAL_PRETEND_ENV_VAR = "SETUPTOOLS_SCM_PRETEND_VERSION"
OVERRIDE_GIT_DESCRIBE_ENV_VAR = "OVERRIDE_GIT_DESCRIBE"
FORCE_STABLE_BUMP_ENV_VAR = "FORCE_STABLE_BUMP"
FORCE_POST_BUMP_ENV_VAR = "FORCE_POST_BUMP"
FORCE_RC_BUMP_ENV_VAR = "FORCE_RC_BUMP"


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
    print(f"[version_scheme] version object: {version}")
    print(f"[version_scheme] version.tag: {version.tag}")
    print(f"[version_scheme] version.distance: {version.distance}")
    print(f"[version_scheme] version.dirty: {version.dirty}")
    
    # Handle case where tag is None
    if version.tag is None:
        raise ValueError("Need a valid version. Did you set a fallback_version in pyproject.toml?")
    
    try:
        return _bump_version(str(version.tag), version.distance, version.dirty)
    except Exception as e:
        raise RuntimeError(f"Failed to bump version: {e}")


def _bump_version(base_version: str, distance: int, dirty: bool = False) -> str:
    """Bump the version if needed."""
    # Validate the base version (this should never include anything else than X.Y.Z or X.Y.Z.[rc|post]N)
    try:
        major, minor, patch, post, rc = parse_version(base_version)
    except ValueError as e:
        raise ValueError(f"Incorrect version format: {base_version} (expected X.Y.Z or X.Y.Z.postN)")

    # If we're exactly on a tag (distance = 0, dirty=False)
    distance = int(distance or 0)
    if distance == 0 and not dirty:
        return format_version(major, minor, patch, post)

    # Otherwise we're at a distance and / or dirty, and need to bump
    if post != 0:
        # We're developing on top of a post-release
        return f"{format_version(major, minor, patch, post=post+1)}.dev{distance}"
    elif rc != 0:
        # We're developing on top of an rc
        return f"{format_version(major, minor, patch, rc=rc+1)}.dev{distance}"
    elif _main_branch_versioning():
        return f"{format_version(major, minor+1, 0)}.dev{distance}"
    return f"{format_version(major, minor, patch+1)}.dev{distance}"


def _handle_version_overrides():
    """
    Handle getting versions from environment variables.

    Only supports a single way of manually overriding the version through
    OVERRIDE_GIT_DESCRIBE. If SETUPTOOLS_SCM_PRETEND_VERSION* is set, it gets unset.
    """
    override_value = os.getenv(OVERRIDE_GIT_DESCRIBE_ENV_VAR)

    if override_value:
        _process_git_describe_override(override_value, SCM_PRETEND_ENV_VAR, OVERRIDE_GIT_DESCRIBE_ENV_VAR)
    elif SCM_PRETEND_ENV_VAR in os.environ:
        _remove_unsupported_env_var(SCM_PRETEND_ENV_VAR)

    # Always check and remove unsupported SETUPTOOLS_SCM_PRETEND_VERSION
    if SCM_GLOBAL_PRETEND_ENV_VAR in os.environ:
        _remove_unsupported_env_var(SCM_GLOBAL_PRETEND_ENV_VAR)


def _process_git_describe_override(override_value, scm_pretend_env_var, override_env_var):
    """Process the OVERRIDE_GIT_DESCRIBE environment variable."""
    print(f"[versioning] Found {override_env_var}={override_value}")

    describe_pattern = re.compile(
        r"""
        ^v(?P<tag>\d+\.\d+\.\d+(?:-post\d+)?)   # vX.Y.Z or vX.Y.Z-postN
        (?:-(?P<distance>\d+))?                  # optional -N
        (?:-g(?P<hash>[0-9a-fA-F]+))?            # optional -g<sha>
        $""",
        re.VERBOSE,
    )

    match = describe_pattern.match(override_value)
    if not match:
        raise ValueError(f"Invalid {override_env_var}: {override_value}")

    tag = match["tag"]
    distance = match["distance"]
    commit_hash = match["hash"]

    # Convert git tag format to PEP440 format (v1.3.1-post1 -> 1.3.1.post1)
    if "-post" in tag:
        tag = tag.replace("-post", ".post")

    # Bump version and format according to PEP440
    pep440_version = _bump_version(tag, int(distance or 0))
    if commit_hash:
        pep440_version += f"+g{commit_hash.lower()}"

    os.environ[scm_pretend_env_var] = pep440_version
    print(f"[versioning] Injected {scm_pretend_env_var}={pep440_version}")


def _remove_unsupported_env_var(env_var):
    """Remove an unsupported environment variable with a warning."""
    print(f"[versioning] WARNING: We do not support {env_var}! Removing.")
    del os.environ[env_var]

# Handle environment overrides on module load
_handle_version_overrides()