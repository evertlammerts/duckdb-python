import os
import re

def _main_branch_versioning() -> bool:
    """Whether we should main branch versioning according to the env var MAIN_BRANCH_VERSIONING

    :return: False if MAIN_BRANCH_VERSIONING is set to "0", True otherwise
    """
    return os.getenv('MAIN_BRANCH_VERSIONING') != "0"

VERSION_RE = re.compile(r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)\.(?P<patch>[0-9]+)$")


def bump_version(base_version: str, distance: int, dirty: bool = False):
    """Bump the version if needed."""
    # Validate the base version (this should never include anything else than X.Y.Z)
    base_version_match = VERSION_RE.match(base_version)
    if not base_version_match:
        raise ValueError(f"Incorrect version format: {base_version} (expected X.Y.Z)")

    major, minor, patch = map(int, base_version_match.groups())

    # If we're exactly on a tag (distance = 0, dirty=False)
    if distance == 0 and not dirty:
        return f"{major}.{minor}.{patch}"

    # Otherwise we're at a distance and / or dirty, and need to bump
    if MAIN_BRANCH_VERSIONING:
        return f"{major}.{minor+1}.0.dev{distance}"
    return f"{major}.{minor}.{patch+1}.dev{distance}"


# Here we handle getting versions from env vars. We only support a single way of
# manually overriding the version, which is through OVERRIDE_GIT_DESCRIBE. If
# SETUPTOOLS_SCM_PRETEND_VERSION* is set we unset it.
SCM_PRETEND_ENV_VAR = "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_DUCKDB"
OVERRIDE_GIT_DESCRIBE_ENV_VAR = "OVERRIDE_GIT_DESCRIBE"
OVERRIDE = os.getenv(OVERRIDE_GIT_DESCRIBE_ENV_VAR)

if OVERRIDE:
    # OVERRIDE_GIT_DESCRIBE_ENV_VAR is set, we'll put it in SCM_PRETEND_ENV_VAR_FOR_DUCKDB
    print(f"[setup.py] Found {OVERRIDE_GIT_DESCRIBE_ENV_VAR}={OVERRIDE}")
    DESCRIBE_RE = re.compile(
        r"""
        ^v(?P<tag>\d+\.\d+\.\d+)        # vX.Y.Z
        (?:-(?P<distance>\d+))?         # optional -N
        (?:-g(?P<hash>[0-9a-fA-F]+))?   # optional -g<sha>
        $""",
        re.VERBOSE,
    )
    match = DESCRIBE_RE.match(OVERRIDE)
    if not match:
        raise ValueError(f"Invalid {OVERRIDE_GIT_DESCRIBE_ENV_VAR}: {OVERRIDE}")

    tag = match["tag"]
    distance = match["distance"]
    commit = match["hash"] and match["hash"].lower()

    # If we get an override we do need to bump
    pep440 = bump_version(tag, int(distance or 0))
    if commit:
        pep440 += f"+g{commit}"

    os.environ[SCM_PRETEND_ENV_VAR] = pep440
    print(f"[setup.py] Injected {SCM_PRETEND_ENV_VAR}={pep440}")
elif SCM_PRETEND_ENV_VAR in os.environ:
    # SCM_PRETEND_ENV_VAR is already set, but we don't allow that
    print(f"[setup.py] WARNING: We do not support {SCM_PRETEND_ENV_VAR}! Removing.")
    del os.environ[SCM_PRETEND_ENV_VAR]


if "SETUPTOOLS_SCM_PRETEND_VERSION" in os.environ:
    print(f"[setup.py] WARNING: We do not support SETUPTOOLS_SCM_PRETEND_VERSION! Removing.")
    del os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"]
########################################################################
# END VERSIONING LOGIC
########################################################################