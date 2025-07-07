"""DuckDB Python versioning utilities. This will only work on Python >= 3.3 and on non-mobile platforms.

This module provides utilities for version management including:
- Version bumping (major, minor, patch, post)
- Git tag creation and management
- Version parsing and validation
"""

import subprocess
from typing import Optional, Tuple
import re


VERSION_RE = re.compile(r"^(?P<major>[0-9]+)\.(?P<minor>[0-9]+)\.(?P<patch>[0-9]+)(?:\.post(?P<post>[0-9]+)|\.rc(?P<rc>[0-9]+))?$")
GIT_TAG_RE = re.compile(r"^v(?P<version>\d+\.\d+\.\d+(?:-post\d+)?)(?:-(?P<distance>\d+)-g(?P<hash>[0-9a-fA-F]+))?(?:-dirty)?$")


def parse_version(version: str) -> tuple[int, int, int, int, int]:
    """Parse a version string into its components.
    
    Args:
        version: Version string (e.g., "1.3.1", "1.3.2.rc3" or "1.3.1.post2")
        
    Returns:
        Tuple of (major, minor, patch, post, rc)
        
    Raises:
        ValueError: If version format is invalid
    """
    match = VERSION_RE.match(version)
    if not match:
        raise ValueError(f"Invalid version format: {version} (expected X.Y.Z, X.Y.Z.rcM or X.Y.Z.postN)")
    
    major, minor, patch, post, rc = match.groups()
    return int(major), int(minor), int(patch), int(post or 0), int(rc or 0)


def format_version(major: int, minor: int, patch: int, post: int = 0, rc: int = 0) -> str:
    """Format version components into a version string.
    
    Args:
        major: Major version number
        minor: Minor version number  
        patch: Patch version number
        post: Post-release number
        rc: RC number
        
    Returns:
        Formatted version string
    """
    version = f"{major}.{minor}.{patch}"
    if post != 0 and rc != 0:
        raise ValueError("post and rc are mutually exclusive")
    if post != 0:
        version += f".post{post}"
    if rc != 0:
        version += f".rc{rc}"
    return version


def git_tag_to_pep440(git_tag: str) -> str:
    """Convert git tag format to PEP440 format.
    
    Args:
        git_tag: Git tag (e.g., "v1.3.1", "v1.3.1-post1")
        
    Returns:
        PEP440 version string (e.g., "1.3.1", "1.3.1.post1")
    """
    # Remove 'v' prefix if present
    version = git_tag[1:] if git_tag.startswith('v') else git_tag
    
    # Convert git tag format to PEP440 format (1.3.1-post1 -> 1.3.1.post1)
    if "-post" in version:
        version = version.replace("-post", ".post")
        
    return version


def pep440_to_git_tag(version: str) -> str:
    """Convert PEP440 version to git tag format.
    
    Args:
        version: PEP440 version string (e.g., "1.3.1.post1")
        
    Returns:
        Git tag format (e.g., "v1.3.1-post1")
    """
    # Convert PEP440 format to git tag format (1.3.1.post1 -> v1.3.1-post1)
    tag_name = version.replace(".post", "-post")
    return f"v{tag_name}"


def get_current_version() -> Optional[str]:
    """Get the current version from git tags.
    
    Returns:
        Current version string or None if no tags exist
    """
    try:
        # Get the latest tag
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            capture_output=True,
            text=True,
            check=True
        )
        tag = result.stdout.strip()
        return git_tag_to_pep440(tag)
    except subprocess.CalledProcessError:
        return None


def create_git_tag(version: str, message: Optional[str] = None) -> None:
    """Create a git tag for the given version.
    
    Args:
        version: Version string (PEP440 format)
        message: Optional tag message
        
    Raises:
        subprocess.CalledProcessError: If git command fails
    """
    tag_name = pep440_to_git_tag(version)
    
    cmd = ["git", "tag"]
    if message:
        cmd.extend(["-a", tag_name, "-m", message])
    else:
        cmd.append(tag_name)
    
    subprocess.run(cmd, check=True)


def get_git_describe() -> Optional[str]:
    """Get git describe output for version determination.
    
    Returns:
        Git describe output or None if no tags exist
    """
    try:
        result = subprocess.run(
            ["git", "describe", "--tags", "--dirty"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None