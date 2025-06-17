#!/usr/bin/env python3
"""
DuckDB Python version bumping CLI tool.

Usage:
    python -m duckdb_packaging.bump_version <bump_type> [options]

Where bump_type is one of: major, minor, patch, post
"""

import argparse
import sys
from ._versioning import (
    pep440_to_git_tag,
    get_current_version,
    bump_version_string,
    create_git_tag,
)


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Bump version and create git tags for DuckDB Python package"
    )
    parser.add_argument(
        "bump_type",
        choices=["major", "minor", "patch", "post"],
        help="Type of version bump to perform"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    parser.add_argument(
        "--tag-message",
        help="Message for the git tag (default: auto-generated)"
    )
    parser.add_argument(
        "--no-tag",
        action="store_true",
        help="Bump version without creating a git tag"
    )
    parser.add_argument(
        "--current-version",
        help="Override current version detection (for testing)"
    )
    
    args = parser.parse_args()
    
    try:
        # Get current version
        if args.current_version:
            current_version = args.current_version
        else:
            current_version = get_current_version()
            
        if not current_version:
            print("No current version found. Use --current-version to specify one.", file=sys.stderr)
            return 1
            
        print(f"Current version: {current_version}")
        
        # Calculate new version
        new_version = bump_version_string(current_version, args.bump_type)
        print(f"New version: {new_version}")
        
        if args.dry_run:
            print("DRY RUN: No changes made")
            if not args.no_tag:
                tag_msg = args.tag_message or f"Release {new_version}"
                tag_name = pep440_to_git_tag(new_version)
                print(f"Would create tag: {tag_name} with message: {tag_msg}")
            return 0
            
        # Create git tag if requested
        if not args.no_tag:
            tag_message = args.tag_message or f"Release {new_version}"
            create_git_tag(new_version, tag_message)
            tag_name = pep440_to_git_tag(new_version)
            print(f"Created git tag: {tag_name}")
        
        print(f"Version bumped from {current_version} to {new_version}")
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())