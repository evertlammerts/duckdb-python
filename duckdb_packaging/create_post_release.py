#!/usr/bin/env python3
"""
DuckDB Python post-release creation CLI tool.

This tool helps create post-releases for extension-specific features and fixes
that don't align with the main DuckDB release cycle.

Usage:
    python -m duckdb_packaging.create_post_release [options]
"""

import argparse
import subprocess
import sys
from typing import Optional
from ._versioning import (
    parse_version,
    pep440_to_git_tag,
    format_version,
    create_git_tag
)


def get_base_version_for_post_release() -> Optional[str]:
    """Get the base version for creating a post-release.
    
    This looks for the latest non-post-release tag to use as the base.
    
    Returns:
        Base version string or None if no suitable base found
    """
    try:
        # Get all tags sorted by version
        result = subprocess.run(
            ["git", "tag", "--sort=-version:refname"],
            capture_output=True,
            text=True,
            check=True
        )
        
        for tag in result.stdout.strip().split('\n'):
            if not tag:
                continue
                
            # Remove 'v' prefix
            version = tag[1:] if tag.startswith('v') else tag
            
            # Convert git tag format to PEP440 (if needed)
            if "-post" in version:
                version = version.replace("-post", ".post")
            
            try:
                major, minor, patch, post = parse_version(version)
                # Return the first non-post-release version
                if post is None:
                    return format_version(major, minor, patch)
            except ValueError:
                continue
                
        return None
    except subprocess.CalledProcessError:
        return None


def get_next_post_number(base_version: str) -> int:
    """Get the next post-release number for a given base version.
    
    Args:
        base_version: Base version (e.g., "1.3.1")
        
    Returns:
        Next post-release number (starting from 1)
    """
    try:
        # Get all tags and find existing post-releases for this base
        result = subprocess.run(
            ["git", "tag"],
            capture_output=True,
            text=True,
            check=True
        )
        
        max_post = 0
        for tag in result.stdout.strip().split('\n'):
            if not tag:
                continue
                
            # Remove 'v' prefix
            version = tag[1:] if tag.startswith('v') else tag
            
            # Convert git tag format to PEP440
            if "-post" in version:
                version = version.replace("-post", ".post")
            
            try:
                major, minor, patch, post = parse_version(version)
                current_base = format_version(major, minor, patch)
                
                if current_base == base_version and post is not None:
                    max_post = max(max_post, post)
            except ValueError:
                continue
                
        return max_post + 1
    except subprocess.CalledProcessError:
        return 1


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create post-releases for DuckDB Python package"
    )
    parser.add_argument(
        "--base-version",
        help="Base version for post-release (default: auto-detect latest release)"
    )
    parser.add_argument(
        "--post-number",
        type=int,
        help="Post-release number (default: auto-increment)"
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
        "--reason",
        help="Reason for post-release (included in tag message)"
    )
    parser.add_argument(
        "--no-tag",
        action="store_true",
        help="Calculate version without creating a git tag"
    )
    
    args = parser.parse_args()
    
    try:
        # Determine base version
        if args.base_version:
            base_version = args.base_version
            # Validate format
            major, minor, patch, post = parse_version(base_version + ".post0")
            if post != 0:
                print(f"Error: Base version should not include post-release suffix", file=sys.stderr)
                return 1
            base_version = format_version(major, minor, patch)
        else:
            base_version = get_base_version_for_post_release()
            if not base_version:
                print("No suitable base version found. Use --base-version to specify one.", file=sys.stderr)
                return 1
        
        print(f"Base version: {base_version}")
        
        # Determine post-release number
        if args.post_number:
            post_number = args.post_number
        else:
            post_number = get_next_post_number(base_version)
        
        # Create new post-release version
        new_version = format_version(*parse_version(base_version)[:3], post_number)
        print(f"New post-release version: {new_version}")
        
        if args.dry_run:
            print("DRY RUN: No changes made")
            if not args.no_tag:
                tag_msg = args.tag_message or f"Post-release {new_version}"
                if args.reason:
                    tag_msg += f": {args.reason}"
                tag_name = pep440_to_git_tag(new_version)
                print(f"Would create tag: {tag_name} with message: {tag_msg}")
            return 0
        
        # Create git tag if requested
        if not args.no_tag:
            tag_message = args.tag_message or f"Post-release {new_version}"
            if args.reason:
                tag_message += f": {args.reason}"
            
            create_git_tag(new_version, tag_message)
            tag_name = pep440_to_git_tag(new_version)
            print(f"Created git tag: {tag_name}")
        
        print(f"Post-release {new_version} created successfully")
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())