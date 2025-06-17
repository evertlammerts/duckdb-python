# DuckDB Python package

## Where to get it

You can install the latest release of DuckDB directly from [PyPi](https://pypi.org/project/duckdb/):

```bash
pip install duckdb
```

## Documentation

[DuckDB.org](https://duckdb.org) is a great resource for documentation:
* We recommend the [Python user guide](https://duckdb.org/docs/stable/guides/python/install) to get started.
* And make sure to check out [the latest API documentation](https://duckdb.org/docs/stable/clients/python/overview).

## Getting Help

See the [DuckDB Community Support Policy](https://duckdblabs.com/community_support_policy/). DuckDB Labs also provides [custom support](https://duckdblabs.com/#support).

## Source Code

The Python package is part of the duckdb source tree. If you want to make changes and / or build the package from source, see the [development documentation](https://duckdb.org/docs/stable/dev/building/python) for build instructions, IDE integration, debugging, and other relevant information.

## Build Configuration Reference

The module supports the following build configuration properties under `tool.duckdb` in pyproject.toml:
* `tool.duckdb.extensions` **(list, default=[])**: The extensions to build into the wheel and / or sdist
* `tool.duckdb.sdist` **(dictionary)**: sdist-specific config:
  * `tool.duckdb.sdist.duckdb_src_target` **(str, required for building sdists)**: The path to a directory to store
  duckdb source files that will be included in the sdist
  * `tool.duckdb.sdist.include_line_numbers` **(bool, default false)**: Include line numbers in the unity build. You 
    may want to set this to true if you want to build e.g. a debug version for lldb with a source map for duckdb.
  * `tool.duckdb.sdist.unity_count` **(int, default 32)**: The amount of unity source files to create. You might 
    want to change the default amount created if you want e.g. an amalgamated file, or more control over the
    parallelism of the build.
  * `tool.duckdb.sdist.short_paths` **(bool, default false)**: Use short paths in the unity build. You may want to 
    use this if you're on a platform that has command length limitations (like Windows) _and_ scikit-build-core 
    doesn't or can't use Ninja for some reason.

## Versioning and Releases

### Versioning Scheme

This package follows DuckDB's versioning scheme with extensions for packaging-specific releases:

- **Standard releases**: `1.3.1`, `1.3.2` (follows DuckDB core versions)
- **Post-releases**: `1.3.1.post1`, `1.3.1.post2` (extension-specific features/fixes)
- **Development versions**: 
  - `1.4.0.dev42` (main branch, targeting next minor version)
  - `1.3.2.dev5` (release branch, targeting next patch version)
  - `1.3.1.post1.dev3` (post-release development)

### Branch Strategy

- **Main branch (`main`)**: Development for next minor version (e.g., 1.3 → 1.4)
- **Release branches** (e.g., `v1.3-ossivalis`): Patches for current minor version (e.g., 1.3.1 → 1.3.2)
- **Post-releases**: Used for extension-specific features that don't align with DuckDB core releases

### Release Management

#### Version Bumping
```bash
# Bump to next major version (1.3.1 → 2.0.0)
python -m duckdb_packaging.bump_version major

# Bump to next minor version (1.3.1 → 1.4.0)  
python -m duckdb_packaging.bump_version minor

# Bump to next patch version (1.3.1 → 1.3.2)
python -m duckdb_packaging.bump_version patch

# Create post-release (1.3.1 → 1.3.1.post1)
python -m duckdb_packaging.bump_version post
```

#### Post-Release Creation
```bash
# Create post-release with auto-detected base version
python -m duckdb_packaging.create_post_release --reason "Extension-specific bug fix"

# Create post-release with specific base version
python -m duckdb_packaging.create_post_release --base-version 1.3.1 --reason "New extension feature"

# Dry run to preview changes
python -m duckdb_packaging.create_post_release --dry-run
```

#### Manual Version Overrides
For CI/CD scenarios, versions can be overridden using the `OVERRIDE_GIT_DESCRIBE` environment variable:
```bash
export OVERRIDE_GIT_DESCRIBE="v1.3.1-5-g1234567"
# Build will use version 1.4.0.dev5+g1234567
```

### Dynamic Versioning Integration

The package uses `setuptools_scm` for automatic version determination:

- **pyproject.toml configuration**:
  ```toml
  [tool.setuptools_scm]
  fallback_version = "1.3.0"
  version_scheme = "duckdb_packaging._setuptools_scm_version:version_scheme"
  ```

- **Environment variables**:
  - `MAIN_BRANCH_VERSIONING=0`: Use release branch versioning (patch increments)
  - `MAIN_BRANCH_VERSIONING=1`: Use main branch versioning (minor increments)
  - `OVERRIDE_GIT_DESCRIBE`: Override version detection

### Git Tag Format

- **Standard releases**: `v1.3.1`, `v1.3.2`
- **Post-releases**: `v1.3.1-post1`, `v1.3.1-post2`

Tags are automatically created by the CLI tools and converted to PEP440 format for Python packaging.

## Conventions

* Follow the [Google Python styleguide](https://google.github.io/styleguide/pyguide.html)
  * See the section on [Comments and Docstrings](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings)