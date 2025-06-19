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

## Build Configuration Reference

The module includes a custom PEP 517/660 build backend in `duckdb_packaging.build_backend`. This backend prepares sdist and wheel builds before handing off to scikit-build-core.

The following build settings can be used in pyproject.toml:

| Setting                                  | Type | Default | Description                                                                                                                                                                                                 |
|------------------------------------------|------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `tool.duckdb.extensions`                 | list | []      | DuckDB extensions to compile into the package.                                                                                                                                                              |
| `tool.duckdb.sdist.duckdb_src_target`    | str  |         | The path to a directory to store duckdb source files that will be included in the sdist. **Required** for building wheels and sdists.                                                                       |
| `tool.duckdb.sdist.include_line_numbers` | bool | False   | Include line numbers in the unity build. You may want to set this to true if you want to build e.g. a debug version for lldb with a source map for duckdb.                                                  |
| `tool.duckdb.sdist.unity_count`          | int  | 32      | The amount of unity source files to create. You might want to change the default amount created if you want e.g. an amalgamated file, or more control over the parallelism of the build.                    |
| `tool.duckdb.sdist.short_paths`          | bool | False   | Use short paths in the unity build. You may want to use this if you're on a platform that has command length limitations (like Windows) _and_ scikit-build-core doesn't or can't use Ninja for some reason. |

## Versioning and Releases

The DuckDB Python package versioning and release scheme follows that of DuckDB itself. This means that a `X.Y.Z[.
postN]` stable release of the Python package ships the DuckDB stable release `X.Y.Z`. The optional `.postN` releases 
ship the same stable release of DuckDB as their predecessors plus Python package-specific fixes and / or features.

| Types                                                                  | DuckDB Version | Resulting Python Extension Version |
|------------------------------------------------------------------------|----------------|------------------------------------|
| Stable release: DuckDB stable release                                  | `1.3.1`        | `1.3.1`                            |
| Stable post release: DuckDB stable release + Python fixes and features | `1.3.1`        | `1.3.1.postX`                      |
| Nightly micro: DuckDB next micro nightly + Python next micro nightly   | `1.3.2.devM`   | `1.3.2.devN`                       |
| Nightly minor: DuckDB next minor nightly + Python next minor nightly   | `1.4.0.devM`   | `1.4.0.devN`                       |

Note that we do not ship nightly post releases (e.g. we don't ship `1.3.1.post2.dev3`).

### Branch and Tag Strategy

We cut releases as follows:

| Type                 | Tag          | How                                                                             |
|----------------------|--------------|---------------------------------------------------------------------------------|
| Stable minor release | vX.Y.0       | Adding a tag on `main`                                                          |
| Stable micro release | vX.Y.Z       | Adding a tag on a minor release branch (e.g. `v1.3-ossivalis`)                  |
| Stable post release  | vX.Y.Z-postN | Adding a tag on a post release branch (e.g. `v1.3.1-post`)                      |
| Nightly micro        | _not tagged_ | Combining HEAD of the _micro_ release branches of DuckDB and the Python package |
| Nightly minor        | _not tagged_ | Combining HEAD of the _minor_ release branches of DuckDB and the Python package |

### Release Runbooks

We cut a new **stable minor release** with the following steps:
1. Create a PR on `main` to pin the DuckDB submodule to the tag of its current release.
1. Iff all tests pass in CI, merge the PR.
1. Manually start the release workflow with the hash of this commit, and the tag name.
1. Iff all goes well, create a new PR to let the submodule track DuckDB main.

We cut a new **stable micro release** with the following steps:
1. Create a PR on the minor release branch to pin the DuckDB submodule to the tag of its current release.
1. Iff all tests pass in CI, merge the PR.
1. Manually start the release workflow with the hash of this commit, and the tag name.
1. Iff all goes well, create a new PR to let the submodule track DuckDB's minor release branch.

We cut a new **stable post release** with the following steps:
1. Create a PR on the post release branch to pin the DuckDB submodule to the tag of its current release.
1. Iff all tests pass in CI, merge the PR.
1. Manually start the release workflow with the hash of this commit, and the tag name.
1. Iff all goes well, create a new PR to let the submodule track DuckDB's minor release branch.

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

The package uses `setuptools_scm` with `scikit-build` for automatic version determination, and implements a custom
versioning scheme.

- **pyproject.toml configuration**:
  ```toml
  [tool.scikit-build]
  metadata.version.provider = "scikit_build_core.metadata.setuptools_scm"
  
  [tool.setuptools_scm]
  version_scheme = "duckdb_packaging._setuptools_scm_version:version_scheme"
  ```

- **Environment variables**:
  - `MAIN_BRANCH_VERSIONING=0`: Use release branch versioning (patch increments)
  - `MAIN_BRANCH_VERSIONING=1`: Use main branch versioning (minor increments)
  - `OVERRIDE_GIT_DESCRIBE`: Override version detection

## Conventions

* Follow the [Google Python styleguide](https://google.github.io/styleguide/pyguide.html)
  * See the section on [Comments and Docstrings](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings)