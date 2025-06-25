<div align="center">
  <picture>
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/duckdb/duckdb/refs/heads/main/logo/DuckDB_Logo-horizontal.svg">
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/duckdb/duckdb/refs/heads/main/logo/DuckDB_Logo-horizontal-dark-mode.svg">
    <img alt="DuckDB logo" src="https://raw.githubusercontent.com/duckdb/duckdb/refs/heads/main/logo/DuckDB_Logo-horizontal.svg" height="100">
  </picture>
</div>
<br />
<p align="center">
  <a href="https://discord.gg/tcvwpjfnZx"><img src="https://shields.io/discord/909674491309850675" alt="discord" /></a>
  <a href="https://pypi.org/project/duckdb/"><img src="https://img.shields.io/pypi/v/duckdb.svg" alt="PyPi Latest Release"/></a>
</p>
<br />
<p align="center">
  <a href="https://duckdb.org">DuckDB.org</a>
  |
  <a href="https://duckdb.org/docs/stable/guides/python/install">User Guide (Python)</a>
  -
  <a href="https://duckdb.org/docs/stable/clients/python/overview">API Docs (Python)</a>
</p>

# DuckDB: A Fast, In-Process, Portable, Open Source, Analytical Database System

* **Simple**: DuckDB is easy to install and deploy. It has zero external dependencies and runs in-process in its host application or as a single binary.
* **Portable**: DuckDB runs on Linux, macOS, Windows, Android, iOS and all popular hardware architectures. It has idiomatic client APIs for major programming languages.
* **Feature-rich**: DuckDB offers a rich SQL dialect. It can read and write file formats such as CSV, Parquet, and JSON, to and from the local file system and remote endpoints such as S3 buckets.
* **Fast**: DuckDB runs analytical queries at blazing speed thanks to its columnar engine, which supports parallel execution and can process larger-than-memory workloads.
* **Extensible**: DuckDB is extensible by third-party features such as new data types, functions, file formats and new SQL syntax. User contributions are available as community extensions.
* **Free**: DuckDB and its core extensions are open-source under the permissive MIT License. The intellectual property of the project is held by the DuckDB Foundation.

## Installation

Install the latest release of DuckDB directly from [PyPi](https://pypi.org/project/duckdb/):

```bash
pip install duckdb
```

Install with all optional dependencies:

```bash
pip install 'duckdb[all]'
```

## Versioning and Releases

The DuckDB Python package versioning and release scheme follows that of DuckDB itself. This means that a `X.Y.Z[.
postN]` release of the Python package ships the DuckDB stable release `X.Y.Z`. The optional `.postN` releases ship the same stable release of DuckDB as their predecessors plus Python package-specific fixes and / or features.

| Types                                                                  | DuckDB Version | Resulting Python Extension Version |
|------------------------------------------------------------------------|----------------|------------------------------------|
| Stable release: DuckDB stable release                                  | `1.3.1`        | `1.3.1`                            |
| Stable post release: DuckDB stable release + Python fixes and features | `1.3.1`        | `1.3.1.postX`                      |
| Nightly micro: DuckDB next micro nightly + Python next micro nightly   | `1.3.2.devM`   | `1.3.2.devN`                       |
| Nightly minor: DuckDB next minor nightly + Python next minor nightly   | `1.4.0.devM`   | `1.4.0.devN`                       |

Note that we do not ship nightly post releases (e.g. we don't ship `1.3.1.post2.dev3`).

## Contributing

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