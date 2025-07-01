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
* cibuildwheel:
  * `CIBW_BUILD='cp39-*' uvx cibuildwheel --platform linux .`

## Development

We use Astral UV for local development and recommend you do as well. Note: we require pip >= 25.1.0 for development, 
because we work with dependency groups.

Some useful commands:

Install duckdb together with the `dev` dependency group in `editable` mode in a Python 3.9 virtual environment:
```bash
brew install uv
uv sync -p 3.9
```

Run all pytests (this includes tests/slow and _will_ take very long):
```bash
uv run pytest ./tests --verbose
```

Exclude the test/slow directory:
```bash
uv run pytest ./tests --verbose --ignore=./tests/slow
```

Run with coverage:
```bash
COVERAGE=1 uv run coverage run -m pytest ./tests --verbose
```

The `COVERAGE` env var will compile the3 extension with `--coverage`, allowing us to collect coverage stats of C++ 
code as well as Python code.

Check coverage for Python code:
```bash
uv run coverage report
```

- We're not running any mypy typechecking tests at the moment
- We're not running any ruff / linting / formatting at the moment

## Merging changes to pythonpkg from duckdb main

Check the git log for the last changes to the pythonpkg since the last ref you have 

```bash
git log <hash>..HEAD -- tools/pythonpkg/
```

## Which duckdb options should we add to the compile definitions?

```bash
duckdb/src/include on  py_oot [⇕] via C v16.0.0-clang via 🐍 v3.9.6 (duckdb)
❯ ag '#ifdef' | rg 'ifdef (.*)' -or '$1' | sort -u
CreateDirectory
DEBUG
DUCKDB_ALTERNATIVE_VERIFY
DUCKDB_API_1_0
DUCKDB_BUILD_LOADABLE_EXTENSION
DUCKDB_CLANG_TIDY
DUCKDB_DEBUG_ASYNC_SINK_SOURCE
DUCKDB_DEBUG_MOVE
DUCKDB_DEBUG_NO_INLINE
DUCKDB_DEBUG_NO_SAFETY
DUCKDB_DISABLE_POINTER_SALT
DUCKDB_ENABLE_DEPRECATED_API
DUCKDB_EXTENSION_API_UNSTABLE_VERSION
DUCKDB_EXTENSION_API_VERSION_UNSTABLE
DUCKDB_EXTENSION_AUTOINSTALL_DEFAULT
DUCKDB_EXTENSION_AUTOLOAD_DEFAULT
DUCKDB_EXTENSION_NAME
DUCKDB_PLATFORM_RTOOLS
DUCKDB_SMALLER_BINARY
DUCKDB_STATIC_BUILD
DUCKDB_WINDOWS
ERROR
MoveFile
RemoveDirectory
SOME_DEFINE
UUID
WIN32
_MSC_VER
_WIN32
__MINGW32__
__MVS__
__clang__
__cplusplus
interface
max
min
small
```

```bash
duckdb/src/include on  py_oot [⇕] via C v16.0.0-clang via 🐍 v3.9.6 (duckdb)
❯ ag '#if defined' | rg '#if defined\((.*)\)' -or '$1' | sort -u
ADBC_EXPORTING
DEBUG) || defined(UNSAFE_NUMERIC_CAST
DUCKDB_BUILD_LIBRARY) && !defined(DUCKDB_BUILD_LOADABLE_EXTENSION
DUCKDB_CUSTOM_PLATFORM
DUCKDB_DEBUG_NO_SAFETY) || defined(DUCKDB_CLANG_TIDY
DUCKDB_WASM_VERSION
GENERATED_EXTENSION_HEADERS) && !defined(DUCKDB_AMALGAMATION
UINTPTR_MAX
_WIN32
_WIN32) && !defined(__MINGW32__
_WIN32) || defined(_WIN64
_WIN32) || defined(__APPLE__) || defined(__FreeBSD__
__ANDROID__
__MUSL__
__aarch64__) || defined(__ARM_ARCH_ISA_A64
```

```bash
duckdb/src/include on  py_oot [⇕] via C v16.0.0-clang via 🐍 v3.9.6 (duckdb) took 8s
❯ ag '#if ' | grep -v '#if defined' | rg '#if (.*)' -or '$1' | sort -u
!defined(ADBC_EXPORT)
!defined(DUCKDB_DISABLE_POINTER_SALT) && defined(__ANDROID__)
!defined(DUCKDB_EXTENSION_API_VERSION_MAJOR) && !defined(DUCKDB_EXTENSION_API_VERSION_MINOR) &&                    \
!defined(_GNU_SOURCE)
!defined(__GNUC__) || (__GNUC__ >= 5)
( \
((__GNUC__ >= 5) || defined(__clang__)) && defined(__SIZEOF_INT128__)
(DEFAULT_BLOCK_ALLOC_SIZE & (DEFAULT_BLOCK_ALLOC_SIZE - 1) != 0)
(DEFAULT_ROW_GROUP_SIZE % STANDARD_VECTOR_SIZE != 0)
(DEFAULT_ROW_GROUP_SIZE < STANDARD_VECTOR_SIZE)
(DUCKDB_BLOCK_ALLOC_SIZE & (DUCKDB_BLOCK_ALLOC_SIZE - 1) != 0)
(STANDARD_VECTOR_SIZE & (STANDARD_VECTOR_SIZE - 1) != 0)
DUCKDB_EXTENSION_API_VERSION_MAJOR != 1
DUCKDB_EXTENSION_API_VERSION_MINOR > 2 ||                                                                          \
INTPTR_MAX == INT64_MAX
_LIBCPP_STD_VER >= 17
_LIBCPP_STD_VER >= 20
__GNUC__
__has_cpp_attribute(clang::fallthrough)
```