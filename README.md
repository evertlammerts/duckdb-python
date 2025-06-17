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

## Conventions

* Follow the [Google Python styleguide](https://google.github.io/styleguide/pyguide.html)
  * See the section on [Comments and Docstrings](https://google.github.io/styleguide/pyguide.html#s3.8-comments-and-docstrings)