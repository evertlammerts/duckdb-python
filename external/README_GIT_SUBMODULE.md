# DuckDB vendored as submodule

The submodule has a relative path. Git resolves it against the super-project’s remote, so a clone of `git@github.com:alice/duckdb-python.git` will automatically look for `git@github.com:alice/duckdb.git`, while a clone of the canonical repo falls back to
`https://github.com/duckdb/duckdb.git`.

### Clone python repo

From the main repo:
```shell
git clone --recurse-submodules git@github.com:duckdb/duckdb-python.git
```

Or from your fork (NOTE that you must also fork duckdb):
```shell
git clone --recurse-submodules git@github.com:<you>/duckdb-python.git
```

### Switch submodule to fork
```shell
git submodule set-url external/duckdb git@github.com:<you>/duckdb.git
```

### Work on a feature branch
```shell
git -C external/duckdb checkout my-branch
```

### Jump to latest DuckDB tag
```shell
git -C external/duckdb fetch --tags && git -C external/duckdb checkout v1.3.1 # Example
```

### Pull latest changes
```shell
git submodule update --remote --merge external/duckdb
```