# Mind the Gap: Evaluating Vision Systems in Small Data Applications

Reproducing our work:

With [`uv`](https://docs.astral.sh/uv/getting-started/installation/)

```sh
uv run benchmark.py --help
```

This will run the benchmark script, which will force a `.venv` to be created.

Set the $NFS and $USER variables or edit the configs in `configs/*.toml` to point towards your NeWt data.
You can download the data using:

```sh
uv run mindthegap/download.py --help
```

Then run

```sh
uv run benchmark.py --cfg configs/mllms.toml
```

This will create the results database and check if you have any results (you probably don't).
It will report what jobs it has to run.

To actually run the jobs, use:

```sh
uv run benchmark.py --cfg configs/mllms.toml --no-dry-run
```

To recreate our figures, run the `notebookes/figures.py` notebook with [marimo](https://github.com/marimo-team/marimo).
