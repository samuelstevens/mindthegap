# Mind the Gap: Evaluating Vision Systems in Small Data Applications

By [Samuel Stevens](https://samuelstevens.me), [S M Rayeed](https://smrayeed.github.io), and [Jenna Kline](https://jennamk14.github.io).


> Code to reproduce our findings in [Mind the Gap: Evaluating Vision Systems in Small Data Applications](https://arxiv.org/abs/2504.06486).

We looked a lot of recent AI methods papers (DINOv2, Gemini Flash 1.5, Claude Sonnet 3.7, V-JEPA, etc) and measured how many training samples were used in each reported evaluation task.
We found that *no papers use any tasks between 100 and 1K training samples*.

| ![Evaluations](https://raw.githubusercontent.com/samuelstevens/mindthegap/main/docs/assets/tasks.png) |
|:--:|
| Image Credit: [arxiv.org/pdf/2504.06486](https://arxiv.org/pdf/2504.06486) |

We decided to use [NeWT](https://github.com/visipedia/newt/tree/main) to evaluate recent AI methods in this regime of 100-1K training samples and reported our findings.

If you want our raw data, you can download it from [`data/results.sqlite.gz`](data/results.sqlite.gz)
Unzip the sqlite3 file and move it to `results/`, then run the below scripts.

---

To reproduce our work, follow the instructions below:

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
