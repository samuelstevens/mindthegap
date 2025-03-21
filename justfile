docs: fmt
    uv run pdoc3 --force --html --output-dir docs --config latex_math=True small_data_metrics benchmark
    uv run python scripts/docs.py --in-paths small_data_metrics benchmark.py --out-fpath docs/llms.txt

lint: fmt
    ruff check --fix .

test: fmt
    uv run pytest small_data_metrics

fmt:
    ruff format --preview .

