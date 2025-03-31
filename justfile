docs: fmt
    uv run pdoc3 --force --html --output-dir docs --config latex_math=True mindthegap benchmark
    uv run python scripts/docs.py --in-paths mindthegap benchmark.py --out-fpath docs/llms.txt

lint: fmt
    ruff check --fix .

test: fmt
    uv run pytest mindthegap

fmt:
    ruff format --preview .

