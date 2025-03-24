""" """

import os.path
import pathlib
import sqlite3

import beartype
import jinja2
import numpy as np
import polars as pl

env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(pathlib.Path(__file__).parent),
    autoescape=jinja2.select_autoescape(),
)

bin_edges = [0, 1, 3, 10, 30, 100, 200, 300, 400, 500, 600]
tasks = [
    ("appearance", "species"),
    ("appearance", "attribute"),
    ("appearance", "health"),
    ("appearance", "age"),
    ("gestalt", None),
    ("context", None),
    ("counting", None),
    ("behavior", None),
]

models = {
    "google/gemini-2.0-flash-001": "Gemini Flash 2.0",
    "google/gemini-flash-1.5-8b": "Gemini Flash 1.5 8B",
    "qwen/qwen2.5-vl-72b-instruct": "Qwen2.5-VL 72B",
    "qwen/qwen-2-vl-7b-instruct": "Qwen2-VL 7B",
    # CLIP
    "ViT-B-16/openai": "CLIP ViT-B/16",
    "ViT-L-14/openai": "CLIP ViT-L/14",
    "ViT-L-14-336/openai": "CLIP ViT-L/14 (336px)",
    # SigLIP
    "ViT-B-16-SigLIP/webli": "SigLIP ViT-B/16",
    "ViT-B-16-SigLIP-256/webli": "SigLIP ViT-B/16 (256px)",
    "ViT-B-16-SigLIP-384/webli": "SigLIP ViT-B/16 (384px)",
    "ViT-B-16-SigLIP-512/webli": "SigLIP ViT-B/16 (512px)",
    "ViT-L-16-SigLIP-256/webli": "SigLIP ViT-L/16 (256px)",
    "ViT-L-16-SigLIP-384/webli": "SigLIP ViT-L/16 (384px)",
    "ViT-SO400M-14-SigLIP/webli": "SigLIP ViT-SO400M/14",
    "ViT-SO400M-14-SigLIP-384/webli": "SigLIP ViT-SO400M/14 (384px)",
    # DINOv2
    "vit_base_patch14_reg4_dinov2.lvd142m": "DINOv2 ViT-B/14",
    "vit_large_patch14_reg4_dinov2.lvd142m": "DINOv2 ViT-L/14",
    "vit_small_patch14_reg4_dinov2.lvd142m": "DINOv2 ViT-S/14",
    "vit_giant_patch14_reg4_dinov2.lvd142m": "DINOv2 ViT-g/14",
    # BioCLIP
    "hf-hub:imageomics/bioclip": "BioCLIP ViT-B/16",
}

model_ranks = {model: i for i, model in enumerate(models)}


@beartype.beartype
def bootstrap(scores, n_resamples: int = 1000):
    scores = np.array(scores)
    # Vectorized bootstrap: sample all at once
    boot_samples = np.random.choice(
        scores, size=(n_resamples, len(scores)), replace=True
    )
    boot_means = boot_samples.mean(axis=1)
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    return {
        "mean": np.mean(scores),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


BootstrapResult = pl.Struct([
    pl.Field("mean", pl.Float64),
    pl.Field("ci_lower", pl.Float64),
    pl.Field("ci_upper", pl.Float64),
])


@beartype.beartype
def mllm_only(df):
    # Document this function. AI!
    for n_train in (0, 1):
        template = env.get_template(os.path.join("tables", "mllm-only.tmpl"))

        data = []
        for cluster, subcluster in tasks:
            filtered_df = df.filter(
                (pl.col("task_cluster") == cluster)
                & (pl.col("n_train_bucketed") == n_train)
            )
            if subcluster:
                filtered_df = filtered_df.filter(
                    (pl.col("task_subcluster") == subcluster)
                )

            subcluster = subcluster.capitalize() if subcluster is not None else "-"

            for model_ckpt, mean, ci_lower, ci_upper in (
                filtered_df.sort(by=("n_train_bucketed", "model_rank"))
                .select("model_ckpt", "mean", "ci_lower", "ci_upper")
                .rows()
            ):
                model_ckpt = models.get(model_ckpt, model_ckpt)
                data.append({
                    "cluster": cluster.capitalize(),
                    "subcluster": subcluster,
                    "model": model_ckpt,
                    "train": int(n_train),
                    "mean": mean,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                })

        label = f"all-results-n{n_train}"
        table = template.render(data=data, label=label, caption="")
        with open(os.path.join("results", f"{label}.tex"), "w") as fd:
            fd.write(table)


@beartype.beartype
def main(db: str = os.path.join("results", "results.sqlite")):
    """Generate LaTeX tables from benchmark results.

    This function reads benchmark results from a SQLite database, computes bootstrap confidence intervals for each model's performance, and generates LaTeX tables for zero-shot (n_train=0) and one-shot (n_train=1) settings. The tables are saved to the results directory.

    Args:
        db: Path to the SQLite database containing benchmark results.
    """
    df = (
        pl.read_database(
            "SELECT results.exp_cfg, results.task_cluster, results.task_subcluster, results.model_ckpt, predictions.score, predictions.n_train FROM results JOIN predictions ON results.rowid = predictions.result_id",
            sqlite3.connect(db),
            infer_schema_length=100_000,
        )
        .lazy()
        .filter(pl.col("model_ckpt").is_in(models))
        .with_columns(
            n_train_bucketed=pl.col("n_train")
            .cut(bin_edges, include_breaks=True, left_closed=False)
            .struct.field("breakpoint")
        )
        .group_by("task_cluster", "task_subcluster", "model_ckpt", "n_train_bucketed")
        .all()
        .with_columns(
            pl.col("score")
            .map_elements(bootstrap, return_dtype=BootstrapResult)
            .alias("boot"),
        )
        .with_columns(
            mean=pl.col("boot").struct.field("mean"),
            ci_lower=pl.col("boot").struct.field("ci_lower"),
            ci_upper=pl.col("boot").struct.field("ci_upper"),
            model_rank=pl.col("model_ckpt").replace(model_ranks),
        )
        .drop("score", "exp_cfg", "boot", "n_train")
        .collect()
    )

    mllm_only(df)


if __name__ == "__main__":
    main()
