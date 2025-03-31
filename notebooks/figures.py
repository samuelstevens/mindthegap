import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import dataclasses
    import json
    import math
    import random
    import sqlite3
    import sys

    import beartype
    import marimo as mo
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl

    if "." not in sys.path:
        sys.path.append(".")
    from mindthegap import config, mllms, newt, reporting

    return (
        beartype,
        config,
        dataclasses,
        json,
        math,
        mllms,
        mo,
        mpl,
        newt,
        np,
        pl,
        plt,
        random,
        reporting,
        sqlite3,
        sys,
    )


@app.cell
def _(sqlite3):
    conn = sqlite3.connect("results/results.sqlite")
    return (conn,)


@app.cell
def _(mo):
    refresh_button = mo.ui.run_button()
    refresh_button
    return (refresh_button,)


@app.cell
def _(conn, mo, pl, refresh_button):
    mo.stop(not refresh_button.value)

    bin_edges = [0, 1, 3, 10, 30, 100, 200, 300, 400, 500, 600]

    preds_df = pl.read_database(
        "SELECT results.exp_cfg, results.task_cluster, results.task_subcluster, results.model_ckpt, predictions.score, predictions.n_train FROM results JOIN predictions ON results.rowid = predictions.result_id",
        conn,
        infer_schema_length=100_000,
    ).with_columns(
        n_train_bucketed=pl.col("n_train")
        .cut(bin_edges, include_breaks=True, left_closed=False)
        .struct.field("breakpoint")
    )
    return bin_edges, preds_df


@app.cell
def _(preds_df):
    preds_df
    return


@app.cell
def _(beartype, np, pl):
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
    return BootstrapResult, bootstrap


@app.cell
def _(beartype, bootstrap, newt, pl, random):
    @beartype.beartype
    def get_random_scores() -> dict[
        tuple[str | None, str | None], tuple[float, float, float]
    ]:
        newt_root = "/$NFS/$USER/datasets/newt"
        df = newt.get_df(newt_root)

        rng = random.Random(17)

        random_scores = {}

        for task_cluster, task_subcluster, n in (
            df.group_by("task_cluster", "task_subcluster")
            .agg(pl.col("id").count())
            .rows()
        ):
            scores = [0 if rng.random() > 0.5 else 1 for _ in range(n)]
            bootstrapped = bootstrap(scores)
            random_scores[(task_cluster, task_subcluster)] = (
                bootstrapped["ci_lower"],
                bootstrapped["mean"],
                bootstrapped["ci_upper"],
            )

        scores = [0 if rng.random() > 0.5 else 1 for _ in range(len(df))]
        bootstrapped = bootstrap(scores)
        random_scores[(None, None)] = (
            bootstrapped["ci_lower"],
            bootstrapped["mean"],
            bootstrapped["ci_upper"],
        )

        return random_scores

    random_scores = get_random_scores()
    return get_random_scores, random_scores


@app.cell
def _(beartype, dataclasses):
    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Line:
        label: str
        color: tuple[float, float, float]
        linestyle: str | tuple[int, tuple[int, int]]

    return (Line,)


@app.cell
def _(
    BootstrapResult,
    Line,
    bootstrap,
    pl,
    plt,
    preds_df,
    random_scores,
    reporting,
):
    def make_main_fig():
        models = {
            "google/gemini-2.0-flash-001": Line(
                "Gemini Flash 2.0", reporting.BLUE_RGB01, "solid"
            ),
            "qwen/qwen2.5-vl-72b-instruct": Line(
                "Qwen2.5-VL 72B", reporting.SEA_RGB01, "solid"
            ),
            "ViT-L-14/openai": Line("CLIP ViT-L/14", reporting.CREAM_RGB01, "dashed"),
            "ViT-SO400M-14-SigLIP/webli": Line(
                "SigLIP ViT-SO400M/14", reporting.GOLD_RGB01, "dashed"
            ),
            "vit_giant_patch14_reg4_dinov2.lvd142m": Line(
                "DINOv2 ViT-g/14", reporting.RUST_RGB01, "dashed"
            ),
        }

        df = (
            preds_df.lazy()
            .drop("n_train")
            .filter(pl.col("model_ckpt").is_in(models.keys()))
            .group_by(
                "task_cluster", "task_subcluster", "n_train_bucketed", "model_ckpt"
            )
            .all()
            .with_columns(
                pl.col("score")
                .map_elements(bootstrap, return_dtype=BootstrapResult)
                .alias("boot")
            )
            .drop("score")
            .with_columns(
                mean=pl.col("boot").struct.field("mean"),
                ci_lower=pl.col("boot").struct.field("ci_lower"),
                ci_upper=pl.col("boot").struct.field("ci_upper"),
            )
            .collect()
        )

        fig, axes = plt.subplots(
            ncols=4, nrows=2, figsize=(16, 8), sharex=True, sharey=True, dpi=300
        )

        axes = axes.reshape(-1)
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
        for ax, (cluster, subcluster) in zip(axes, tasks):
            for model_ckpt, line in models.items():
                filtered_df = df.filter(
                    (pl.col("model_ckpt") == model_ckpt)
                    & (pl.col("task_cluster") == cluster)
                )
                if subcluster:
                    filtered_df = filtered_df.filter(
                        (pl.col("task_subcluster") == subcluster)
                    )

                if filtered_df.is_empty():
                    print(f"Skipping {line.label} for {cluster, subcluster}.")
                    continue

                filtered_df = filtered_df.sort("n_train_bucketed")

                means = filtered_df.get_column("mean").to_list()
                lowers = filtered_df.get_column("ci_lower").to_list()
                uppers = filtered_df.get_column("ci_upper").to_list()
                xs = filtered_df.get_column("n_train_bucketed").to_list()

                ax.plot(
                    xs,
                    means,
                    marker=".",
                    label=line.label,
                    color=line.color,
                    alpha=0.8,
                    linestyle=line.linestyle,
                )
                ax.fill_between(
                    xs, lowers, uppers, alpha=0.2, color=line.color, linewidth=0
                )

                if min(xs) != 0:
                    # ViT, plot random chance
                    lower, mean, upper = random_scores[(cluster, subcluster)]
                    ax.plot(
                        [0, 1, 3],
                        [mean, mean, means[0]],
                        marker=".",
                        color=line.color,
                        alpha=0.4,
                        linestyle=(0, (1, 5)),
                    )

                    ax.fill_between(
                        [0, 1, 3],
                        [lower, lower, lowers[0]],
                        [upper, upper, uppers[0]],
                        alpha=0.1,
                        color=line.color,
                        linewidth=0,
                    )

                ax.set_ylim(0, 1.05)
                if subcluster:
                    ax.set_title(f"{cluster.capitalize()} ({subcluster.capitalize()})")
                else:
                    ax.set_title(f"{cluster.capitalize()}")
                ax.set_xscale("symlog", linthresh=2)
                ax.set_xticks(
                    [0, 1, 3, 10, 30, 100, 300, 1000], [0, 1, 3, 10, 30, 100, 300, "1K"]
                )
                ax.set_xlim(-0.15, 1100)
                ax.spines[["right", "top"]].set_visible(False)

        axes[0].set_ylabel("Mean Accuracy")
        axes[4].set_ylabel("Mean Accuracy")

        axes[4].set_xlabel("Number of Training Samples")
        axes[5].set_xlabel("Number of Training Samples")
        axes[6].set_xlabel("Number of Training Samples")
        axes[7].set_xlabel("Number of Training Samples")

        ax.legend(loc="lower right")

        fig.tight_layout(pad=0.0)
        fig.savefig("results/main.pdf")
        return fig

    make_main_fig()
    return (make_main_fig,)


@app.cell
def _(
    BootstrapResult,
    Line,
    bootstrap,
    pl,
    plt,
    preds_df,
    random_scores,
    reporting,
):
    def make_hook_fig():
        models = {
            "google/gemini-2.0-flash-001": Line(
                "Gemini Flash 2.0", reporting.BLUE_RGB01, "dashed"
            ),
            "qwen/qwen2.5-vl-72b-instruct": Line(
                "Qwen2.5-VL 72B", reporting.SEA_RGB01, "dashed"
            ),
            "ViT-L-14/openai": Line("CLIP ViT-L/14", reporting.CREAM_RGB01, "solid"),
            "vit_giant_patch14_reg4_dinov2.lvd142m": Line(
                "DINOv2 ViT-g/14", reporting.RUST_RGB01, "solid"
            ),
        }

        df = (
            preds_df.lazy()
            .drop("n_train")
            .filter(pl.col("model_ckpt").is_in(models.keys()))
            .group_by("n_train_bucketed", "model_ckpt")
            .all()
            .with_columns(
                pl.col("score")
                .map_elements(bootstrap, return_dtype=BootstrapResult)
                .alias("boot")
            )
            .drop("score")
            .with_columns(
                mean=pl.col("boot").struct.field("mean"),
                ci_lower=pl.col("boot").struct.field("ci_lower"),
                ci_upper=pl.col("boot").struct.field("ci_upper"),
            )
            .collect()
        )

        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)

        for model_ckpt, line in models.items():
            filtered_df = df.filter((pl.col("model_ckpt") == model_ckpt))

            if filtered_df.is_empty():
                print(f"Skipping {line.label}.")
                continue

            filtered_df = filtered_df.sort("n_train_bucketed")

            means = filtered_df.get_column("mean").to_list()
            lowers = filtered_df.get_column("ci_lower").to_list()
            uppers = filtered_df.get_column("ci_upper").to_list()
            xs = filtered_df.get_column("n_train_bucketed").to_list()

            ax.plot(
                xs,
                means,
                marker=".",
                label=line.label,
                color=line.color,
                alpha=0.8,
                linestyle=line.linestyle,
            )
            ax.fill_between(
                xs, lowers, uppers, alpha=0.2, color=line.color, linewidth=0
            )

            if min(xs) != 0:
                # ViT, plot random chance
                lower, mean, upper = random_scores[(None, None)]
                ax.plot(
                    [0, 1, 3],
                    [mean, mean, means[0]],
                    marker=".",
                    color=line.color,
                    alpha=0.4,
                    linestyle=(0, (1, 5)),
                )

                ax.fill_between(
                    [0, 1, 3],
                    [lower, lower, lowers[0]],
                    [upper, upper, uppers[0]],
                    alpha=0.1,
                    color=line.color,
                    linewidth=0,
                )

            ax.set_ylim(0.45, 1.05)
            ax.set_xscale("symlog", linthresh=2)
            ax.set_xticks(
                [0, 1, 3, 10, 30, 100, 300, 1000], [0, 1, 3, 10, 30, 100, 300, "1K"]
            )
            ax.set_xlim(-0.15, 1100)

        ax.set_ylabel("Mean Accuracy")
        ax.set_xlabel("Number of Training Samples")
        ax.spines[["right", "top"]].set_visible(False)

        ax.legend(loc="best")
        fig.tight_layout(pad=0.0)
        fig.savefig("results/hook.pdf")
        return fig

    make_hook_fig()
    return (make_hook_fig,)


@app.cell
def _(
    Line,
    beartype,
    bootstrap,
    json,
    newt,
    np,
    pl,
    plt,
    preds_df,
    random,
    reporting,
):
    def make_mllm_cost_fig():
        models = {
            "google/gemini-2.0-flash-001": Line(
                "Gemini Flash 2.0", reporting.BLUE_RGB01, "solid"
            ),
            "google/gemini-flash-1.5-8b": Line(
                "Gemini Flash 1.5 8B", reporting.CYAN_RGB01, "dashed"
            ),
            "qwen/qwen2.5-vl-72b-instruct": Line(
                "Qwen2.5-VL 72B", reporting.SEA_RGB01, "solid"
            ),
            "qwen/qwen-2-vl-7b-instruct": Line(
                "Qwen2-VL 7B", reporting.SEA_RGB01, "dashed"
            ),
        }

        newt_root = "/$NFS/$USER/datasets/newt"

        task_lookup = {
            (task_cluster, task_subcluster): task_names
            for task_cluster, task_subcluster, task_names in newt.get_df(newt_root)
            .group_by("task_cluster", "task_subcluster")
            .all()
            .select("task_cluster", "task_subcluster", "task")
            .rows()
        }

        with open("results/costs.json") as fd:
            cost_lookup = {
                (row["ckpt"], row["task_name"], row["n_train"]): row["cost_usd"]
                for row in json.load(fd)
            }

        @beartype.beartype
        def get_cost(
            model_ckpt: str,
            task_cluster: str,
            task_subcluster: str | None,
            n_train_bucketed: int | float,
        ) -> float:
            n_train = int(n_train_bucketed)
            task_name = random.choice(task_lookup[(task_cluster, task_subcluster)])
            while (model_ckpt, task_name, n_train) not in cost_lookup:
                n_train += 1
            return cost_lookup[(model_ckpt, task_name, n_train)]

        df = (
            preds_df.lazy()
            .filter(pl.col("model_ckpt").is_in(models.keys()))
            .with_columns(
                cost_usd=pl.struct(
                    "model_ckpt", "task_cluster", "task_subcluster", "n_train_bucketed"
                ).map_elements(lambda cols: get_cost(**cols), return_dtype=pl.Float64),
            )
            .group_by("n_train_bucketed", "model_ckpt")
            .all()
            .with_columns(
                score_boot=pl.col("score").map_elements(
                    bootstrap,
                    return_dtype=pl.Struct([
                        pl.Field("mean", pl.Float64),
                        pl.Field("ci_lower", pl.Float64),
                        pl.Field("ci_upper", pl.Float64),
                    ]),
                ),
                cost_usd=pl.col("cost_usd").list.mean(),
            )
            .drop("score")
            .with_columns(mean=pl.col("score_boot").struct.field("mean"))
            .collect()
        )

        fig, ax = plt.subplots(figsize=(5, 4), dpi=300)

        for model_ckpt, line in models.items():
            filtered_df = df.filter((pl.col("model_ckpt") == model_ckpt)).sort(
                "n_train_bucketed"
            )

            means = filtered_df.get_column("mean").to_list()
            xs = filtered_df.get_column("n_train_bucketed").to_list()
            costs = filtered_df.get_column("cost_usd").to_numpy()

            costs = np.interp(costs, [costs.min(), costs.max()], [8, 1024])

            ax.scatter(
                xs,
                means,
                marker=".",
                label=line.label,
                color=line.color,
                alpha=0.8,
                s=costs,
                linewidth=0,
            )
            ax.plot(
                xs,
                means,
                linestyle=line.linestyle,
                linewidth=1,
                color=line.color,
                markersize=0,
            )

            ax.set_ylim(0.45, 0.65)

            ax.set_xscale("symlog", linthresh=2)
            ax.set_xticks(
                [0, 1, 3, 10, 30, 100, 300, 1000], [0, 1, 3, 10, 30, 100, 300, "1K"]
            )
            ax.set_xlim(-0.15, 1100)
            ax.spines[["right", "top"]].set_visible(False)

        ax.legend(loc="lower right")
        fig.tight_layout(pad=0.0)
        fig.savefig("results/mllm_costs.pdf")
        return fig

    make_mllm_cost_fig()
    return (make_mllm_cost_fig,)


@app.cell
def _(beartype, dataclasses, mpl, pl, plt, reporting):
    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Patch:
        model: str
        color: tuple[float, float, float]
        hatch: str | None = None

    def make_hist():
        df = (
            pl.read_csv("docs/existing-work.csv")
            .with_columns(
                size=pl.col("size").str.replace_all("_", "").str.to_integer(strict=True)
                + 1
            )
            .with_columns(logsize=pl.col("size").log10())
        )
        sizes = df.get_column("logsize").to_numpy()
        largest = 8

        for name in sorted([
            b.lower() for b in df.get_column("benchmark").unique().to_list()
        ]):
            print(name)

        fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
        sizes = df.group_by("benchmark", "size").first().get_column("logsize").to_list()
        bins = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7]

        ax.hist(sizes, bins=bins, histtype="bar", color=reporting.CYAN_RGB01)
        xticks = [
            "0",
            "",
            "10",
            "",
            "100",
            "",
            "1K",
            "",
            "10K",
            "",
            "100K",
            "",
            "1M",
            "",
            "10M",
        ]
        ax.set_xticks(bins, xticks)
        ax.spines[["right", "top"]].set_visible(False)
        ax.set_ylabel("Number of Tasks")
        ax.set_xlabel("Number of Training Examples")

        rect = mpl.patches.Rectangle(
            (1.4, -1),
            1.7,
            2.0,
            linewidth=1.8,
            linestyle=(0, (1.5, 1)),
            edgecolor=reporting.GOLD_RGB01,
            facecolor="none",
            clip_on=False,
        )
        ax.add_patch(rect)
        # rect = mpl.patches.Rectangle(
        #     (1.35, -0.8),
        #     1.8,
        #     1.6,
        #     linewidth=1.8,
        #     edgecolor=reporting.GOLD_RGB01,
        #     facecolor="none",
        #     clip_on=False,
        # )
        # ax.add_patch(rect)

        fig.tight_layout(pad=0.0)
        fig.savefig("results/tasks.pdf")
        return fig

    make_hist()
    return Patch, make_hist


@app.cell
def _(pl):
    def get_flops():
        clip_url = "https://raw.githubusercontent.com/mlfoundations/open_clip/refs/heads/main/docs/openclip_results.csv"
        clip_df = pl.read_csv(clip_url).select(
            pl.concat_str(pl.col("name"), pl.col("pretrained"), separator="/").alias(
                "model_ckpt"
            ),
            pl.col("FLOPs (B)").alias("gflops"),
        )
        return clip_df

    flops_df = get_flops()
    return flops_df, get_flops


@app.cell
def _(Line, bootstrap, flops_df, pl, plt, preds_df, reporting):
    def make_flops_fig():
        lines = {
            3: Line("$n=3$", reporting.CREAM_RGB01, "solid"),
            10: Line("$n=10$", reporting.GOLD_RGB01, "solid"),
            30: Line("$n=30$", reporting.RUST_RGB01, "solid"),
            100: Line("$n=100$", reporting.RED_RGB01, "solid"),
        }

        df = (
            preds_df.lazy()
            .drop("n_train")
            .filter(pl.col("model_ckpt").str.contains("/webli"))
            .group_by("n_train_bucketed", "model_ckpt")
            .all()
            .with_columns(
                pl.col("score")
                .map_elements(
                    bootstrap,
                    return_dtype=pl.Struct([
                        pl.Field("mean", pl.Float64),
                        pl.Field("ci_lower", pl.Float64),
                        pl.Field("ci_upper", pl.Float64),
                    ]),
                )
                .alias("boot")
            )
            .drop("score")
            .with_columns(
                mean=pl.col("boot").struct.field("mean"),
                ci_lower=pl.col("boot").struct.field("ci_lower"),
                ci_upper=pl.col("boot").struct.field("ci_upper"),
            )
            .join(flops_df.lazy(), on="model_ckpt")
            .collect()
        )

        fig, ax = plt.subplots(figsize=(5, 3.7), dpi=300)

        for n_train, line in sorted(lines.items(), key=lambda kv: kv[0], reverse=True):
            filtered_df = df.filter((pl.col("n_train_bucketed") == n_train))

            if filtered_df.is_empty():
                print(f"Skipping {line.label}.")
                continue

            filtered_df = filtered_df.sort("gflops")

            means = filtered_df.get_column("mean").to_list()
            lowers = filtered_df.get_column("ci_lower").to_list()
            uppers = filtered_df.get_column("ci_upper").to_list()
            xs = filtered_df.get_column("gflops").to_list()

            ax.plot(
                xs,
                means,
                marker=".",
                label=line.label,
                color=line.color,
                alpha=0.8,
                linestyle=line.linestyle,
            )
            ax.fill_between(
                xs, lowers, uppers, alpha=0.2, color=line.color, linewidth=0
            )

            ax.set_ylim(0.45, 1.0)

        ax.set_ylabel("Mean Accuracy")
        ax.set_xlabel("GFLOPs")
        ax.spines[["right", "top"]].set_visible(False)

        ax.legend(loc="best", alignment="left")
        fig.tight_layout(pad=0.0)
        fig.savefig("results/flops.pdf")
        return fig

    make_flops_fig()
    return (make_flops_fig,)


@app.cell
def _(preds_df):
    preds_df.get_column("model_ckpt").unique()
    return


@app.cell
def _(BootstrapResult, bootstrap, np, pl, plt, preds_df, reporting):
    def make_pretraining_fig():
        vision_ckpts = ("vit_large_patch14_reg4_dinov2.lvd142m",)
        lang_ckpts = (
            "ViT-L-14/openai",
            "ViT-L-16-SigLIP-256/webli",
        )

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

        n_train = 30

        cluster_rank = {cluster: i for i, (cluster, _) in enumerate(tasks)}
        subcluster_rank = {subcluster: i for i, (_, subcluster) in enumerate(tasks)}

        def get_df(ckpts):
            return (
                preds_df.lazy()
                .filter(
                    (pl.col("n_train") == n_train) & (pl.col("model_ckpt").is_in(ckpts))
                )
                .group_by("task_cluster", "task_subcluster")
                .all()
                .with_columns(
                    pl.col("score")
                    .map_elements(bootstrap, return_dtype=BootstrapResult)
                    .alias("boot"),
                    pl.col("task_cluster").replace(cluster_rank).alias("cluster_rank"),
                    pl.col("task_subcluster")
                    .replace(subcluster_rank)
                    .alias("subcluster_rank"),
                )
                .with_columns(
                    mean=pl.col("boot").struct.field("mean"),
                    ci_lower=pl.col("boot").struct.field("ci_lower"),
                    ci_upper=pl.col("boot").struct.field("ci_upper"),
                )
                .drop("score", "exp_cfg", "boot", "n_train", "n_train_bucketed")
                .sort(by=["cluster_rank", "subcluster_rank"])
                .collect()
            )

        fig, ax = plt.subplots(figsize=(5, 3), dpi=300)

        xs = np.arange(8)
        width = 0.35

        vision_df = get_df(vision_ckpts)

        errorbar_kwargs = dict(
            color=reporting.BLACK_RGB01,
            linewidth=0,
            elinewidth=0.6,
            capsize=1,
            capthick=0.6,
        )

        ys = vision_df.get_column("mean").to_numpy()
        yerr = np.abs(ys - vision_df.select("ci_lower", "ci_upper").to_numpy().T)
        ax.bar(
            xs - width / 2, ys, width, label="Vision Only", color=reporting.RUST_RGB01
        )
        ax.errorbar(xs - width / 2, ys, yerr, **errorbar_kwargs)

        lang_df = get_df(lang_ckpts)
        ys = lang_df.get_column("mean").to_numpy()
        yerr = np.abs(ys - lang_df.select("ci_lower", "ci_upper").to_numpy().T)
        ax.bar(
            xs + width / 2,
            ys,
            width,
            label="Language Supervision",
            color=reporting.CREAM_RGB01,
            hatch="///",
        )
        ax.errorbar(xs + width / 2, ys, yerr, **errorbar_kwargs)

        labels = [
            cluster.capitalize() if subcluster is None else subcluster.capitalize()
            for cluster, subcluster in tasks
        ]
        ax.tick_params(axis="x", length=0)
        ax.set_xticks(xs, labels, fontsize=7.2)
        ax.set_ylim(0.43, 1.0)
        ax.set_yticks(
            [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
            ["", "0.5", "", "0.6", "", "0.7", "", "0.8", "", "0.9", "", "1.0"],
        )
        ax.set_ylabel("Mean Accuracy")
        ax.set_xlabel("Tasks")
        ax.spines[["right", "top"]].set_visible(False)
        ax.legend(loc="best", ncols=2)
        # ax.grid(axis="y", linewidth=0.2)
        # ax.set_axisbelow(True)

        fig.tight_layout(pad=0.0)
        fig.savefig(f"results/pretraining-n_train{n_train}.pdf")

        return fig

    make_pretraining_fig()
    return (make_pretraining_fig,)


@app.cell
def _(BootstrapResult, bootstrap, np, pl, plt, preds_df, reporting):
    def make_pretraining_fig_app():
        vision_ckpts = ("vit_large_patch14_reg4_dinov2.lvd142m",)
        lang_ckpts = (
            "ViT-L-14/openai",
            "ViT-L-16-SigLIP-256/webli",
        )

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

        n_trains = (3, 10, 30, 100)

        cluster_rank = {cluster: i for i, (cluster, _) in enumerate(tasks)}
        subcluster_rank = {subcluster: i for i, (_, subcluster) in enumerate(tasks)}

        def get_df(ckpts, n_train):
            return (
                preds_df.lazy()
                .filter(
                    (pl.col("n_train") == n_train) & (pl.col("model_ckpt").is_in(ckpts))
                )
                .group_by("task_cluster", "task_subcluster")
                .all()
                .with_columns(
                    pl.col("score")
                    .map_elements(bootstrap, return_dtype=BootstrapResult)
                    .alias("boot"),
                    pl.col("task_cluster").replace(cluster_rank).alias("cluster_rank"),
                    pl.col("task_subcluster")
                    .replace(subcluster_rank)
                    .alias("subcluster_rank"),
                )
                .with_columns(
                    mean=pl.col("boot").struct.field("mean"),
                    ci_lower=pl.col("boot").struct.field("ci_lower"),
                    ci_upper=pl.col("boot").struct.field("ci_upper"),
                )
                .drop("score", "exp_cfg", "boot", "n_train", "n_train_bucketed")
                .sort(by=["cluster_rank", "subcluster_rank"])
                .collect()
            )

        fig, axes = plt.subplots(
            ncols=2,
            nrows=2,
            figsize=(10, 6),
            dpi=300,  # sharey=True, sharex=True,
        )
        axes = axes.reshape(-1)

        xs = np.arange(8)
        width = 0.35
        labels = [
            cluster.capitalize() if subcluster is None else subcluster.capitalize()
            for cluster, subcluster in tasks
        ]

        for n_train, ax in zip(n_trains, axes):
            vision_df = get_df(vision_ckpts, n_train)

            errorbar_kwargs = dict(
                color=reporting.BLACK_RGB01,
                linewidth=0,
                elinewidth=0.6,
                capsize=1,
                capthick=0.6,
            )

            ys = vision_df.get_column("mean").to_numpy()
            yerr = np.abs(ys - vision_df.select("ci_lower", "ci_upper").to_numpy().T)
            ax.bar(
                xs - width / 2,
                ys,
                width,
                label="Vision Only",
                color=reporting.RUST_RGB01,
            )
            ax.errorbar(xs - width / 2, ys, yerr, **errorbar_kwargs)

            lang_df = get_df(lang_ckpts, n_train)
            ys = lang_df.get_column("mean").to_numpy()
            yerr = np.abs(ys - lang_df.select("ci_lower", "ci_upper").to_numpy().T)
            ax.bar(
                xs + width / 2,
                ys,
                width,
                label="Language Supervision",
                color=reporting.CREAM_RGB01,
                hatch="///",
            )
            ax.errorbar(xs + width / 2, ys, yerr, **errorbar_kwargs)

            ax.spines[["right", "top"]].set_visible(False)
            ax.set_ylim(0.43, 1.0)
            ax.tick_params(axis="x", length=0)
            ax.set_xticks(xs, labels, fontsize=7.2)
            ax.set_yticks(
                [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
                ["", "0.5", "", "0.6", "", "0.7", "", "0.8", "", "0.9", "", "1.0"],
            )
            ax.set_title(f"$n={n_train}$")

        axes[0].set_ylabel("Mean Accuracy")
        axes[2].set_ylabel("Mean Accuracy")

        axes[2].set_xlabel("Tasks")
        axes[3].set_xlabel("Tasks")

        axes[1].legend(loc="best", ncols=2)

        fig.tight_layout(pad=0.0)
        fig.savefig(f"results/pretraining-all.pdf")

        return fig

    make_pretraining_fig_app()
    return (make_pretraining_fig_app,)


@app.cell
def _(pl):
    pl.read_csv("docs/existing-work.csv")
    return


@app.cell
def _(BootstrapResult, bootstrap, pl, preds_df):
    def make_tables():
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

        df = (
            preds_df.lazy()
            .filter(pl.col("model_ckpt").is_in(models))
            .group_by(
                "task_cluster", "task_subcluster", "model_ckpt", "n_train_bucketed"
            )
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

        for n_train in (0, 1):
            print(
                r"""
    \begin{table*}[t]
    \centering
    \begin{tabular}{lllrrr}
    \toprule
    Task Cluster & Task Subcluster & Model & Train & Mean & Confidence Interval \\
    \midrule        
    """.strip()
            )
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
                    print(
                        f"{cluster.capitalize()} & {subcluster} & {model_ckpt} & ${int(n_train)}$ & ${mean:.2f}$ & $[{ci_lower:.2f}, {ci_upper:.2f}]$ \\\\"
                    )

                if cluster != tasks[-1][0]:
                    print(r"\midrule")

            print(
                r"""
    \bottomrule
    \end{tabular}
    \caption{""".strip()
            )
            if n_train == 0:
                print("All results for $0$ training samples.")
            else:
                print("All results for $1$ training sample.")
            print(
                f"""
    Only MLLMs are evaluated because SVMs cannot be fit with fewer than one sample per class.
    }}\\label{{tab:all-results-n{n_train}}}
    \\end{{table*}}
    """.strip()
            )

    make_tables()
    return (make_tables,)


if __name__ == "__main__":
    app.run()
