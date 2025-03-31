"""Quickly calculates costs for all MLLM experiments in USD.

Calculate the cost by specifying newt_root:
"""

import concurrent.futures
import json
import logging
import multiprocessing
import os.path
import random
import sqlite3

import beartype
import litellm
import polars as pl
import tyro

from mindthegap import config, helpers, mllms, newt

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("costs")


@beartype.beartype
def get_cost(cfg: config.Experiment, task_name: str) -> tuple[str, str, int, float]:
    """Gets cost for all tasks in NeWT for a given config."""
    mllm = mllms.load_mllm(cfg.model)

    rng = random.Random(hash(cfg))

    train_dataset, test_dataset = newt.get_splits_mllm(cfg, task_name)
    i_train = list(range(len(train_dataset)))
    if cfg.n_train >= 0:
        i_train = rng.sample(i_train, k=min(cfg.n_train, len(i_train)))
    else:
        i_train = i_train

    train_examples = [train_dataset[i].to_example(rng) for i in i_train]

    # Randomly choose the test example.
    test_i = rng.randrange(0, len(test_dataset))
    example = test_dataset[test_i]

    # Set up prompt.
    n = 0
    fewshot = []
    while mllms.fits(cfg, fewshot, example.img_b64, example.make_user(rng)) and (
        cfg.n_train < 0 or n < cfg.n_train
    ):
        # Add another example.
        n += 1
        fewshot = train_examples[:n]

    messages = mllms.make_prompt(cfg, fewshot, example.img_b64, example.make_user(rng))
    n_tokens = litellm.token_counter(model=cfg.model.ckpt, messages=messages)

    cost = mllm.usd_per_m_input / 1_000_000 * n_tokens
    return cfg.model.ckpt, task_name, cfg.n_train, cost


@beartype.beartype
def make_cost_lookup(newt_root: str) -> dict[tuple[str, str, int], float]:
    """Uses multiprocessing to quickly construct a lookup from model checkpoint, newt task and number of training samples to mean cost."""
    preds_df = pl.read_database(
        "SELECT results.exp_cfg FROM results", sqlite3.connect("results/results.sqlite")
    )

    task_names = newt.get_df(newt_root).get_column("task").unique().to_list()

    jobs = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=32, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for exp_cfg_s in helpers.progress(
            preds_df.get_column("exp_cfg").unique().to_list(),
            desc="submitting",
            every=20,
        ):
            exp_cfg = json.loads(exp_cfg_s)
            exp_cfg["newt"] = config.Newt(**exp_cfg["newt"])
            exp_cfg["model"] = config.Model(**exp_cfg["model"])
            if "newt_data" in exp_cfg:
                exp_cfg["newt_root"] = exp_cfg.pop("newt_data")
            assert "newt_root" in exp_cfg
            cfg = config.Experiment(**exp_cfg)

            if cfg.model.method != "mllm":
                continue

            for task_name in task_names:
                jobs.append(pool.submit(get_cost, cfg, task_name))

        logger.info("Submitted %d jobs.", len(jobs))

        lookup = {}
        for future in helpers.progress(
            concurrent.futures.as_completed(jobs),
            desc="calculating",
            every=len(jobs) // 100,
            total=len(jobs),
        ):
            model_ckpt, task_name, n_train, cost = future.result()
            lookup[(model_ckpt, task_name, n_train)] = cost
        return lookup


if __name__ == "__main__":
    cost_lookup = tyro.cli(make_cost_lookup)
    with open(os.path.join("results", "costs.json"), "w") as fd:
        cost_lookup_json = [
            {"ckpt": ckpt, "task_name": task_name, "n_train": n_train, "cost_usd": cost}
            for (ckpt, task_name, n_train), cost in cost_lookup.items()
        ]
        json.dump(cost_lookup_json, fd)
