import pandas as pd
from tqdm import tqdm, trange
from glob import glob
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
import json

# Desired percentage of NIL mentions
perc_nil = 0.60

# Number of batch to create
n_batch = 10

# Desired NIL mentions per batch (in dev and test)
desired_nil = 15

# Desired size of a single dev and test batch
dev_test_desired_batch_size = 100

# Random state
random_state = 1234

# Output dir
outdir = "incremental_dataset"


def create_dataset(name, instanceOf):
    ds_path = (
        f"./Datasets/Base/{name}-test-kilt.jsonl"
        if not instanceOf
        else f"./Datasets/InstanceOf/{name}_test_instanceof.jsonl"
    )

    df = pd.read_json(ds_path, lines=True)
    df["answer"] = df["output"].apply(lambda x: x[0]["answer"])
    mention_frequency = df.groupby("answer").size()

    freq_df = pd.DataFrame(mention_frequency)
    freq_df.columns = ["freq"]

    med_freq = np.median(freq_df["freq"])

    np.random.seed(random_state)

    freq_df["p_formula"] = perc_nil ** (freq_df["freq"] / med_freq)
    s = np.random.uniform(0, 1, freq_df.shape[0])
    freq_df["p_uniform"] = s

    freq_df["NIL"] = freq_df["p_uniform"] < freq_df["p_formula"]
    freq_df.loc[freq_df["NIL"] == True, "freq"] = 0
    train_df_merged = df.join(freq_df, how="left", on="answer")
    print(
        "Percentage of NIL mentions in train:",
        train_df_merged.eval("NIL").sum() / train_df_merged.shape[0] * 100,
    )
    results = []

    for index, row in train_df_merged.iterrows():
        if row["NIL"] == True:
            candidates = row["candidates"]
            filtered_cands = list(
                filter(
                    lambda x: row["answer"] not in x,
                    candidates,
                )
            )
            row["candidates"] = filtered_cands
            row["output"][0]["answer"] = "Not In Candidates"
            results.append(row.to_dict())
        else:
            results.append(row.to_dict())
    with open(ds_path.split(".jsonl")[0] + "-nil.jsonl", "w") as outfile:
        for entry in results:
            json.dump(entry, outfile)
            outfile.write("\n")


ds_names = [
    "msnbc",
    "ace2004",
    "aquaint",
    "clueweb",
    "wiki",
]

for dataset in tqdm(ds_names):
    create_dataset(dataset, False)
    create_dataset(dataset, True)
