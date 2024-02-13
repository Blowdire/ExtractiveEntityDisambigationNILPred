import json
from tqdm import tqdm
import pandas as pd
import requests

mentions = 0


def get_dataset(ds_path):
    dataset = []
    with open(ds_path, "r") as file:
        for line in file:
            # Load each line as a JSON object
            data_line = json.loads(line)

            dataset.append(data_line)
    return dataset


ds_names = [
    "aida",
    "msnbc",
    "ace2004",
    "aquaint",
    "clueweb",
    "wiki",
]
preformances = []
for dataset in ds_names:
    ds = get_dataset(f"./Datasets/Base/{dataset}-test-kilt-nil.jsonl")
    results_bi = []
    for item in tqdm(ds):
        try:
            obj = {
                "context_left": item["meta"]["left_context"],
                "context_right": item["meta"]["right_context"],
                "mention": item["meta"]["mention"],
            }
            res_biencoder = requests.post(
                "http://10.0.0.113:20980/api/blink/biencoder/mention", json=[obj]
            )
            assert res_biencoder.ok
            results_bi.append(res_biencoder.json())
        except Exception as e:
            print(e)
    correct = 0
    wrong = 0
    correctNIL = 0
    wrongNIL = 0
    index = 0
    for biEncRes in tqdm(results_bi):
        try:
            data = ds[index]
            index += 1
            res_indexer = requests.post(
                "http://10.0.0.113:20982/api/indexer/search",
                json={
                    "encodings": biEncRes["encodings"],
                    "top_k": 10,
                    "only_indexes": [],
                },
            )
            assert res_indexer.ok
            res_indexer = res_indexer.json()[0]
            nilpred_input = [
                {
                    "max_bi": res_indexer[0]["score"],
                    "secondiff": res_indexer[0]["score"] - res_indexer[1]["score"],
                }
            ]
            res_nilp = requests.post(
                "http://10.0.0.113:20983/api/nilprediction", json=nilpred_input
            )
            assert res_nilp.ok
            score = round(res_nilp.json()["nil_score_bi"][0])

            if data["output"][0]["answer"] == "Not In Candidates":
                if score == 1:

                    correct += 1
                    correctNIL += 1
                else:
                    wrong += 1
                    wrongNIL += 1
            else:

                if score == 1:
                    wrong += 1
                else:
                    correct += 1

        except Exception as e:
            print(e)
    acc = correct / len(ds)
    nil_acc = correctNIL / (correctNIL + wrongNIL)
    print(f"Dataset: {dataset} - Acc: {acc} - NIL Acc: {nil_acc}")
    preformances.append(
        {
            "dataset": dataset,
            "acc": acc,
            "nil_acc": nil_acc,
            "correct_nil": correctNIL,
            "wrong_nil": wrongNIL,
        }
    )
perf_df = pd.DataFrame(preformances)
perf_df.to_csv("./results/modelloRiccardo.csv")
