import json
from tqdm import tqdm
import gatenlp
import joblib
from multiprocessing import Pool


def get_dataset(ds_path):
    dataset = []

    with open(ds_path, "r") as file:
        for line in file:
            # Load each line as a JSON object
            data_line = json.loads(line)

            dataset.append(data_line)
    return dataset


def convert_to_gatenlp_doc(data):
    print(data)
    # Create a GATE document
    doc = gatenlp.Document()
    doc.text = data["input"].replace("[START_ENT]", "").replace("[END_ENT]", "")
    # add "gold" annotation set
    annset = doc.annset("gold")
    # find start and end index
    start = doc.text.find(data["meta"]["mention"])
    end = start + len(data["meta"]["mention"])

    # add "mention" annotation with linking features

    ann = annset.add(start=start, end=end, anntype="MISC", features=[])

    return doc


ds_names = [
    "aida",
    "msnbc",
    "ace2004",
    "aquaint",
    "clueweb",
    "wiki",
]

preformances = []

for dataset in tqdm(ds_names):
    gatenlp_docs = []
    ds = get_dataset(f"./Datasets/Base/{dataset}-test-kilt-nil.jsonl")
    with Pool(processes=4) as pool:  # Adjust the number of processes as needed
        gatenlp_docs = list(tqdm(pool.imap(convert_to_gatenlp_doc, ds), total=len(ds)))

    # save the documents

    joblib.dump(gatenlp_docs, f"./Datasets/Base/{dataset}-gatenlp.pkl")
