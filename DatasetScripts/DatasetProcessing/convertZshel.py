import json
import os
from tqdm import tqdm

basePath = "./Datasets/zeshel/documents/"


def load_json(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def load_corpuses():
    corpuses = os.listdir(basePath)
    corpusesDict = {}
    for corpus in corpuses:
        data = []
        with open(basePath + corpus, "r", encoding="utf-8") as file:
            for line in file:
                data.append(json.loads(line))
        corpusesDict[corpus.replace(".json", "")] = data
    return corpusesDict


def findDoc(document_id_to_find, corpus):
    matching_objects = [
        obj for obj in corpus if obj["document_id"] == document_id_to_find
    ]
    return matching_objects[0] if matching_objects else None


def merge_data(train_data, corpus_data):
    merged_dataset = []

    for entry in train_data:
        try:
            corpuseName = entry["corpus"]
            corpusDoc = findDoc(entry["document_id"], corpus_data[corpuseName])
            startIdx = entry["start_index"]
            endIdx = entry["end_index"]
            entSurroundedText = (
                corpusDoc["text"][:startIdx]
                + "[START_ENT]"
                + corpusDoc["text"][startIdx:endIdx]
                + "[END_ENT]"
                + corpusDoc["text"][endIdx:]
            )
            merged_entry = {
                "id": len(merged_dataset),
                "input": entSurroundedText,
                "output": [
                    {
                        "answer": entry["text"],
                        "provenance": [{"title": corpusDoc["title"]}],
                    }
                ],
                "candidates": [],
                "answer": entry["text"],
            }

            merged_dataset.append(merged_entry)
        except:
            print("Error with entry: ", entry)

    return merged_dataset


corpuses = load_corpuses()
# Load data from JSON files
train_data = load_json("./Datasets/zeshel/mentions/test.json")
# corpus_data = [load_json(entry["corpus"] + ".json") for entry in train_data]

# # Merge data
merged_dataset = merge_data(train_data, corpuses)

# # Save merged dataset to a new JSON file
with open("./zeshel-conv.json", "w", encoding="utf-8") as output_file:
    json.dump(merged_dataset, output_file, indent=2)
