import json
from tqdm import tqdm
import requests
from multiprocessing import Pool

filepath = "./validation.jsonl"

processed = []

from sklearn.cluster import DBSCAN
import numpy as np


def compute_clusters(vectors):
    try:
        clustering = DBSCAN(eps=1, min_samples=1).fit(vectors)
        labels = clustering.labels_

        # Count the number of occurrences of each value in 'labels'
        counts = np.bincount(labels[labels >= 0])  # Ignore noise labeled as -1

        # Get the cluster label that occurs most frequently
        most_common_label = counts.argmax()

        # Get the indices of the items in the most common cluster
        indices = np.arange(len(labels))  # Array of indices
        most_common_cluster_indices = indices[labels == most_common_label]
        return most_common_cluster_indices
    except:
        return []


def get_wikidata_embedding(wikidata_id):
    url = "http://localhost:5000/api/vector/"
    response = requests.get(url + wikidata_id)
    if response.status_code == 200:
        return response.json()["vector"]
    else:
        return []


def getRelatedEntities(entry):
    embeddings = []
    for linked in entry["linked_ents"]:
        embedding = get_wikidata_embedding(linked[3])
        if len(embedding) > 0:
            embeddings.append(embedding)
    related = compute_clusters(embeddings)
    most_related = []
    for index in related:
        most_related.append(entry["linked_ents"][index])
    entry["most_related"] = most_related
    related_string = "<additional> "
    for related in most_related:
        if len(related) == 4:
            print(related)
            if related[2] != None:
                related_string += related[2] + " "
    related_string += "</additional>"
    entry["input"] = entry["input"] + related_string
    return entry


if __name__ == "__main__":
    with open(filepath, "r") as file:
        dataset = []
        processed = []
        for line in tqdm(file):
            dataset.append(json.loads(line))

        # with Pool(2) as p:
        #     processed = list(tqdm(p.imap(getRelatedEntities, dataset), total=len(dataset)))
        for line in tqdm(dataset):
            processed.append(getRelatedEntities(line))


with open("./valid2.jsonl", "w") as file:
    for line in processed:
        json.dump(line, file)
        file.write("\n")
