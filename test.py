import json
from tqdm import tqdm
import requests
from multiprocessing import Pool
from sklearn.metrics import pairwise_distances
import numpy as np

filepath = "./valid1.jsonl"

processed = []

from sklearn.cluster import DBSCAN
import numpy as np


def compute_clusters(vectors):
    try:
        vectors = np.array(vectors)
        labels = DBSCAN(eps=1, min_samples=2).fit_predict(vectors)
        
        unique_labels = np.unique(labels)
        selected_label = unique_labels[0]
        cluster_points = vectors[labels == selected_label]
        intra_cluster_distances = pairwise_distances(cluster_points)
        average_intra_cluster_distance = np.sum(intra_cluster_distances) / (
            len(cluster_points) * (len(cluster_points) - 1)
        )
        max_cohesion = average_intra_cluster_distance
        # Calculate intra-cluster distance for each cluster
        for cluster_label in unique_labels:
            cluster_points = vectors[labels == cluster_label]
            if (
                len(cluster_points) > 1
            ):  # Ensure there are at least two points in the cluster for distance calculation
                intra_cluster_distances = pairwise_distances(cluster_points)
                average_intra_cluster_distance = np.sum(intra_cluster_distances) / (
                    len(cluster_points) * (len(cluster_points) - 1)
                )
                if average_intra_cluster_distance > max_cohesion:
                    max_cohesion = average_intra_cluster_distance
                    selected_label = cluster_label
                # print(
                #     f"Cluster {cluster_label}: Average Intra-Cluster Distance = {average_intra_cluster_distance}"
                # )
            

        # Get the indices of the items in the most common cluster
        indices = np.arange(len(labels))  # Array of indices
        most_common_cluster_indices = indices[labels == selected_label]
        return most_common_cluster_indices
    except Exception as e:
        print(e)
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
            if related[2] is not None:
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


with open("./valid3.jsonl", "w") as file:
    for line in processed:
        json.dump(line, file)
        file.write("\n")