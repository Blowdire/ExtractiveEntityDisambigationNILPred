import json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import Dataset
import gc
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

tokenizer = AutoTokenizer.from_pretrained("./robertaLargeNil/checkpoint-2772")
model = AutoModelForQuestionAnswering.from_pretrained(
    "./robertaLargeNil/checkpoint-2772", return_dict=False
).to("cuda:0")


def check_brackets(input_string):
    stack = []
    brackets = {"(": ")", "{": "}", "[": "]"}
    for char in input_string:
        if char in brackets.keys():
            stack.append(char)
        elif char in brackets.values():
            if not stack or brackets[stack.pop()] != char:
                return False
    return not stack


def process_answer(answer, candidates):
    if answer == "Not In Candidates":
        return answer
    else:
        modified_answer = answer.split("</ec>")[0]
        modified_answer = modified_answer.split(": instance of ")[0]
        modified_answer = modified_answer.split(": instance")[0]
        modified_answer = modified_answer.replace("<s>", "")
        modified_answer = modified_answer.replace("</ec", "")
        modified_answer = modified_answer.replace("</s>", "")
        modified_answer = modified_answer.replace("<s", "")
        modified_answer = modified_answer.replace("</", "")
        modified_answer = modified_answer.replace(" ec", "")
        modified_answer = modified_answer.replace(">", "")
        modified_answer = modified_answer.strip()
        if not check_brackets(modified_answer):
            modified_answer = modified_answer.replace("(", "")
            modified_answer = modified_answer.replace(")", "")
        if modified_answer == "Not In Candidates":
            modified_answer = "Not In Candidates"
        if modified_answer == "":
            modified_answer = "Not In Candidates"
        if modified_answer not in candidates:
            modified_answer = "Not In Candidates"
        return modified_answer


def make_prediction(data_entry, nil_prediction):
    with torch.no_grad():
        question = data_entry["input"]
        context = ""
        # if nil_prediction:
        #     context += " Not In Candidates </ec> "
        index = 0
        added = False
        for item in data_entry["candidates"]:
            context += item + f" </ec> "
            index += 1
            if index == 1 and nil_prediction:
                context += " Not In Candidates </ec> "
                added = True
        if not added and nil_prediction:
            context = " Not In Candidates </ec> " + context
        input_pairs = [question, context]

        encodings = tokenizer.encode_plus(
            input_pairs, return_tensors="pt", truncation="only_second"
        ).to("cuda")
        start_scores, end_scores = model(
            encodings["input_ids"], attention_mask=encodings["attention_mask"]
        )
        start_scores = F.softmax(start_scores, dim=1)
        end_scores = F.softmax(end_scores, dim=1)
        # start_scores.to("cpu")
        # end_scores.to("cpu")
        start = torch.argmax(
            start_scores
        )  # Get the most likely beginning of answer with the argmax of the score
        end = (
            torch.argmax(end_scores) + 1
        )  # Get the most likely end of answer with the argmax of the score
        mean_score = 0
        try:
            mean_score = end_scores[0][end.item()] + start_scores[0][start.item()]
        except:
            None
        answer_tokens = encodings["input_ids"][0, start.item() : end.item() + 1]
        answer = ""

        answer = process_answer(tokenizer.decode(answer_tokens), context)
        classified = 0
        if (mean_score < 0.494949) and nil_prediction:
            answer = "Not In Candidates"
        else:
            if answer == "":
                answer = "Not In Candidates"
        del encodings
        del end_scores
        del start_scores
        del start
        del end
        score = float(mean_score)
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "correct": data_entry["output"],
            "non_processed": tokenizer.decode(answer_tokens),
            "predicted": answer,
            "input_phrase": question,
            "scores": score,
            "candidates": context,
        }


def get_dataset(ds_path):
    dataset = []

    with open(ds_path, "r") as file:
        for line in file:
            # Load each line as a JSON object
            data_line = json.loads(line)

            dataset.append(data_line)
    return dataset


ds_names = [
    "wnum",
]

preformances = []

for dataset in tqdm(ds_names):
    ds = get_dataset(f"./Datasets/Base/msnbc-test-kilt-nil.jsonl")
    results = []
    for item in tqdm(ds):
        try:
            pred = make_prediction(item, True)
            results.append(pred)
        except Exception as e:
            print(e)
    correct = 0
    correct_trace = []
    wrong = []
    correct_nil = 0
    wrong_nil = 0
    for result in tqdm(results):
        processed = process_answer(result["predicted"], result["candidates"])
        if processed == result["correct"][0]["answer"]:
            correct += 1
            correct_trace.append(result)
            if processed == "Not In Candidates":
                correct_nil += 1
        else:
            if result["correct"][0]["answer"] == "Not In Candidates":
                wrong_nil += 1
            wrong.append(result)
    acc = correct / len(results)
    nil_acc = -1
    if correct_nil + wrong_nil != 0:
        nil_acc = correct_nil / (correct_nil + wrong_nil)
    print(f"Dataset: {dataset} - Acc: {acc} - NIL Acc: {nil_acc}")
    preformances.append(
        {
            "dataset": dataset,
            "acc": acc,
            "nil_acc": nil_acc,
            "correct_nil": correct_nil,
            "wrong_nil": wrong_nil,
        }
    )
perf_df = pd.DataFrame(preformances)
perf_df.to_csv("./results/baseWNUM.csv")
