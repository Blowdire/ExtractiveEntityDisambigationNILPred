import json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import argparse
from tqdm import tqdm


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
        modified_answer = answer.split(": instance of ")[0]
        modified_answer = answer.split(": instance")[0]
        modified_answer = modified_answer.replace("<s>", "")
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
            modified_answer = "NIL"
        if modified_answer == "":
            modified_answer = "Not In Candidates"
        if modified_answer not in candidates:
            modified_answer = "Not In Candidates"
        return modified_answer


import gc
import torch.nn.functional as F


def make_prediction(data_entry, nil_prediction):
    with torch.no_grad():
        question = data_entry["input"]
        context = ""
        context += "Not In Candidates </ec>"
        for item in data_entry["candidates"]:
            context += item + f" </ec> "

        input_pairs = [question, context]
        # get sentence and candidates encodings
        encodings = tokenizer.encode_plus(
            input_pairs, return_tensors="pt", truncation="only_second"
        ).to("cuda")

        # predict candidate start and end indexes
        start_scores, end_scores = model(
            encodings["input_ids"], attention_mask=encodings["attention_mask"]
        )
        start_scores = F.softmax(start_scores, dim=1)
        end_scores = F.softmax(end_scores, dim=1)

        start = torch.argmax(
            start_scores
        )  # Get the most likely beginning of answer with the argmax of the score
        end = (
            torch.argmax(end_scores) + 1
        )  # Get the most likely end of answer with the argmax of the score

        # try to calculate the mean score of the start and end index
        mean_score = 0
        try:
            mean_score = end_scores[0][end.item()] + start_scores[0][start.item()]
        except:
            None
        answer_tokens = encodings["input_ids"][0, start.item() : end.item() + 1]
        answer = ""
        # clear decoded answer of special tokens
        answer = process_answer(tokenizer.decode(answer_tokens), context)

        # do nil prediction if needed
        if (mean_score < 0.9) and nil_prediction:
            answer = "Not In Candidates"
        else:
            if answer == "":
                answer = "Not In Candidates"
        # clear memory
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


def load_model(enriched):
    modelFolder = (
        "./robertaLarge/checkpoint-2625/"
        if not enriched
        else "./robertaLargeInstanceOf/checkpoint-2625/"
    )
    print("Loading model from", modelFolder)
    tokenizer = AutoTokenizer.from_pretrained(modelFolder)
    model = AutoModelForQuestionAnswering.from_pretrained(
        modelFolder,
        return_dict=False,
    ).to("cuda")
    return tokenizer, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optional app description")
    parser.add_argument("--enriched", action="store_true", help="Enriched dataset")

    parser.add_argument("dataset_path", type=str, help="Dataset to evaluate")
    parser.add_argument(
        "--nil",
        action="store_true",
        help="Enriched dataset",
    )

    args = parser.parse_args()
    enriched = args.enriched
    dataset_path = args.dataset_path
    nil = args.nil
    print(enriched)

    tokenizer, model = load_model(enriched)

    results = []
    dataset = []
    try:
        with open(dataset_path, "r") as file:
            for line in file:
                # Load each line as a JSON object
                data_line = json.loads(line)

                dataset.append(data_line)

        for item in tqdm(dataset):
            pred = make_prediction(item, False)
            results.append(pred)

        correct = 0
        correct_trace = []
        wrong = []
        correct_nil = 0
        wrong_nil = 0
        full_data = []
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
            full_data.append(
                {
                    "score": result["scores"],
                    "nil": int(result["correct"][0]["answer"] == "Not In Candidates"),
                }
            )
        accuracy = correct / len(results)
        if correct_nil + wrong_nil != 0:
            nil_accuracy = correct_nil / (correct_nil + wrong_nil)
        else:
            nil_accuracy = 0
        # write to file the accuracy
        with open(f"results-{dataset_path.split('.')[0]}.txt", "a") as file:
            file.write(f"Accuracy: {accuracy} \n")
            file.write(f"Nil accuracy: {nil_accuracy} \n")
    except Exception as e:
        print("Error while processing dataset", e)
