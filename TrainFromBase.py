import json
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import random

print("Loading longformer model")
file_path = "./train_converted.jsonl"
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
model = AutoModelForQuestionAnswering.from_pretrained(
    "allenai/longformer-base-4096", return_dict=False
)
print("Importing NIL EL Data")
tokenized = []
file_path = "./train_converted2.jsonl"
from datasets import Dataset

original_data = []
with open(file_path, "r") as file:
    for line in file:
        # Load each line as a JSON object
        data_line = json.loads(line)
        for training_item in data_line:
            original_data.append(training_item)


print("Importing Aida Data")
tokenized = []
tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")
file_path = "./aida_train_converted.jsonl"
from datasets import Dataset

original_data_aida = []
with open(file_path, "r") as file:
    for line in file:
        # Load each line as a JSON object
        data_line = json.loads(line)
        for training_item in data_line:
            original_data_aida.append(training_item)


print("Creating Huggingface dataset object")

concat = original_data_aida + original_data
random.shuffle(concat)
train_original = concat[: int(len(concat) * 0.7)]
eval_original = concat[int(len(concat) * 0.7) :]
dataset = Dataset.from_dict(
    {
        "context": [item["question"] for item in train_original],
        "question": [item["context"] for item in train_original],
        "answers": [item["answers"] for item in train_original],
    }
)
dataset_ev = Dataset.from_dict(
    {
        "context": [item["question"] for item in eval_original],
        "question": [item["context"] for item in eval_original],
        "answers": [item["answers"] for item in eval_original],
    }
)


def get_correct_alignement(context, answer):
    """Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here."""
    gold_text = answer["text"][0]
    start_idx = answer["answer_start"][0]
    end_idx = start_idx + len(gold_text)
    if context[start_idx:end_idx] == gold_text:
        return start_idx, end_idx  # When the gold label position is good
    elif context[start_idx - 1 : end_idx - 1] == gold_text:
        return start_idx - 1, end_idx - 1  # When the gold label is off by one character
    elif context[start_idx - 2 : end_idx - 2] == gold_text:
        return start_idx - 2, end_idx - 2  # When the gold label is off by two character
    else:
        raise ValueError()


# Tokenize our training dataset
def convert_to_features(example):
    try:
        # Tokenize contexts and questions (as pairs of inputs)
        input_pairs = [example["question"], example["context"]]
        encodings = tokenizer.encode_plus(
            input_pairs, pad_to_max_length=True, max_length=512
        )
        context_encodings = tokenizer.encode_plus(example["context"])

        # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methodes.
        # this will give us the position of answer span in the context text

        if example["answers"]["answer_start"][0] == -1:
            print("l'e' che")
            start_idx = example["context"].find("NIL")
            example["answers"]["answer_start"][0] = start_idx
            example["answers"]["text"][0] = "NIL"

        start_idx, end_idx = get_correct_alignement(
            example["context"], example["answers"]
        )
        start_positions_context = context_encodings.char_to_token(start_idx)
        end_positions_context = context_encodings.char_to_token(end_idx - 1)

        # here we will compute the start and end position of the answer in the whole example
        # as the example is encoded like this <s> question</s></s> context</s>
        # and we know the postion of the answer in the context
        # we can just find out the index of the sep token and then add that to position + 1 (+1 because there are two sep tokens)
        # this will give us the position of the answer span in whole example
        sep_idx = encodings["input_ids"].index(tokenizer.sep_token_id)
        start_positions = start_positions_context + sep_idx + 1
        end_positions = end_positions_context + sep_idx + 1

        encodings.update(
            {
                "start_positions": start_positions,
                "end_positions": end_positions,
                "attention_mask": encodings["attention_mask"],
            }
        )
        return encodings
    except:
        print("errors")


print("Processing datasets")

train_ds = dataset.map(convert_to_features)
eval_ds = dataset_ev.map(convert_to_features)

print("Training model")
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

args = TrainingArguments(
    "bert-finetuned-squad",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_eval_batch_size=8,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    num_train_epochs=2,
    weight_decay=0.01,
    fp16=True,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
)
trainer.train()

print("Saving model")

trainer.save_model("./TrainedModel")
