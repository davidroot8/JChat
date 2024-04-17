import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import pandas as pd

def get_batch_size(model_name: str, dataset, max_batch_size: int = None, num_iterations: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.train()

    batch_size = 2
    dataset_size = len(dataset)

    while True:
        if max_batch_size is not None and batch_size >= max_batch_size:
            batch_size = max_batch_size
            break

        if batch_size >= dataset_size:
            batch_size = dataset_size // 2
            break

        try:
            for _ in range(num_iterations):
                inputs = tokenizer(dataset[:batch_size]["user_input"], return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
                labels = torch.tensor([0 if sample["classification"] == "clean" else 1 for sample in dataset[:batch_size]]).to(device)
                outputs = model(**inputs, labels=labels)[1]
                loss = outputs.loss
                loss.backward()
                model.zero_grad()

            batch_size *= 2

        except RuntimeError:
            batch_size //= 2
            break

    del model, tokenizer
    torch.cuda.empty_cache()
    return batch_size

with open("./input_data/combinedReduced.json", "r") as f:
    data = json.load(f)

x = get_batch_size("../Octopus-v2", data, max_batch_size=16, num_iterations=5)
print(x)