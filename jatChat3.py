import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import json
from sklearn.model_selection import train_test_split
import torch
torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
import gc
gc.collect()
# Load the JSON data

BATCH_SIZE = 6
MODEL_PATH = "../Octopus-v2"

# Load the JSON data
with open("./input_data/combinedReduced.json", "r") as f:
    data = json.load(f)

# Convert data to a pandas DataFrame
df = pd.DataFrame(data)

# Create a dataset from the DataFrame
train_dataset, eval_dataset = train_test_split(df, test_size=0.2, random_state=42)
dataset = Dataset.from_pandas(df)
train_dataset = Dataset.from_pandas(train_dataset)
eval_dataset = Dataset.from_pandas(eval_dataset)

dataset = DatasetDict({"train": train_dataset, "eval": eval_dataset})

# Load the tokenizer
MODEL_PATH = "../Octopus-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)

# Tokenize the user_input and classification columns
tokenized_dataset = dataset.map(lambda x: tokenizer(x["user_input"], truncation=True, padding=True, max_length=512, return_tensoractivis="pt"), batched=True)
#tokenized_dataset = tokenized_dataset.rename_column("input_ids", "text")
tokenized_dataset = tokenized_dataset.map(lambda x: {"labels": [1 if x["classification"] == "malicious" else 0]})


# Define the training arguments
training_args = TrainingArguments(
    output_dir=MODEL_PATH+"-finetuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["eval"],
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model(MODEL_PATH+"-finetuned")