
# malicious_data = [json.loads(line) for line in Path("./input_data/malicious_responses.jsonl").read_text().splitlines()]
# clean_data = [json.loads(line) for line in Path("./input_data/clean_responses.jsonl").read_text().splitlines()]

# tokenizer = AutoTokenizer.from_pretrained("../Octopus-v2")
# model = GemmaForCausalLM.from_pretrained("../Octopus-v2", torch_dtype=torch.bfloat16, device_map="auto")
 
import json
import random
from typing import List
from pathlib import Path
from transformers import AutoTokenizer, GemmaForCausalLM, Trainer, TrainingArguments
import torch
print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load the data
malicious_data = [json.loads(line) for line in Path("./input_data/malicious_responses1.json").read_text().splitlines()]
clean_data = [json.loads(line) for line in Path("./input_data/clean_responses1.json").read_text().splitlines()]

# Combine the data and add labels
data = [(example["user_input"], "malicious") for example in malicious_data] + [(example["user_input"], "clean") for example in clean_data]

# Split the data into train and test sets
random.shuffle(data)
train_data = data[:int(0.8 * len(data))]
test_data = data[int(0.8 * len(data)):]

# Load the tokenizer and model
model_id = "../Octopus-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = GemmaForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")

# Tokenize the data
train_encodings = tokenizer([f"Classify the following text as clean or malicious: {text}" for text, _ in train_data], truncation="max_length", padding="max_length", return_tensors="pt")
test_encodings = tokenizer([f"Classify the following text as clean or malicious: {text}" for text, _ in test_data], truncation="max_length", padding="max_length", return_tensors="pt")

# Define the training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True,
)

# Create the trainer and fine-tune the model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_encodings,
    eval_dataset=test_encodings,
    tokenizer=tokenizer,
)

trainer.train()

# Evaluate the model on the test set
test_results = trainer.evaluate()
print(f"Test results: {test_results}")

# Use the model to classify new inputs
while True:
    user_input = input("Enter some text to classify (or 'q' to quit): ")
    if user_input.lower() == 'q':
        break

    input_text = f"Classify the following text as clean or malicious: {user_input}"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    output = model(input_ids)[0]
    predicted_class = torch.argmax(output, dim=-1).item()
    print(f"Predicted class: {'malicious' if predicted_class == 0 else 'clean'}")
    