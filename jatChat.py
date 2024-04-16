import json
import ollama
import random
from typing import Dict, List
from pathlib import Path
from transformers import LlamaForSequenceClassification, LlamaTokenizer, TrainingArguments, Trainer

# Load the data
malicious_data = [json.loads(line) for line in Path("./input_data/malicious_responses.jsonl").read_text().splitlines()]
clean_data = [json.loads(line) for line in Path("./input_data/clean_responses.jsonl").read_text().splitlines()]

# Combine the data and add labels
data = [(example["user_input"], "malicious") for example in malicious_data] + [(example["user_input"], "clean") for example in clean_data]

# Split the data into train and test sets
random.shuffle(data)
train_data = data[:int(0.8 * len(data))]
test_data = data[int(0.8 * len(data)):]

# Load the tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained("../Llama-2-7b/tokenizer.model")
model = LlamaForSequenceClassification.from_pretrained("../Llama-2-7b/consolidated.00.pth", num_labels=2)

# Tokenize the data
train_encodings = tokenizer([(example[0], example[1]) for example in train_data], truncation=True, padding=True)
test_encodings = tokenizer([(example[0], example[1]) for example in test_data], truncation=True, padding=True)

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
while 1 == 1:
    user_input = input("Enter some text to classify (or 'q' to quit): ")
    if user_input.lower() == 'q':
        break

    encoding = tokenizer(user_input, return_tensors="pt")
    output = trainer.model(**encoding)
    logits = output.logits
    predicted_class = logits.argmax().item()
    print(f"Predicted class: {'malicious' if predicted_class == 0 else 'clean'}")