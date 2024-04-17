import json
import random
from typing import List
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
print(torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
from transformers import DataCollatorWithPadding
import numpy

class CustomDataCollator(DataCollatorWithPadding):
  def __call__(self, features):
    batch = super().__call__(features)
    print("\n these are the features: ", features)
    labels = [example["label"] for example in features]  # Assuming "label" is the key in your features dict (if applicable)
    # If labels are not part of features, remove this line
    input_ids = [f.input_ids for f in features]  # Extract input IDs from each Encoding object
    batch["input_ids"] = torch.stack(input_ids, dim=0)  # Stack input_ids into a tensor
    if labels:  # Only add labels if they exist
      batch["labels"] = tokenizer(labels, return_tensors="pt", padding=True).input_ids
    return batch
  

max_input_length = 512
def preprocess_function(examples):
    model_inputs = AutoTokenizer(
        examples["user_input"],  # Assuming "user_input" is the text for classification
        max_length=max_input_length,
        truncation=True,
        padding="max_length",  # Explicitly set padding for consistent length
        return_tensors="pt"  # Convert to tensors
    )
    return model_inputs
# Load the data
malicious_data = json.loads(Path("./input_data/malicious_responses1.json").read_text())
clean_data = json.loads(Path("./input_data/clean_responses1.json").read_text())

# Combine the data and add labels
data = []
for example in malicious_data:
    data.append({"user_input": example["user_input"], "label": "malicious"})

for example in clean_data:
    data.append({"user_input": example["user_input"], "label": "clean"})

# Split the data into train and test sets
random.shuffle(data)
train_data, test_data = data[:int(0.8 * len(data))], data[int(0.8 * len(data)):]


train_encodings = preprocess_function(train_data)
test_encodings = preprocess_function(test_data)
# Load the tokenizer and model
model_id = "../Octopus-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

print(f"Length of training data: {len(train_data)}")
print(f"Length of test data: {len(test_data)}")



# Tokenize the data
train_encodings = {k: v.to(device) for k, v in train_encodings.items()}  # Assuming all encodings have same keys, convert to tensors on device
test_encodings = {k: v.to(device) for k, v in test_encodings.items()}
print(f"Length of training data: {len(train_encodings)}")
print(f"Length of test data: {len(test_encodings)}")

# Define training arguments (adjust as needed)
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
data_collator = CustomDataCollator(tokenizer=tokenizer)
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=train_encodings,
  eval_dataset=test_encodings,
  data_collator=data_collator,
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