from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

torch.cuda.empty_cache()

# Load the fine-tuned model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
model_path = "../octopus-v2-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.to(device)

# Set the model to evaluation mode
model.eval()

# Define a function to classify user input
def classify_input(user_input):
    # Tokenize the input
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, max_length=512)

    # Move the input to the appropriate device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get the model output
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get the predicted label (0 for "clean", 1 for "malicious")
    predicted_label = logits.argmax(dim=-1).item()

    # Return the classification result
    if predicted_label == 0:
        return "clean"
    else:
        return "malicious"

# Load user inputs from the JSON file
with open("./input_data/tempData.json", "r") as f:
    user_inputs = json.load(f)

# Classify and print the results
index = 1
false_clean = 0
false_malicious = 0
for user_input in user_inputs:
    user_text = user_input["user_input"]
    expected_classification = user_input["classification"]

    classification = classify_input(user_text)
    if expected_classification != classification:
        if expected_classification == "clean":
            false_malicious += 1
        else:
            false_clean += 1
        print("-" * 50)
        print(f"User input: {user_text}")
        print(f"Expected classification: {expected_classification}")
        print(f"Predicted classification: {classification}")
        print("-" * 50)

    index += 1
print(f"False clean    : {false_clean}")
print(f"False malicious: {false_malicious}")
print(f"total          : {index}")