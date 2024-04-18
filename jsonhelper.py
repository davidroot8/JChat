import json
import re

# Open the input JSON file
with open('input_data/no_robots.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Create a new list to store the modified data
output_data = []

# Modify the data
for item in data:
    user_input = item['user_input']
    classification = item['classification']
    prompt_id = item['prompt_id']
    messages = item['messages']
    category = item['category']

    # Extract the user input and assistant response from messages
    user_input = messages[0]['content']
    assistant_response = messages[1]['content']

    # Check for Unicode characters and skip if found
    if not re.search(r'[\u0080-\uffff]', user_input) and not re.search(r'[\u0080-\uffff]', assistant_response):
        # Create a new dictionary with the modified format
        new_item = {
            'prompt_id': prompt_id,
            'user_input': user_input,
            'assistant_response': assistant_response,
            'classification': classification
        }

        output_data.append(new_item)

# Write the modified data to a new JSON file
with open('input_data/no_robots1.json', 'w') as file:
    json.dump(output_data, file, indent=2)