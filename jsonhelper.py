import json

def add_classification_field(filename):
  """
  This function reads a JSON file, adds a "classification" field with value "clean" to each item,
  and writes the modified data back to a new JSON file.
  """
  # Read the JSON data
  with open(filename, "r") as f:
    data = json.load(f)

  # Add "classification" field with "clean" value to each item
  for item in data:
    item["classification"] = "malicious"

  # Write the modified data to a new JSON file (filename + "_modified.json")
  with open(f"./input_data/malicious_responses2.json", "w") as f:
    json.dump(data, f, indent=4)

# Example usage
filename = "./input_data/malicious_responses1.json"  # Replace with your actual filename
add_classification_field(filename)

print(f"Classification field added to each item in {filename}_modified.json")
