import json
import os

from frozendict import frozendict

# Initialize a dictionary to store frozendicts
json_data_store = {}
repo_path = os.path.dirname(os.path.abspath(__file__))

for file_name in os.listdir(repo_path):
    if file_name.endswith(".json"):  # Process only JSON files
        file_path = os.path.join(repo_path, file_name)

        with open(file_path) as f:
            data = json.load(f)

        # Store the frozendict with the file name (without extension) as the key
        key = os.path.splitext(file_name)[0]
        json_data_store[key] = frozendict(data)

# Expose the json_data_store as a module-level variable
__all__ = ["json_data_store"]
