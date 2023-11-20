"""
This script will change the image_paths in the json data that will be used to fine tune the model.
"""
import json
# ead json file
with open('/workspace/ilknur/sample_data/data.json', 'r') as file:
    data = json.load(file)
# loop for per item in json
for item in data:
    # change path
    item['image'] = item['image'].replace("/content/drive/MyDrive/sample_data_llava", "/workspace/ilknur/sample_data/data/sample_data_llava")
# Save changes
with open('/workspace/ilknur/sample_data/data2.json', 'w') as file:
    json.dump(data, file, indent=2)
