import os
import json

# Define the root directory and the filename pattern
root_dir = '/home/msiau/data/tmp/ibeltran/sat/sen2venus'
file_suffix = '10m_b2b3b4b8.pt'

# List to hold file paths
file_paths = []

# Walk through the directory
for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.endswith(file_suffix):
            # Construct the full file path
            file_path = os.path.join(dirpath, filename)
            file_paths.append(file_path)

# Output JSON file path
output_json = 'sentinel2_images.json'

# Write the list of file paths to a JSON file
with open(output_json, 'w') as json_file:
    json.dump(file_paths, json_file, indent=4)

print(f"JSON file '{output_json}' created with {len(file_paths)} paths.")
