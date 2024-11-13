import json
import pandas as pd
URLS_JSON = 'images.json'

def combine_objects_to_dict(filename):
    # Load the JSON file
    with open(filename, 'r') as file:
        objects = json.load(file)
    
    combined_dict = {}
    errored = 0
    print(len(objects))
    # Iterate through each dictionary in the list
    for obj in objects:
        print(len(obj))
        for key in obj.keys():
            if key in combined_dict:
                errored+=1
                # print(f"Error: Duplicate key found: '{key}'")
            else:
                combined_dict[key] = obj[key]
    print(f"Errors: {errored}")
    return combined_dict

try:
    combined_dict = combine_objects_to_dict(URLS_JSON)

    with open('TCG-image-urls.csv', 'w') as file:
        file.write("id,url\n")
        for key in combined_dict.keys():
            file.write(f"{key},{combined_dict[key]}\n")
    # print("Combined dictionary:", combined_dict)
except Exception as e:
    print("An error occurred:", e)

df = pd.read_csv('TCG-image-urls.csv')
print(len(df)==len(df["id"].unique()))