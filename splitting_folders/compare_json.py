import json

# Load the three JSON files
with open('unlabeled-5000-depth/mapping_test.json') as f1, open('unlabeled-5000-depth/mapping_test.json') as f2, open('segmented_distance_depth/mapping_test.json') as f3:
    data1 = json.load(f1)
    data2 = json.load(f2)
    data3 = json.load(f3)

# Get the sets of keys from each file
keys1 = set(data1.keys())
keys2 = set(data2.keys())
keys3 = set(data3.keys())

keys1 = { '_'.join(key.split('_')[:2]) for key in keys1 }
keys2 = { '_'.join(key.split('_')[:2]) for key in keys2 }
keys3 = { '_'.join(key.split('_')[:2]) for key in keys3 }


# Compare keys
common = keys1 & keys3
only_in_1 = keys1 - (keys2 | keys3)
only_in_2 = keys2 - (keys1 | keys3)
only_in_3 = keys3 - (keys1 | keys2)

# Print results
print("Keys in all 2 files:", sorted(common))
print("Keys only in file1:", sorted(only_in_1))
print("Keys only in file2:", sorted(only_in_2))
print("Keys only in file3:", sorted(only_in_3))


####################################################
#Create mapping file


# Load the files
with open("unlabeled-5000-depth/mapping_all_plants.json", "r") as f:
    all_data = json.load(f)

with open("segmented_distance_depth/mapping_test.json", "r") as f:
    mapping_test = json.load(f)

# Extract genotype IDs from mapping_test
mapping_genotypes = {v["genotype_id"] for v in mapping_test.values()}

# Prepare output splits
test_split = {}
rest_split = {}

# Split based on genotype_id
for key, entry in all_data.items():
    genotype = entry.get("genotype_id")
    if genotype in mapping_genotypes:
        test_split[key] = entry
    else:
        rest_split[key] = entry

# Save the results
with open("unlabeled-5000-depth/mapping_test.json", "w") as f:
    json.dump(test_split, f, indent=2)

with open("unlabeled-5000-depth/mapping_train.json", "w") as f:
    json.dump(rest_split, f, indent=2)


print(f"Entries in test_split.json: {len(test_split)}")
print(f"Entries in rest_split.json: {len(rest_split)}")