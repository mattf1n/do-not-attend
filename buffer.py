import json
import glob

json_files = sorted(glob.glob('output/16000_tokens/*.json'))

if not json_files:
    print("No JSON files found in output/16000_tokens/")
    exit(1)

print("\n=== Available JSON files ===")
for i, f in enumerate(json_files, 1):
    print(f"{i}. {f}")

while True:
    try:
        file_idx = int(input("\nSelect file number: ")) - 1
        if 0 <= file_idx < len(json_files):
            break
        print("Invalid selection")
    except ValueError:
        print("Enter a number")

with open(json_files[file_idx]) as f:
    data = json.load(f)

keys = list(data.keys())
print(f"\n=== Top-level keys in {json_files[file_idx]} ===")
for i, k in enumerate(keys, 1):
    print(f"{i}. {k}")

while True:
    try:
        key_idx = int(input("\nSelect key number: ")) - 1
        if 0 <= key_idx < len(keys):
            break
        print("Invalid selection")
    except ValueError:
        print("Enter a number")

selected_key = keys[key_idx]
value = data[selected_key]

print(f"\n=== Value for '{selected_key}' ===")

if selected_key == 'main_data':
    words = list(value.keys())
    print(f"main_data has {len(words)} words")
    if words:
        first_word = words[0]
        print(f"\nFirst entry ('{first_word}'):")
        print(json.dumps({first_word: value[first_word]}, indent=2))
elif isinstance(value, (dict, list)):
    print(json.dumps(value, indent=2))
else:
    print(value)
