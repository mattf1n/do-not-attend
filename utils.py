def load_json(json_path):
    """
    Loads a JSON file and returns its contents as a Python dictionary.

    Args:
        json_path (str): The path to the JSON file.

    Returns:
        dict: The contents of the JSON file as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
    """
    import json
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_occurrences(json_path: str) -> None:
    """
    Loads a 2-token max-word output JSON and prints each word's occurrence
    count and token indices.

    Args:
        json_path (str): Path to the output JSON file.
    """
    import json

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    main_data = data.get("main_data", {})

    total = 0
    for word, info in main_data.items():
        occurrences = info.get("occurrences", [])
        total += len(occurrences)
        print(f"{word}: {len(occurrences)} occurrence(s)")
        for i, occ in enumerate(occurrences):
            token_indices = occ.get("token_indices", [])
            print(f"  [{i}] token_indices={token_indices}")

    print(f"\nTotal occurrences: {total}")