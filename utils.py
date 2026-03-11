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