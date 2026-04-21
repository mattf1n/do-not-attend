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


def save_output_npz(input_json: str, npz_meta_path: str) -> None:
    """
    Loads a JSON output file and saves it as binary npz + metadata JSON inside
    a folder created at npz_meta_path. Files are named after the input JSON stem.

      npz_meta_path/
          {stem}.npz        : attention arrays as float32, shape (num_layers, num_heads, num_subtokens)
          {stem}_meta.json  : text, choice, num_paragraphs, word keys, token_indices

    Args:
        input_json (str): Path to the source JSON file, e.g. "output/my_results.json".
        npz_meta_path (str): Path to the folder to create, e.g. "output/binary/".
    """
    import json
    import numpy as np
    from pathlib import Path

    stem = Path(input_json).stem
    folder = Path(npz_meta_path) / stem
    folder.mkdir(parents=True, exist_ok=True)

    with open(input_json, encoding="utf-8") as f:
        out = json.load(f)

    metadata = {
        'text': out['text'],
        'choice': out.get('choice'),
        'num_paragraphs': out.get('num_paragraphs'),
        'index': {},
    }
    arrays = {}

    for word, word_data in out['main_data'].items():
        metadata['index'][word] = []
        for i, occ in enumerate(word_data['occurrences']):
            arr = np.array(
                [layer['heads'] for layer in occ['attentions']['layers']],
                dtype=np.float32
            )
            arrays[f"{word}__occ{i}"] = arr
            metadata['index'][word].append({'token_indices': occ['token_indices']})

    np.savez_compressed(folder / f"{stem}.npz", **arrays)
    with open(folder / f"{stem}_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)


def load_output_npz(npz_meta_path: str) -> dict:
    """
    Loads data from a folder created by save_output_npz and reconstructs the
    original output dict. Attentions per occurrence are numpy float32 arrays
    of shape (num_layers, num_heads, num_subtokens).

    Args:
        npz_meta_path (str): Path to the folder containing the .npz and _meta.json files.

    Returns:
        dict: Reconstructed output dict with keys 'text', 'choice', 'num_paragraphs', 'main_data'.
    """
    import json
    import numpy as np
    from pathlib import Path

    folder = Path(npz_meta_path)
    npz_files = list(folder.glob("*.npz"))
    meta_files = list(folder.glob("*_meta.json"))

    if not npz_files or not meta_files:
        raise FileNotFoundError(f"Could not find .npz and _meta.json files in {npz_meta_path}")

    npz = np.load(npz_files[0])
    with open(meta_files[0], encoding="utf-8") as f:
        metadata = json.load(f)

    main_data = {}
    for word, occurrences in metadata['index'].items():
        main_data[word] = {'occurrences': []}
        for i, occ_meta in enumerate(occurrences):
            arr = npz[f"{word}__occ{i}"]
            main_data[word]['occurrences'].append({
                'token_indices': occ_meta['token_indices'],
                'attentions': arr,
            })

    return {
        'text': metadata['text'],
        'choice': metadata.get('choice'),
        'num_paragraphs': metadata.get('num_paragraphs'),
        'main_data': main_data,
    }


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


def print_summary(data: dict) -> None:
    """
    Prints a high-level summary of an output dict.

    Args:
        data (dict): Output dict with 'main_data' and 'num_paragraphs' keys.
    """
    main_data = data.get("main_data", {})
    num_words = len(data.get("text", "").split())
    num_multitoken_words = len(main_data)
    num_paragraphs = data.get("num_paragraphs")
    num_occurrences = sum(len(v.get("occurrences", [])) for v in main_data.values())

    print(f"num_words:            {num_words}")
    print(f"num_multitoken_words: {num_multitoken_words}")
    print(f"num_paragraphs:       {num_paragraphs}")
    print(f"num_occurrences:      {num_occurrences}")


def print_json_info(data) -> None:
    """
    Prints a formatted summary of an output dict or a path to one.

    Args:
        data (dict | str): Output dict with 'text', 'num_paragraphs', and 'main_data'
                           keys, or a path to a JSON file.
    """
    import json
    if isinstance(data, str):
        with open(data, encoding="utf-8") as f:
            data = json.load(f)
    main_data = data.get("main_data", {})

    num_words = len(data.get("text", "").split())
    num_multitoken_words = len(main_data)
    num_paragraphs = data.get("num_paragraphs")
    num_occurrences = sum(len(v.get("occurrences", [])) for v in main_data.values())

    print(f"num_words:            {num_words}")
    print(f"num_multitoken_words: {num_multitoken_words}")
    print(f"num_paragraphs:       {num_paragraphs}")
    print(f"num_occurrences:      {num_occurrences}")

    print(f"\nbi_words: {main_data.keys()}")


def print_occurrences_dict(data: dict) -> None:
    """
    Prints each word's occurrence count and token indices from an already-loaded
    output dict.

    Args:
        data (dict): Output dict with a 'main_data' key, as returned by load_json
                     or load_output_npz.
    """
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