from utils import *
import json

def main():
    # print("Hello from do-not-attend!")
    import torch

    (data, model, tokenizer) = get_data_model()

    example = open("sample.txt").read()

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions



def test_pipeline(output_json_path="test_pipeline_results_all.json"):
    """End-to-end pipeline: load data/model, run inference, analyze attention.

    Returns:
        paragraph: the input text
        multi: dict of multi-token words with their tokens and positions
        results: per-word, per-subtoken attention analysis
        summary: condensed mapping of subtoken strings to max attention scores
    """

    # choice = input("1: paragraph \n2: one doc ")

    model, tokenizer = get_model()

    # if (choice == "1"):
    #     print("paragraph selected")
    #     text = get_first_paragraph()
    # elif (choice == "2"):
    print("one doc selected")
    text = get_data_sample()
    # else:
    #     print("default selected")
    #     text = get_first_paragraph()
    attentions = get_attentions(text, model, tokenizer)

    # Free GPU memory from the model before heavy analysis
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    multi = get_multi_token_words(text, tokenizer)
    scores = get_all_attention_results(attentions, multi, text, tokenizer)
    # summary = summarize_results_max(scores, tokenizer)
    # Convert everything to serializable form for JSON
    json_data = {
        "text": text,
        "multi": multi,
        "scores": scores,
        # "summary": summary
    }

    # Safe serialization: convert any numpy types or non-serializable elements
    def safedump(obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        return obj

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2, default=safedump)
    
    print("output saved")

    return (text, multi, scores)

if __name__ == "__main__":  

    # pick = input("1: Visualization \nelse: pipeline")

    # if pick == "1": 
    #     # plot_by_word_length_hist_summary()
    # else: 
    test_pipeline()
    #     text, multi, results, max_summary = test_pipeline()
    #     pprint(max_summary)

