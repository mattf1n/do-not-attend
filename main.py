from utils import *


def main():
    # print("Hello from do-not-attend!")
    import torch

    (data, model, tokenizer) = get_data_model()

    example = open("sample.txt").read()

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions




if __name__ == "__main__":
    main()



