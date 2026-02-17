from datasets import load_dataset

ds = load_dataset("monology/pile-uncopyrighted", streaming = True)

print(dir(ds))