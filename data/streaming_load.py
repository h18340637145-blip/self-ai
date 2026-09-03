from datasets import load_dataset

# streaming loading dataset   # 使用20231101  之前的数据集没有了
dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

for i, example in enumerate(dataset):
    print(example["title"])
    if i >= 4:
        break