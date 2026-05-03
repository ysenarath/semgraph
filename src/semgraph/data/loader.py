import penman

from semgraph.data.preprocess import preprocess


def load_dataset(name: str) -> dict[str, list[penman.Graph]]:
    splits = preprocess(name)
    return {split: penman.load(path) for split, path in splits.items()}


if __name__ == "__main__":
    dataset = load_dataset("ldc2014t12")
    inputs = []
    for i in range(10):
        item = dataset["test"][i]
        snt = item.metadata["snt"]
        inputs.append(snt)
    print(inputs)
