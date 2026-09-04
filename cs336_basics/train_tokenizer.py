from tests.adapters import run_train_bpe
import pickle
import os
import json
import specs

def save(root, vocab, merges):
    with open(f"{root}/vocab.txt", "w") as f:
        json.dump({i: w.decode("utf-8", errors="backslashreplace") for i, w in vocab.items()}, f, indent=2)
    with open(f"{root}/merges.txt", "w") as f:
        json.dump([(a.decode("utf-8", errors="backslashreplace"), b.decode("utf-8", errors="backslashreplace")) for a,b in merges], f)
    with open(f"{root}/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open(f"{root}/merges.pkl", "wb") as f:
        pickle.dump(merges, f)

def load(root):
    with open(f"{root}/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(f"{root}/merges.pkl", "rb") as f:
        merges = pickle.load(f)
    return vocab, merges

if __name__ == "__main__":
    root = specs.ROOT
    print(f"root={root}")

    #vocab, merges = run_train_bpe("tests/fixtures/tinystories_sample_5M.txt", 500, special_tokens=["<|endoftext>"])
    vocab, merges = run_train_bpe("/Users/dmaliuk/fall25/python/cs336/assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", 10000, special_tokens=["<|endoftext>"])

    save(root, vocab, merges)
    vocab_, merges_ = load(root)

    assert vocab == vocab_
    assert merges == merges_