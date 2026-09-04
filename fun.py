# %%
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from pprint import pprint
import regex as re
from collections import defaultdict

# %%
training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))

# %%
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]))

# %%
train_dataloaderr = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

for X, y in test_dataloader:
    print(f"X={X.shape}")
    print(f"y={y.shape}, {y.dtype}")
    break

# %%
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.core = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.core(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

#%%
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-3)

#%%
def train(dataloader, model, optimizer, loss_fn):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            print(f"batch={batch} loss={loss.item():>7f} [{(batch+1)*len(X):>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss, accuracy = 0., 0.
    with torch.no_grad():
        for X,y in dataloader:
            X,y = X.to(device), y.to(device)
            pred = model(X)
            loss += loss_fn(pred, y).item()
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
    loss /= num_batches
    accuracy /= size
    print(f"Avg loss = {loss:>7f} accuracy={100*accuracy:>0.1f}")

#%%
epochs = 5
for e in range(epochs):
    train(train_dataloaderr, model, optimizer, loss_fn)
    test(test_dataloader, model, loss_fn)

#%%
classes = test_data.classes
idx = 7
x, y = test_data[idx][0], test_data[idx][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred.argmax()], classes[y]
print(f"predicted={predicted} actual={actual} preds={pred}")

#%%
ord('牛')
chr(0)

#%%
s = "hello! こんにちは!"
print(s)
s_enc = s.encode("utf-8")
print(s_enc)
print(type(s_enc))
print(list(s_enc))
print(s_enc.decode("utf-8"))

#%%
import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
re.findall(PAT, "some text I'll pre-tokenize")

#%%
from collections import defaultdict
text= """low low low low low lower lower widest widest widest newest newest newest newest newest newest"""
freqmap = defaultdict(int)
for w in text.split(" "):
    freqmap[w] += 1
print(freqmap)
freqmap = {tuple(bytes([b]) for b in key.encode("utf-8")): count for key, count in freqmap.items()}
print(f"freqmap={freqmap}")
vocab = [chr(i) for i in range(256)] + [ "<|endoftext|>"]

#%%
num_merges = 5
for iter in range(num_merges):
    p = defaultdict(int)
    for word, count in freqmap.items():
        for a,b in zip(word[:-1], word[1:]):
            p[a+b] += count

    max_value = max(p.values())
    max_key = max([key for key, val in p.items() if val == max_value])
    print(f"iter={iter} max_key={max_key}")
    vocab.append(max_key.decode("utf-8"))

    # merge most frequent pair into a new token
    freqmap_ = defaultdict(int)
    for word, count in freqmap.items():
        i = 0
        new_word = []
        while i < len(word):
            if (i < len(word)-1) and (word[i]+word[i+1] == max_key):
                new_word.append(max_key)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        freqmap_.update({tuple(new_word): count})
    freqmap = freqmap_
    pprint(f"iter={iter} new freqmap={freqmap}")
print(f"new vocal={vocab[-num_merges:]}")

#%%
re.split("|".join([" ", re.escape("|")]), "a|b c")
text =  "a|b c doom|haha "
prev_idx = 0
for m in re.finditer("|".join([" ", re.escape("|")]), text):
    print(text[prev_idx: m.start()])
    prev_idx = m.end()
if prev_idx < len(text):
    print(text[prev_idx:])

#%%
def train_bpe(text, special_characters = ["<|endoftext|>"]):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    vocab = [bytes([i]) for i in range(256)] + ["<|endoftext|>".encode("utf-8")]
    chunks = re.split("|".join([re.escape(spec) for spec in special_characters]), text)
    print(f"chunks={chunks}")
    wordmap = defaultdict(int)
    for chunk in chunks:
        for m in re.finditer(PAT, chunk):
            wordmap[m.group()] += 1
    print(wordmap)
    wordmap = {tuple(bytes([b]) for b in key.encode("utf-8")): count for key, count in wordmap.items()}
    pprint(wordmap)

    num_merges = 10
    for iter in range(num_merges):
        p = defaultdict(int)
        for word, count in wordmap.items():
            for a,b in zip(word[:-1], word[1:]):
                p[(a,b)] += count

        max_value = max(p.values())
        max_key = max([key for key, val in p.items() if val == max_value])
        print(f"iter={iter} max_key={max_key}")
        vocab.append((max_key[0]+max_key[1]).decode("utf-8"))

        # merge most frequent pair into a new token
        new_wordmap = defaultdict(int)
        for word, count in wordmap.items():
            i = 0
            new_word = []
            while i < len(word):
                if (i < len(word)-1) and ((word[i],word[i+1]) == max_key):
                    new_word.append(max_key[0]+max_key[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_wordmap.update({tuple(new_word): count})
        wordmap = new_wordmap
        pprint(f"iter={iter} new wordmap={wordmap}")

text= """low low low low low lower lower widest widest widest newest<|endoftext|> newest newest newest newest newest"""
train_bpe(text)

#%%
with open("tests/fixtures/tinystories_sample.txt", "r", encoding="utf-8") as file:
    content = file.read()
train_bpe(content)

#%%
print(chr(178))
#print(bytes([178]).decode("utf-8"))
print(chr(178).encode("utf-8"))
