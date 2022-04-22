from abc import ABC
from Utils.array import printArray
import torch
import torch.nn as nn

emojies = ['🐁', '👒', '🐈️', '🦇', '🏡', '🤵🏻', '🥚', '🥩', '⚽']
words = ['rat ', 'hat ', 'cat ', 'bat ', 'flat', 'matt', 'egg ', 'meat', 'ball']
chars = [c for c in ''.join(set(''.join(words)))]  # array of all unique chars in words

EPOCHS = 26
LEARNING_RATE = 0.001

testWords = ['rt', 'ht', 'c', 'bt', 'f', 'tt', 'g', 'ea', 'l']
TEST_INTERVAL = 5


class LongShortTermMemoryModel(nn.Module, ABC):
    def __init__(self, in_size, out_size):
        super(LongShortTermMemoryModel, self).__init__()
        self.lstm = nn.LSTM(in_size, 128)
        self.dense = nn.Linear(128, out_size)

    def reset(self, batch_size):
        zero_state = torch.zeros(1, batch_size, 128)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out[-1].reshape(-1, 128))

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


# Turns array of letters/emojies into diagonal matrix of 0. and 1.
def encodeChars(arrayOfChars):
    return [[1. if i == j else 0. for j in range(len(arrayOfChars))] for i in range(len(arrayOfChars))]


# Turns the string into list of [char_encoding[x]]
def encodeWord(string):
    return [char_encodings[chars.index(letter)] for letter in string]


char_encodings = encodeChars(chars)
emoji_encodings = encodeChars(emojies)

x_train = torch.tensor([encodeWord(w) for w in words]).transpose(0, 1)
y_train = torch.tensor(emoji_encodings)

model = LongShortTermMemoryModel(len(chars), len(emojies))

# Variables for printing results later
results = []
resultHeaders = ["epoch:"]

# Optimize model
optimizer = torch.optim.RMSprop(model.parameters(), LEARNING_RATE)
for epoch in range(EPOCHS):
    model.reset(len(emojies))
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

    # Add current state of model to result list
    if epoch % TEST_INTERVAL == 0:
        resultHeaders.append(epoch)
        curResult = []
        # find emoji for all test-words
        for word in testWords:
            model.reset(1)
            x = torch.tensor([encodeWord(word)]).transpose(0, 1)
            index = model.f(x).argmax(1).numpy()[0]
            # add emoji to current result-row
            curResult.append(emojies[index])
        results.append(curResult)

# Print result
printArray(array=results, flip=True, spacing=6, columnHeaders=resultHeaders, rowHeaders=testWords)
