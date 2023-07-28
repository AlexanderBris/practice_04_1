# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 18:21:01 2023

@author: Alexander
"""

# подгрузка стандартных библиотек

import torch
import torch.nn as nn
import numpy as np


import pprint

text = ['hey how are you','good i am fine','have a nice day']

# Join all the sentences together and extract the unique characters 
# from the combined sentences
chars = set(''.join(text))
# Creating a dictionary that maps integers to the characters
int2char = dict(enumerate(chars))
# Creating another dictionary that maps characters to integers
char2int = {char: ind for ind, char in int2char.items()}

print("Dictionary for mapping character to the integer:")
pprint.pprint(char2int)

# выравнивание данных
#------------------------------------------------------------------
lengths = [len(sent) for sent in text]
maxlen = max(lengths)
print(f"The longest string has {maxlen} characters.\n")

print(f"Initial texts:\n{text}")
# A simple loop that loops through the list of sentences and adds
# a ' ' whitespace until the length of the sentence matches
# the length of the longest sentence
for i in range(len(text)):
    while len(text[i])<maxlen:
        text[i] += ' '

print(f'Resulting texts:\n{text}')

#------------------------------------------------------------------
input_seq = []
target_seq = []

for i in range(len(text)):
    # Remove last character for input sequence
    input_seq.append(text[i][:-1])
    
    # Remove first character for target sequence
    target_seq.append(text[i][1:])
    
    print("Input sequence:".ljust(18), f"'{input_seq[i]}'")
    print("Target sequence:".ljust(18), f"'{target_seq[i]}'")
    print()

#------------------------------------------------------------------
for i in range(len(text)):
    input_seq[i] = [char2int[character] for character in input_seq[i]]
    target_seq[i] = [char2int[character] for character in target_seq[i]]

    print("Encodded input sequence:".ljust(25), input_seq[i])
    print("Encodded target sequence:".ljust(25), target_seq[i])
    print()




#------------------------------------------------------------------
dict_size = len(char2int)
seq_len = maxlen - 1
batch_size = len(text)

def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float32)

    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features

input_seq = one_hot_encode(input_seq, dict_size, seq_len, batch_size)
print("Input shape: {} --> (Batch Size, Sequence Length, One-Hot Encoding Size)".format(input_seq.shape))
print(input_seq[0])       #First of three sentences

# создаем и тренеруем модель
#------------------------------------------------------------------
input_seq = torch.Tensor(input_seq)
target_seq = torch.Tensor(target_seq)

class NextCharacterGeneratorWithLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super().__init__()

        # RNN Layer
        self.lstm = nn.LSTM(input_size, hidden_size=hidden_dim, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        #Initializing hidden state for first input using method defined below
        hidden_0 = torch.zeros(1, batch_size, self.lstm.hidden_size) # 1 correspond to number of layers
        cells_0 = torch.zeros(1, batch_size, self.lstm.hidden_size) # 1 correspond to number of layers

        # Passing in the input and hidden state into the model and obtaining outputs
        out, (hidden, c) = self.lstm(x, (hidden_0, cells_0))

        # Reshaping the outputs such that it can be fit into the fully connected layer
        # Need Only if n_layers > 1
        out = out.contiguous().view(-1, self.lstm.hidden_size)
        out = self.fc(out)

        return out, (hidden, c)

model = NextCharacterGeneratorWithLSTM(input_size=dict_size, output_size=dict_size, hidden_dim=12, n_layers=1)

# Define hyperparameters
num_epochs = 500

# Define Loss, Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training Run
for epoch in range(1, num_epochs + 1):
    optimizer.zero_grad() # Clears existing gradients from previous epoch
    output, (hidden, c) = model(input_seq)
    loss = criterion(output, target_seq.view(-1).long())
    loss.backward() # Does backpropagation and calculates gradients
    optimizer.step() # Updates the weights accordingly

    if epoch % 10 == 0:
        print(f'Epoch: {epoch}/{num_epochs}'.ljust(20), end=' ')
        print("Loss: {:.4f}".format(loss.item()))

def predict(model, character):
    # One-hot encoding our input to fit into the model
    character = np.array([[char2int[c] for c in character]])
    character = one_hot_encode(character, dict_size, character.shape[1], 1)
    character = torch.from_numpy(character)

    out, hidden = model(character)
    #print(out.shape)
    #print(out)
    prob = nn.functional.softmax(out[-1], dim=0).data
    # Taking the class with the highest probability score from the output
    char_ind = torch.max(prob, dim=0)[1].item()

    return int2char[char_ind], hidden

def sample(model, out_len, start='hey'):
    model.eval() # eval mode
    start = start.lower()
    # First off, run through the starting characters
    chars = [ch for ch in start]
    size = out_len - len(chars)
    # Now pass in the previous characters and get a new one
    for _ in range(size):
        char, h = predict(model, chars)
        chars.append(char)

    return ''.join(chars)

sample(model, 15, 'i am')

# проверка
#------------------------------------------------------------------
for _ in range (3):
  print(sample(model, 15, 'good'))