import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNetwork
from preprocessing import tokenize, stem_word, create_bag_of_words

# Load intents data
with open("./training data/intents.json", "r") as f:
    intents_data = json.load(f)

# Initialize lists
all_words = []
tags = []
xy = []

# Prepare data
for intent in intents_data["intents"]:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        tokenized_pattern = tokenize(pattern)
        all_words.extend(tokenized_pattern)
        xy.append((tokenized_pattern, tag))

# Remove punctuation and stem words
ignore_words = ["?", "!", ".", ","]
all_words = [stem_word(word) for word in all_words if word not in ignore_words]
all_words = sorted(set(all_words))

# Prepare training data
train_x = []
train_y = []
for tokenized_sentence, tag in xy:
    bag = create_bag_of_words(tokenized_sentence, all_words)
    train_x.append(bag)
    label = tags.index(tag)
    train_y.append(label)

train_x = np.array(train_x)
train_y = np.array(train_y)

# Define custom dataset
class TextDataset(Dataset):
    def __init__(self):
        self.n_samples = len(train_x)
        self.x_data = train_x
        self.y_data = train_y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Hyperparameters
batch_size = 8
input_dim = len(train_x[0])
hidden_dim = 10
output_dim = len(tags)
learning_rate = 0.001
num_epochs = 1200

# Initialize dataset and data loader
dataset = TextDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = NeuralNetwork(input_dim, hidden_dim, output_dim).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Save model data
model_data = {
    "model_state": model.state_dict(),
    "input_dim": input_dim,
    "hidden_dim": hidden_dim,
    "output_dim": output_dim,
    "all_words": all_words,
    "tags": tags,
}

MODEL_FILE = "./model_file/model_data.pth"
torch.save(model_data, MODEL_FILE)

print(f"Training complete. Model data saved to {MODEL_FILE}")
