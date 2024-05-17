import random
import json
import torch
from model import NeuralNetwork
from preprocessing import create_bag_of_words, tokenize , compare_methods

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./training data/intents.json', 'r') as json_data:
    intents_data = json.load(json_data)

MODEL_FILE = "./model_file/model_data.pth"
model_data = torch.load(MODEL_FILE)

input_dim = model_data["input_dim"]
hidden_dim = model_data["hidden_dim"]
output_dim = model_data["output_dim"]
all_words = model_data['all_words']
tags = model_data['tags']
model_state = model_data["model_state"]

model = NeuralNetwork(input_dim, hidden_dim, output_dim).to(device)
model.load_state_dict(model_state) #Loading Pre-trained Weights
model.eval()

def chat_bot(input_text):
    tokenized_input = tokenize(input_text)
    compare_methods(input_text,tokenized_input)
    bag_of_words_input = create_bag_of_words(tokenized_input, all_words)
    bag_of_words_input = bag_of_words_input.reshape(1, bag_of_words_input.shape[0])
    bag_of_words_input = torch.from_numpy(bag_of_words_input).to(device)

    output = model(bag_of_words_input)
    _, predicted_idx = torch.max(output, dim=1)

    predicted_tag = tags[predicted_idx.item()]

    probabilities = torch.softmax(output, dim=1)
    probability = probabilities[0][predicted_idx.item()]

    if probability.item() > 0.80:
        for intent in intents_data['intents']:
            if predicted_tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "I'm sorry, I didn't quite understand that. Maybe I'm still learning. 😞"
