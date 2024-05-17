import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

# Function to tokenize a sentence
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# Function to stem a word to its root form
def stem_word(word):
    porter_stemmer = PorterStemmer()
    return porter_stemmer.stem(word.lower())

# Function to create a bag of words array
def create_bag_of_words(tokenized_sentence, vocabulary):
    stemmed_sentence = [stem_word(word) for word in tokenized_sentence]
    bag = np.zeros(len(vocabulary), dtype=np.float32)
    for idx, word in enumerate(vocabulary):
        if word in stemmed_sentence:
            bag[idx] = 1.0
    return bag
