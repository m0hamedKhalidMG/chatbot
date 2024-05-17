import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer



import spacy
from time import time

nlp = spacy.load('en_core_web_sm')



from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk import pos_tag, ne_chunk
from nltk.chunk import RegexpParser
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re

# Function to tokenize a sentence
def tokenize(sentence):
    #tag_pos(nltk.word_tokenize(sentence))
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




def filter_stop_words(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]


def tag_pos(tokens):
    print(pos_tag(tokens))
    return pos_tag(tokens)


def tokenize_spacy(sentence):
    doc = nlp(sentence)
    return [token.text for token in doc]

def stem_word_spacy(word):
    doc = nlp(word)
    return doc[0].lemma_

def create_bag_of_words_spacy(tokenized_sentence, vocabulary):
    stemmed_sentence = [stem_word_spacy(word) for word in tokenized_sentence]
    bag = np.zeros(len(vocabulary), dtype=np.float32)
    for idx, word in enumerate(vocabulary):
        if word in stemmed_sentence:
            bag[idx] = 1.0
    return bag

def compare_methods(sentence, vocabulary):
    # NLTK
    start_time_nltk = time()
    tokenized_nltk = tokenize(sentence)
    bag_nltk = create_bag_of_words(tokenized_nltk, vocabulary)
    end_time_nltk = time()
    nltk_time = end_time_nltk - start_time_nltk

    # SpaCy
    start_time_spacy = time()
    tokenized_spacy = tokenize_spacy(sentence)
    bag_spacy = create_bag_of_words_spacy(tokenized_spacy, vocabulary)
    end_time_spacy = time()
    spacy_time = end_time_spacy - start_time_spacy

 
    print(f"NLTK Time: {nltk_time:.6f} seconds")
    print(f"SpaCy Time: {spacy_time:.6f} seconds")

   
    print("NLTK Bag of Words:", bag_nltk)
    print("SpaCy Bag of Words:", bag_spacy)


sentence = "This is a simple example sentence for testing."
vocabulary = ["this", "is", "a", "simple", "example", "sentence", "for", "testing"]


















