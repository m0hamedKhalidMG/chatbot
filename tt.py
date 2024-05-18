import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
import spacy
from spacy.matcher import Matcher
'''

CC	coordinating conjunction
CD	cardinal digit
DT	determiner
EX	existential there
FW	foreign word
IN	preposition/subordinating conjunction
JJ	This NLTK POS Tag is an adjective (large)
JJR	adjective, comparative (larger)
JJS	adjective, superlative (largest)
LS	list market
MD	modal (could, will)
NN	noun, singular (cat, tree)
NNS	noun plural (desks)
NNP	proper noun, singular (sarah)
NNPS	proper noun, plural (indians or americans)
PDT	predeterminer (all, both, half)
POS	possessive ending (parent\ ‘s)
PRP	personal pronoun (hers, herself, him, himself)
PRP$	possessive pronoun (her, his, mine, my, our )
RB	adverb (occasionally, swiftly)
RBR	adverb, comparative (greater)
RBS	adverb, superlative (biggest)
RP	particle (about)
TO	infinite marker (to)
UH	interjection (goodbye)
VB	verb (ask)
VBG	verb gerund (judging)
VBD	verb past tense (pleaded)
VBN	verb past participle (reunified)
VBP	verb, present tense not 3rd person singular(wrap)
VBZ	verb, present tense with 3rd person singular (bases)
WDT	wh-determiner (that, what)
WP	wh- pronoun (who)
WRB	wh- adverb (how)

'''
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')
nlp = spacy.load("en_core_web_sm")


text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, information engineering, and artificial intelligence 
concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and 
"""

# 1.
def remove_punctuation(text):
    return ''.join([char for char in text if char.isalnum() or char.isspace()])

# 2. 
def clean_and_tokenize(text):
    cleaned_text = remove_punctuation(text)
    sentences = sent_tokenize(cleaned_text)
    tokens = [word_tokenize(sentence) for sentence in sentences]
    return tokens

# 3. 
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [[word for word in sentence if word.lower() not in stop_words] for sentence in tokens]

# 4. Stemming
def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return [[stemmer.stem(word) for word in sentence] for sentence in tokens]

# 5. Tagging Parts of Speech
def pos_tagging(tokens):
    return [pos_tag(sentence) for sentence in tokens]

# 6. Lemmatizing
def lemmatize_tokens(tokens):
    lemmatizer = WordNetLemmatizer()
    return [[lemmatizer.lemmatize(word) for word in sentence] for sentence in tokens]

# 7. Chunking

# 6. Chunking
def chunking(tokens, grammar=r"NP: {<DT>?<JJ>*<NN>}"):
    chunk_parser = nltk.RegexpParser(grammar)
    return [chunk_parser.parse(sentence) for sentence in tokens]


def chunking(sentences, grammar=r"NP: {<DT>?<JJ>*<NN>}"):
    chunk_parser = nltk.RegexpParser(grammar)
    chunked_sentences = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        pos_tags = pos_tag(tokens)
        chunked_sentence = chunk_parser.parse(pos_tags)
        chunked_sentences.append(chunked_sentence)
    return chunked_sentences

# 8. Chinking

def chinking(chunked_sentences, grammar=r"""
    NP:
    {<.*>+}          # Chunk everything
    }<VBZ|VBD|VB|VBN|VBG|VBP>+{      # Chink sequences of verbs
"""):
    chunk_parser = nltk.RegexpParser(grammar)
    chinked_sentences = []
    for chunked_sentence in chunked_sentences:
        chinked_sentence = chunk_parser.parse(chunked_sentence)
        chinked_sentences.append(chinked_sentence)
    return chinked_sentences
# 9. Named-Entity Recognition
def named_entity_recognition(text):
    sentences = sent_tokenize(text)
    words = [word_tokenize(sentence) for sentence in sentences]
    tagged_words = [pos_tag(sentence) for sentence in words]
    return [ne_chunk(tagged_sentence) for tagged_sentence in tagged_words]

# 10. Dependency Parsing


def dependency_parsing(text):
    doc = nlp(text)
    return [(token.text, token.dep_, token.head.text) for token in doc]

# Apply dependency parsing
dependency_results = dependency_parsing(text)

# Print results
print("Dependency Parsing Results:")
for token, dep, head in dependency_results:
    print(f"Token: {token}, Dependency: {dep}, Head: {head}")

# Print results by sentence for better clarity
def print_dependency_by_sentence(text):
    doc = nlp(text)
    for sent in doc.sents:
        print(f"Sentence: {sent.text}")
        for token in sent:
            print(f"  Token: {token.text}, Dependency: {token.dep_}, Head: {token.head.text}")
        print()

print_dependency_by_sentence(text)

# 11. Rule-Based Matching
'''

amod (adjectival modifier): "Natural" modifies "language".
compound: "language" is part of the compound noun with "processing".
nsubj (nominal subject): "processing" is the subject of the verb "is".
ROOT: The main verb of the sentence.
det (determiner): "a" is a determiner for "subfield".
attr (attribute): "subfield" is an attribute of the subject.
prep (prepositional modifier): "of" introduces a prepositional phrase modifying "subfield".
pobj (prepositional object): "linguistics" is the object of the preposition "of".
punct (punctuation): The period at the end of the sentence.
'''

tokens = clean_and_tokenize(text)
tokens_without_stopwords = remove_stopwords(tokens)
stemmed_tokens = stem_tokens(tokens)
pos_tags = pos_tagging(tokens)
lemmatized_tokens = lemmatize_tokens(tokens)
ner_results = named_entity_recognition(text)
sentences = sent_tokenize(text)

# Apply chunking
chunked_sentences = chunking(sentences)
chinked_sentences = chinking(chunked_sentences)

# Print results
for i, (chunked_sentence, chinked_sentence) in enumerate(zip(chunked_sentences, chinked_sentences)):
    print(f"Chunked Sentence {i+1}:")
    print(chunked_sentence)
    print()
    print(f"Chinked Sentence {i+1}:")
    print(chinked_sentence)
    print()
# Print results
for i, chunked_sentence in enumerate(chunked_sentences):
    print(f"Chunked Sentence {i+1}:")
    print(chunked_sentence)
    print()
print("Tokens without stopwords:", tokens_without_stopwords)
print("Stemmed Tokens:", stemmed_tokens)
print("POS Tags:", pos_tags)
print("Lemmatized Tokens:", lemmatized_tokens)
print("Named-Entity Recognition Results:")
for ner_result in ner_results:
    print(ner_result)



# Function to perform rule-based matching
def rule_based_matching(text):
    pattern = [
        {"LOWER": "natural"}, 
        {"IS_PUNCT": True}, 
        {"LOWER": "language"}, 
        {"IS_PUNCT": True}, 
        {"LOWER": "processing"}
    ]
    matcher = Matcher(nlp.vocab)
    matcher.add("NLP", [pattern])
    doc = nlp(text)
    matches = matcher(doc)
    return [(doc[start:end].text, start, end) for match_id, start, end in matches]

text2 = "Apple is looking at buying U.K. startup for $1 billion. Steve Jobs founded Apple Inc. Natural, Language, Processing is a field of Artificial Intelligence."


# Apply rule-based matching
rule_based_results = rule_based_matching(text2)

# Print results


print("\nRule-Based Matching Results:")
for match_text, start, end in rule_based_results:
    print(f"Match: '{match_text}' found at positions {start} to {end}")