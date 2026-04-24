documents = [
    ("I love this movie", "pos"),
    ("This movie is terrible", "neg"),
    ("I love this film", "pos"),
    ("This film is boring", "neg")
]
wlist = {'boring': -2, 'film': 0, 'i': 0, 'is': 0, 'love': 2, 'movie': 0, 'terrible': -2, 'this': 0}
def sentence_split(text):
    return text.lower().split()

for doc, label in documents:
    print(label, "->", sentence_split(doc))

def build_vocab(documents):
    vocab = set()
    for doc, _ in documents:
        vocab.update(sentence_split(doc))
    return sorted(list(vocab))

print("Vocabulary:", build_vocab(documents))
vocab = build_vocab(documents)

def vectorize_sentence(sentence, vocab):
    vocab_index = {word: i for i, word in enumerate(vocab)}  
    vector = [0] * len(vocab)
    for word in sentence:
        if word in vocab_index:
            index = vocab_index[word]
            vector[index] = wlist.get(word, 0)
    return vector
print("Vectorized sentences:")
for doc, label in documents:
    print(label, "->", vectorize_sentence(sentence_split(doc), vocab))

def classify(sentence, vocab):
    vector = vectorize_sentence(sentence_split(sentence), vocab)
    score = sum(vector)
    if score > 0:
        return "pos"
    elif score < 0:
        return "neg"
    else:
        return "neutral"


for doc, label in documents:
    prediction = classify(doc, vocab)
    print(f"Real: {label} | Predicted: {prediction} | Text: {doc}")