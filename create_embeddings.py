import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences

t = keras.preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', char_level=False, oov_token=None, document_count=0)
t.fit_on_texts(corpus)
vocab_size = len(t.word_index) + 1
print(vocab_size)
encoded_corpus = t.texts_to_sequences(corpus)

max_length = 80
padded_docs = pad_sequences(encoded_corpus, maxlen=max_length, padding='post')
print(padded_docs)

# load the whole embedding into memory
embeddings_index = dict()
f = open('data_embeddings/en/glove.6B.50d.txt',encoding = "utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 50))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector