from simple_word2vec import cbow, Word2Vec

docs = [
    'this is a book',
    'this is a cat',
    'how are you',
    'cat like milk',
    'a book about cat',
    'you saw a cat',
    'you saw a book',
]
print('Training document:')
print(docs)

print('Initialize Word2Vec...')
word2vec = Word2Vec(dimensions=5)

print(word2vec)

print('Training Word2Vec on the document...')
word2vec.train(docs, num_iterations=1000, verbose=50)

print('')
print('== Word vector prediction ==')
print('Prediction:')
words = ['this', 'is', 'a', 'book', 'cat', 'you', 'milk']
for w in words:
    print(w, word2vec.predict([w]))

print('Word Embedding Matrix:')
print(word2vec.W_i)

print('')
print('== Center word prediction ==')
input_sentence = 'this is milk'
input_tokens = word2vec.token_encoder.transform([input_sentence.split(' ')])[0]
print('Input sentence:', input_sentence)
print('Encoded tokens:', input_tokens)
y_pred = cbow.predict(input_tokens, word2vec.W_i, word2vec.W_o)
w_pred = word2vec.token_encoder.list_of_tokens[y_pred]
y_prod_pred = cbow.predict_prob(input_tokens, word2vec.W_i, word2vec.W_o)
print('CBOW predict most likely center word to be:', y_pred, w_pred,
      'with probability =', y_prod_pred[y_pred])
print('CBOW probability of each words as center word:')
for w, p in zip(word2vec.token_encoder.list_of_tokens, y_prod_pred):
    print(w, p)
