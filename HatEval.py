import fasttext
"""
classifier = fasttext.supervised('offenseval-trial-pre.txt', 'model', label_prefix='__label__')

result = classifier.test('test.txt')
print('P@1:', result.precision)
print('R@1:', result.recall)
print('Number of examples:', result.nexamples)
"""
print("")
print("")
print("Word2vec: ")
import multiprocessing
import gensim.models.word2vec as w2v
import sklearn.manifold
import pandas as pd

trainfile = "offenseval-trial-pre.txt"
testfile = "test.txt"
#list of sentences
listoflists = []

with open('offenseval-trial-pre.txt','r') as f:
    for line in f:
        listrepresentation = []
        for word in line.split():
           listrepresentation.append(word)
        listoflists.append(listrepresentation)

token_count = sum([len(sentence) for sentence in listoflists])
print("our list contains {0:,} tokens".format(token_count))

num_features = 300 #dimensions
min_word_count = 3
num_workers = multiprocessing.cpu_count()
context_size = 7
downsampling = 1e-3
seed = 1

word2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)
word2vec.build_vocab(listoflists)
word2vec.train(listoflists, total_examples=word2vec.corpus_count, epochs=word2vec.iter)
word2vec.save("thrones2vec.w2v")

tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
all_word_vectors_matrix = word2vec.wv.syn0
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)
points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[word2vec.wv.vocab[word].index])
            for word in word2vec.wv.vocab
        ]
    ],
    columns=["word", "x", "y"]
)
print("")
print(points.head(10))
