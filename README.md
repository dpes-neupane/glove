## Implementation of GloVe Word Embeddings

Implementation of [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) word embeddings in tensorflow. 

### Description of the GloVe:

GloVe or Global Vectors of Word Representations is statitics based method for generating word embeddings. The idea is to use the co-occurrences counts of the target words and context words to generate the vector embeddings.

It establishes how different words correlates to each other through the use of ratio of probabilities of $P_{ik} / P_{jk} $. Here, $ P_{ik} = P(k|i) = X_{ik} / X_i $ is the probability that the word $k$ appear in the context of word $i$ and same with $ P_{jk}$.

An interesting aspect of GloVe method is that how it has used a simple loss function to get the embedding vectors unlike the __word2vec__ model which used a neural language model. 

### Pros:

- Simple to understand. 
- Simple Loss function. 
- No window based approach. Whole corpus is taken into account while training in each step toward convergence.

### Cons:

- Loves memory as large part of vocabulary have zero elements in the cooccurrence matrix. Can be mitigated by using a sparse matrix but still the embedding weights will have huge intermediate matrices- if used matrix multiplication for processing. Although, this can also be mitigated if a sequential process is used to first compute dot products of $i$ and $j$ and compute the loss for it and sum all the values. This has not been implemented in here.
- Like every other word embeddings techniques, it also gives better results in huge datasets. But as the paper has pointed out, it decently captures word sense in smaller corpuses than __word2vec__. (I might upload the cosine similarity data in the future.)

Overall, an interesting read and easy to implement algorithm to produce pretty good embeddings. 






