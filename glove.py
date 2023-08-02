import tensorflow as tf
import numpy as np
import keras
from keras import layers
import cooccur as co

class GEmbedding(keras.layers.Layer):
    def __init__(self,  vocab_size, embedding_dim):
        super().__init__()
        self.target_e = self.add_weight(
            shape=(vocab_size, embedding_dim),
            initializer="random_uniform",
            trainable=True,
        )
        self.context_e = self.add_weight(
            shape=(vocab_size, embedding_dim),
            initializer="random_uniform",
            trainable=True,
        )
        self.b_t = self.add_weight(
            shape=(vocab_size,), 
            initializer="random_uniform",
            trainable=True,
        )
        self.b_c = self.add_weight(
            shape=(vocab_size,), 
            initializer="random_uniform",
            trainable=True,
        )
        
    def call(self,input):
        return tf.matmul(self.target_e, tf.transpose(self.context_e)) + self.b_t + self.b_c
    
def gloveLoss(dots, cooc, xmax):
    # def weightingFunc(x, xmax):
    #     # t = x_ij / xmax
    #     t = x.data / xmax

    #     # print(t)
    #     f_x = (t < 1) * t**(3/4)
    #     f_x = (t >= 1) + f_x 
    #     return sp.csr_matrix((f_x, x.indices, x.indptr), shape=x.shape).todense()
    
    def weightingFunc(x, xmax):
        t = x.values / xmax

        # print(t)
        f_x = tf.where(t > 1, 1.0, 0.0) * t**(3/4)
        # print(f_x)
        f_x = tf.where(t >= 1, 1.0, 0.0) + f_x 
        return tf.sparse.SparseTensor(indices=x.indices, 
                                    values=f_x, 
                                    dense_shape=x.shape)
    # f_x = weightingFunc(cooc, xmax)
    # logs = tf.math.log(cooc + 1e-15)
    # logs = sp.csr_matrix((np.log(cooc.data), cooc.indices, cooc.indptr), shape=cooc.shape)
    logs = tf.sparse.SparseTensor(indices=cooc.indices, 
                                  values=tf.math.log(cooc.values), 
                                  dense_shape=cooc.shape)
    
    l_s = tf.sparse.to_dense(weightingFunc(cooc, xmax)) * tf.math.square(dots - tf.sparse.to_dense(logs))
    # print(l_s)
    return tf.reduce_sum(l_s)
    
def train(vocab_size, cooc, xmax, emb_dim, epochs=100, learning_rate=0.001):
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model = GEmbedding(vocab_size=vocab_size, embedding_dim=emb_dim)
    # print(model(1).numpy())
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            y = model(1)
            # print(y)
            loss = gloveLoss(y, cooc, xmax)
            # print(loss)
        grads = tape.gradient(loss, model.trainable_weights)
        # print(grads)
        # exit()
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        print(f"Epoch: {epoch}\tLoss: {loss:.4f}")
    return model

def saveEmb(model, vocabulary, foldername):
    t_w = model.target_e.numpy()
    c_w = model.context_e.numpy()
    # print(t_w.numpy().shape, c_w.numpy().shape)
    import io
    weights = ( t_w + c_w ) / 2
    # vocab = vectorize_layer.get_vocabulary()

    out_v = io.open(foldername + '/vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open(foldername + '/metadata.tsv', 'w', encoding='utf-8')

    for index, word in enumerate(vocabulary):
        if index == 0:
          continue
        vec = weights[index]
        out_v.write('\t'.join([str(x) for x in vec]) + '\n')
        out_m.write(word + '\n')
    out_v.close()
    out_m.close()



if __name__ == "__main__":
    from cooccur import *
    all_files = getFiles(1)
    # test_txt = """The are the of  in the when a is than the of when are the in of """
    txt_ds = tf.data.TextLineDataset(all_files).filter(lambda x: tf.cast(tf.strings.length(x), bool))
    # txt_ds = tf.data.Dataset.from_tensor_slices([test_txt])
    vocab_size = 500
    sequence_length = 40
    vectorizer_layer = keras.layers.TextVectorization( max_tokens=vocab_size,
                                                       output_sequence_length=sequence_length
                                                     )
    vectorizer_layer.adapt(txt_ds.batch(1024))
    # vocab_size = vectorizer_layer.vocabulary_size()
    txt_vector_ds = txt_ds.batch(1024).prefetch(AUTOTUNE).map(vectorizer_layer).unbatch()
    sequences = list(txt_vector_ds.as_numpy_iterator())
    inverse_vocabulary = vectorizer_layer.get_vocabulary()
    for seq in sequences[100:105]:
        print(f"{seq}  =>  {[inverse_vocabulary[i] for i in seq]}")
    mat = createCooccurrenceMat(sequences, 
                          window_size=5, 
                          vocab_size=vocab_size
                          )
    # print(f"{[inverse_vocabulary[i] for i in sequences[100]]}")
    
    # for s in skipgrams:
    #     print(f"{[inverse_vocabulary[s[0]], inverse_vocabulary[s[1]]]}")
    # for row in mat:
    #     print(f"{}")
    np.set_printoptions(precision=2, threshold=1000)
    # print(inverse_vocabulary)
    print(vocab_size, sep="\n")
    # print(*mat, sep="\n")
    # print("\ny\n")
    model = train(vocab_size, mat, xmax=100, emb_dim=100, epochs=20, learning_rate=0.09)
    # saveEmb(model, inverse_vocabulary)
    # y = model(1)
    # print(*y.numpy(), sep="\n")
    
    