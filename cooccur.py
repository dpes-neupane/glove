import numpy as np
import tqdm
from tensorflow import keras
import tensorflow as tf

SEED = 42
AUTOTUNE = tf.data.AUTOTUNE 
def getFiles(samplesize=25):
    import os
    import random
    random.seed(SEED)
    files = []
    total = 0
    added = 0
    for dir in os.listdir('./datag/Gutenberg'):
        t_files = os.listdir(os.path.join('./datag/Gutenberg',dir))
        sample = random.sample(range(len(t_files)), samplesize)
        total += len(t_files)
        for i, file in enumerate(t_files):
            file = os.path.join('./datag/Gutenberg', dir + '/' + file)
            if sample != [] and i in sample: 
                files.append(file)
                added += 1
                
    print(f"Total Files: {total} and the added files to vectorize is: {added}")
    return files

def createCooccurrenceMat(sequences, window_size, vocab_size):
    cooccurr = np.zeros(shape=(vocab_size, vocab_size), dtype='float32') 
    for sequence in tqdm.tqdm(sequences):
        skipgrams , _ = keras.preprocessing.sequence.skipgrams(
            sequence=sequence, 
            window_size=window_size,
            vocabulary_size=vocab_size,
            seed=SEED,
            negative_samples=0,
        )
        for target, context in skipgrams:
            cooccurr[target,context] +=1

    
    return tf.sparse.from_dense(cooccurr)
    
if __name__ == "__main__":
    all_files = getFiles(2)
    test_txt = """The newspapers are the alone of  in the  """
    # txt_ds = tf.data.TextLineDataset(all_files).filter(lambda x: tf.cast(tf.strings.length(x), bool))
    txt_ds = tf.data.Dataset.from_tensor_slices([test_txt])
    sequence_length = 40
    vectorizer_layer = keras.layers.TextVectorization(output_sequence_length=sequence_length)
    vectorizer_layer.adapt(txt_ds.batch(1024))
    vocab_size = vectorizer_layer.vocabulary_size()
    txt_vector_ds = txt_ds.batch(1024).prefetch(AUTOTUNE).map(vectorizer_layer).unbatch()
    sequences = list(txt_vector_ds.as_numpy_iterator())
    inverse_vocabulary = vectorizer_layer.get_vocabulary()
    for seq in sequences[100:105]:
        print(f"{seq}  =>  {[inverse_vocabulary[i] for i in seq]}")
    mat = createCooccurrenceMat(sequences, 
                          window_size=2, 
                          vocab_size=vocab_size
                          )
    # print(f"{[inverse_vocabulary[i] for i in sequences[100]]}")
    
    # for s in skipgrams:
    #     print(f"{[inverse_vocabulary[s[0]], inverse_vocabulary[s[1]]]}")
    # for row in mat:
    #     print(f"{}")
    # print(inverse_vocabulary)
    # print(vocab_size, sep="\n")
    # print(*mat, sep="\n")
    print(tf.sparse.to_dense(mat))
    # print(mat.count_nonzero())