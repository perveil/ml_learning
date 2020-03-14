import tensorflow as tf
import numpy as np


tf.reset_default_graph()

sentences = [ "i like dog Luckid", "i love coffee mi", "i hate milk hello"]
word_list=" ".join(sentences).split();
word_dict={w:i for i,w in enumerate(word_list)}
number_dict={i:w  for i,w in enumerate(word_list)}
n_class=len(word_dict)

#NNLM parameter
n_step=3
n_hidden =2

def make_batch(sentences):
    input_batch=[]
    target_batch=[]
    for sen in sentences:
        word=sen.split();
        input = [word_dict[n] for n in word[:-1]]
        target = word_dict[word[-1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(np.eye(n_class)[target])

    return input_batch,target_batch


input_batch, target_batch = make_batch(sentences)


if __name__ == '__main__':
    print(word_list)