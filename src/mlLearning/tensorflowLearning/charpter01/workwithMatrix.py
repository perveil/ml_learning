import tensorflow as tf
import  numpy as np

if __name__ == '__main__':
    sess = tf.Session()
    identity_matrix = tf.diag([1.0, 2.0, 3.0, 4.0])
    #print(sess.run(tf.truncated_normal([5,5,4,3], stddev=0.1)))
    print(sess.run( tf.reshape(identity_matrix,[1,16*1])))

