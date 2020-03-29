import  tensorflow as tf
import  numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sess=tf.Session()

x_shape=[1,4,4,1] # 4*4的picture
x_val=np.random.uniform(size=x_shape)

x_data=tf.placeholder(tf.float32,shape=x_shape)

my_filter=tf.constant(0.25,shape=[2,2,1,1]) #2*2 的 filter
my_strides=[1,2,2,1]

mov_avg_layer=tf.nn.conv2d(x_data,my_filter,my_strides,padding='SAME',name='Moving')
# 计算卷积之后的图片的尺寸公式 （w-f+2p）/s +1
# (4-2)/2 +1 =2


def custom_layer(input_matrix):
    input_matrix_squeeze=tf.squeeze(input_matrix)
    A=tf.constant([[1., 2.], [-1., 3.]])
    b=tf.constant(1., shape=[2, 2])
    temp1=tf.matmul(A,input_matrix_squeeze)
    temp=tf.add(temp1,b)
    return tf.sigmoid(temp)

with tf.name_scope('custom_layer') as scope:
    custom_layer1=custom_layer(mov_avg_layer)

if __name__ == '__main__':
    print(sess.run(custom_layer1,feed_dict={x_data:x_val}))





