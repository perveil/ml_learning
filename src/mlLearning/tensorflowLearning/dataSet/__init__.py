import  requests
import tensorflow  as tf
mnist=tf.keras.datasets.mnist
birthdata_url = 'https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
birth_file=requests.get(birthdata_url)
birth_data=birth_file.text.split('\'r\n')
