import  tensorflow as tf
import  pandas as pd
import numpy as np

#处理数据
ratings_df=pd.read_csv('ratings.csv')
movies_df=pd.read_csv('movies.csv')

movies_df['movieRow']=movies_df.index
movies_df=movies_df[['movieRow','movieId','title']]
#movies_df.to_csv('moviesP.csv',index=False,header=True,encoding='utf-8')

ratings_df=pd.merge(ratings_df,movies_df,on='movieId')

ratings_df=ratings_df[['userId','movieRow','rating']]
#ratings_df.to_csv('ratingsP.csv',index=False,header=True,encoding='utf-8')

#创建矩阵
userNo=ratings_df['userId'].max()+1
movieNo=movies_df['movieRow'].max()+1
rating=np.zeros((movieNo,userNo))
flag=0
ratings_df_length=np.shape(ratings_df)[0]
for index,row in ratings_df.iterrows():
   rating[int(row['movieRow']),int(row['userId'])]=row['rating']
   flag=+1
record=rating>0
record=np.array(record,dtype=int)

#数据平滑
def normalizeRating(rating,record):
   m,n=rating.shape
   rating_mean=np.zeros((m,1))
   rating_norm=np.zeros((m,n))
   for i in range(m):
      idx=record[i,:]!=0
      rating_mean[i]=np.mean(rating[i,idx])
      rating_norm[i,idx]-=rating_mean[i]
   return rating_norm,rating_mean
rating_mean,rating_norm=normalizeRating(rating,record)
rating_norm=np.nan_to_num(rating_norm)
rating_mean=np.nan_to_num(rating_norm)

#10种类型的电影
num_features=10
X_paramater=tf.Variable(tf.random_normal([movieNo,num_features]),stddev=0.35)   #初始化电影种类
Theta_paramater=tf.Variable(tf.random_normal([userNo,num_features]),stddev=0.35) #初始化用户喜好参数
#拉姆达=1
loss=1/2 * tf.reduce_sum((((tf.matmul(X_paramater,Theta_paramater,transpose_b=True))-rating_norm)*record)**2)+1/2(tf.reduce_sum(X_paramater**2)+tf.reduce_sum(Theta_paramater**2))
optmizer=tf.train.AdamOptimizer(1e-4)
train=optmizer.minimize


#训练模型






if __name__ == '__main__':
   ratings_df.tail();