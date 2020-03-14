import pandas as pd
from  surprise import SVD,Dataset,Reader
from surprise.model_selection import cross_validate,train_test_split

reader=Reader(rating_scale=(1,5),line_format="user item rating")
df_data=pd.read_csv('ratings.csv',usecols=['userId','movieId','rating'])
data=Dataset.load_from_df(df_data,reader)
trainset,testset=train_test_split(data,0.2)
model=SVD(n_factors=30)
model.fit(trainset)
print(model.pu.shape)
print(model.qi.shape)
print(model.get_neighbors(10,5))
# print(model.compute_similarities())

# n_factors=30
# pu=np.random.RandomState(0).normal(0.1,1,(610,30))
# qi=np.random.RandomState(0).normal(0.1,1,(9743,30))
# n_epochs=1000
# bu=np.random.RandomState(0).normal(0.1,1,(610,30))
# bi=np.random.RandomState(0).normal(0.1,1,(9743,30))
#
# lr_qi=lr_pu=lr_bu=lr_bi=0.1 #learning rate
# reg_qi=reg_pu=reg_bi=reg_bu=0.1 #正则化参数
#
# #train 函数
# for current_epoch in range(n_epochs):
#     for u, i, r in trainset.all_ratings():
#         # 计算(5)的第一项
#         dot = 0  # <q_i, p_u>
#         for f in range(30):
#             dot += qi[i, f] * pu[u, f]
#         err = r - (1 + bu[u] + bi[i] + dot)
#
#         bu[u] += lr_bu * (err - reg_bu * bu[u])
#         # 根据式(7)
#         bi[i] += lr_bi * (err - reg_bi * bi[i])
#
#         # update factors
#         for f in range(n_factors):
#             puf = pu[u, f]
#             qif = qi[i, f]
#             # 根据式(8)
#             pu[u, f] += lr_pu * (err * qif - reg_pu * puf)
#             # 根据式(9)
#             qi[i, f] += lr_qi * (err * puf - reg_qi * qif)


if __name__ == '__main__':
    print()
