# 梯度下降
import numpy as np
import matplotlib.pyplot as plt

m=20

#参数  y=Θ0+Θ1x
X0 = np.ones((m, 1))
X1 = np.arange(1, m+1).reshape(m, 1)
X = np.hstack((X0, X1))  #对应轴堆叠
#

# 结果y
y = np.array([
    3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
    11, 13, 13, 16, 17, 18, 17, 19, 21
]).reshape(m, 1)
alpha=0.01

#损失函数 采用最小二乘法
def error_function(theta,x,y):
    diff=np.dot(x,theta)-y;
    return (1.0/2*m)*np.dot(np.transpose(diff),diff)
#梯度计算
def gradient_function(theta,X,y):
    diff=np.dot(X,theta)-y;
    return (1.0/m)*np.dot(np.transpose(X),diff)
#梯度下降
def gradient_descent(X,y,alpha):
    theta=np.array([1,1]).reshape(2,1) #20*2 的全是1的矩阵
    gradient=gradient_function(theta,X,y)
    #测试精度
    while not np.all(np.absolute(gradient)<=1e-5):
        theta=theta-alpha*gradient
        gradient=gradient_function(theta,X,y)
    return theta

def pic(theta):
    x_cord1 = []
    y_cord1 = []
    for i in range(m):
        x_cord1.append(X1[i])
        y_cord1.append(y[i])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, color='k', marker='^')
    x = np.arange(-3.0, 20.0, 0.1)
    y1 = (theta[0]+ theta[1] * x)
    ax.plot(x, y1)
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()

if __name__ == '__main__':
    # theta=gradient_descent(X,y,alpha)
    pic(gradient_descent(X,y,alpha))