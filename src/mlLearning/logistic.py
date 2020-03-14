import  numpy as np
import matplotlib.pyplot as plt

"""
 数据预处理
"""
def load_data():
    data_arr=[]
    label_arr=[]
    f=open("./set.txt",'r')
    for line in f.readlines():
        line_arr=line.strip().split()
        data_arr.append([1.0
                            ,np.float(line_arr[0])
                            ,np.float(line_arr[1])])
        label_arr.append(int(line_arr[2]))
    return data_arr,label_arr

# 分类函数
def sigmoid(x):
    return  1.0/(1+np.exp(-x))

#梯度上升/梯度下降
def grad_ascent(data_arr,class_labels):
    data_mat=np.mat(data_arr)
    label_mat=np.mat(class_labels).transpose()
    m,n=np.shape(data_mat)
    alpha=0.001 #学习率
    cycles=500
    weights=np.ones((n,1))
    for k in range(cycles):
        h=sigmoid(data_mat*weights)
        error=label_mat-h
        weights=weights+alpha*data_mat.transpose()*error    # -/+ 区分是梯度下降还是梯度上升
    return  weights

def plot_best_fit(weights):
    data_mat, label_mat = load_data()
    data_arr = np.array(data_mat)
    n = np.shape(data_mat)[0]
    x_cord1 = []
    y_cord1 = []
    x_cord2 = []
    y_cord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            x_cord1.append(data_arr[i, 1])
            y_cord1.append(data_arr[i, 2])
        else:
            x_cord2.append(data_arr[i, 1])
            y_cord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x_cord1, y_cord1, s=30, color='k', marker='^')
    ax.scatter(x_cord2, y_cord2, s=30, color='red', marker='s')

    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    """
    y的由来，卧槽，是不是没看懂？
    首先理论上是这个样子的。
    dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
    w0*x0+w1*x1+w2*x2=f(x)
    x0最开始就设置为1， x2就是我们画图的y值，而f(x)被我们磨合误差给算到w0,w1,w2身上去了
    所以： w0+w1*x+w2*y=0 => y = (-w0-w1*x)/w2   
    """
    ax.plot(x, y)
    plt.xlabel('x1')
    plt.ylabel('y1')
    plt.show()

def test():
    """
    这个函数只要就是对上面的几个算法的测试，这样就不用每次都在power shell 里面操作，不然麻烦死了
    :return:
    """
    data_arr, class_labels = load_data()
    # 注意，这里的grad_ascent返回的是一个 matrix, 所以要使用getA方法变成ndarray类型
    weights = grad_ascent(data_arr, class_labels).getA()
    # weights = stoc_grad_ascent0(np.array(data_arr), class_labels)
    #weights = stoc_grad_ascent1(np.array(data_arr), class_labels)
    plot_best_fit(weights)

if __name__ == '__main__':
    test();