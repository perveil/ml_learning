from sklearn import  tree
from sklearn.preprocessing  import LabelEncoder
from sklearn.externals.six import StringIO
import  pandas as pd
import pydotplus
'''
  4个参数预测一个参数的决策树
'''
def DT():
    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])

    lensesLables = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_list = []
    lenses_dict = {}
    for each_label in lensesLables:
        for each in lenses:
            lenses_list.append(each[lensesLables.index(each_label)])
        lenses_dict[each_label] = lenses_list
        lenses_list = []
    lenses_pd = pd.DataFrame(lenses_dict)  # 构建pd Data结构

    # 序列化,String --->int
    le = LabelEncoder()
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])

    clf=tree.DecisionTreeClassifier(criterion="entropy",max_depth=4)
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)
    print(clf.predict([[0,0,1,0]]))
    # dot_data = StringIO()
    # tree.export_graphviz(clf, out_file=dot_data,  # graphviz绘制决策树
    #                      feature_names=lenses_pd.keys(),
    #                      class_names=clf.classes_,
    #                      filled=True, rounded=True,
    #                      special_characters=True)
    # graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
    # graph.write_pdf('tree.pdf')

if __name__ == '__main__':
    DT()
'''
     DecisionTreeClassifier的参数
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 presort=False):
'''