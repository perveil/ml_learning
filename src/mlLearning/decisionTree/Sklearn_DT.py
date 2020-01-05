from sklearn import  tree
from sklearn.preprocessing  import LabelEncoder
from sklearn.externals.six import StringIO
import  pandas as pd
import pydotplus
if __name__ == '__main__':
    fr=open('lenses.txt')
    lenses=[inst.strip().split('\t') for inst in fr.readlines()]
    lenses_target = []
    for each in lenses:
        lenses_target.append(each[-1])
    lensesLables=['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_list = []
    lenses_dict = {}
    for each_label in lensesLables:
        for each in lenses:
            lenses_list.append(each[lensesLables.index(each_label)])
        lenses_dict[each_label]=lenses_list
        lenses_list=[]
    lenses_pd = pd.DataFrame(lenses_dict) #构建pd Data结构
    le=LabelEncoder()
    #序列化,String --->int
    for col in lenses_pd.columns:
         lenses_pd[col]=le.fit_transform(lenses_pd[col])

    clf=tree.DecisionTreeClassifier(max_depth=4)
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,  # graphviz绘制决策树
                         feature_names=lenses_pd.keys(),
                         class_names=clf.classes_,
                         filled=True, rounded=True,
                         special_characters=True)
    graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf('tree.pdf');
