# Author:yifan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

from sklearn.tree import DecisionTreeClassifier #分类树
from sklearn.model_selection  import train_test_split#测试集和训练集
from sklearn.pipeline import Pipeline #管道
from sklearn.feature_selection import SelectKBest #特征选择
from sklearn.feature_selection import chi2 #卡方统计量

from sklearn.preprocessing import MinMaxScaler  #数据归一化
from sklearn.decomposition import PCA #主成分分析
from sklearn.model_selection import GridSearchCV #网格搜索交叉验证，用于选择最优的参数
## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore', category=FutureWarning)

#读取数据
path = './datas/iris.data'
data = pd.read_csv(path, header=None)
x=data[list(range(4))]#获取X变量
y=pd.Categorical(data[4]).codes    #把Y转换成分类型的0,1,2
print("总样本数目：%d;特征属性数目:%d" % x.shape)   #总样本数目：150;特征属性数目:4
# print(y)
#数据进行分割（训练数据和测试数据）
x_train1, x_test1, y_train1, y_test1 = train_test_split(x, y, train_size=0.8, random_state=14)
x_train, x_test, y_train, y_test = x_train1, x_test1, y_train1, y_test1
print ("训练数据集样本数目：%d, 测试数据集样本数目：%d" % (x_train.shape[0], x_test.shape[0]))
## 因为需要体现以下是分类模型，因为DecisionTreeClassifier是分类算法，要求y必须是int类型
y_train = y_train.astype(np.int)
y_test = y_test.astype(np.int)

ss = MinMaxScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
print ("原始数据各个特征属性的调整最小值:",ss.min_)
print ("原始数据各个特征属性的缩放数据值:",ss.scale_)


#参数优化
pipe = Pipeline([
            ('mms', MinMaxScaler()),
            ('skb', SelectKBest(chi2)),
            ('pca', PCA()),
            ('decision', DecisionTreeClassifier(random_state=0))
        ])

# 参数
parameters = {
    "skb__k": [1,2,3,4],
    "pca__n_components": [0.5,0.99],#设置为浮点数代表主成分方差所占最小比例的阈值，这里不建议设置为数值，思考一下？
    "decision__criterion": ["gini", "entropy"],
    "decision__max_depth": [1,2,3,4,5,6,7,8,9,10]
}
#数据
x_train2, x_test2, y_train2, y_test2 = x_train1, x_test1, y_train1, y_test1
#模型构建：通过网格交叉验证，寻找最优参数列表， param_grid可选参数列表，cv：进行几折交叉验证
gscv = GridSearchCV(pipe, param_grid=parameters,cv=3)
#模型训练
gscv.fit(x_train2, y_train2)
#算法的最优解
print("最优参数列表:", gscv.best_params_)
#结果最优参数列表: {'decision__criterion': 'gini', 'decision__max_depth': 4, 'pca__n_components': 0.99, 'skb__k': 3}
print("score值：",gscv.best_score_)    #score值： 0.95
print("最优模型:", end='')
print(gscv.best_estimator_)
#预测值
y_test_hat2 = gscv.predict(x_test2)

#应用最优参数看效果
mms_best = MinMaxScaler()
skb_best = SelectKBest(chi2, k=3)
pca_best = PCA(n_components=0.99)
decision3 = DecisionTreeClassifier(criterion='gini', max_depth=4)
#构建模型并训练模型
x_train3, x_test3, y_train3, y_test3 = x_train1, x_test1, y_train1, y_test1
x_train3 = pca_best.fit_transform(skb_best.fit_transform(mms_best.fit_transform(x_train3), y_train3))
x_test3 = pca_best.transform(skb_best.transform(mms_best.transform(x_test3)))
decision3.fit(x_train3, y_train3)
print("正确率:", decision3.score(x_test3, y_test3))    #正确率: 0.9666666666666667

# 基于原始数据前3列比较一下决策树在不同深度的情况下错误率
### TODO: 将模型在训练集上的错误率也画在图中
x_train4, x_test4, y_train4, y_test4 = train_test_split(x.iloc[:, :2], y, train_size=0.7, random_state=14)

depths = np.arange(1, 15)
err_list = []
for d in depths:
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=d,
                                 min_samples_split=10)  # 仅设置了这二个参数，没有对数据进行特征选择和降维，所以跟前面得到的结果不同
    clf.fit(x_train4, y_train4)

    ## 计算的是在训练集上的模型预测能力
    score = clf.score(x_test4, y_test4)
    err = 1 - score
    err_list.append(err)
    print("%d深度，测试集上正确率%.5f" % (d, clf.score(x_train4, y_train4)))
    print("%d深度，训练集上正确率%.5f\n" % (d, score))

## 画图
plt.figure(facecolor='w')
plt.plot(depths, err_list, 'ro-', lw=3)
plt.xlabel(u'决策树深度', fontsize=16)
plt.ylabel(u'错误率', fontsize=16)
plt.grid(True)
plt.title(u'决策树层次太多导致的拟合问题(欠拟合和过拟合)', fontsize=18)
plt.show()


