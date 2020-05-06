# Author:yifan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

from sklearn.tree import DecisionTreeClassifier #分类树
from sklearn.model_selection  import train_test_split#测试集和训练集
from sklearn.feature_selection import SelectKBest #特征选择
from sklearn.feature_selection import chi2 #卡方统计量

from sklearn.preprocessing import MinMaxScaler  #数据归一化
from sklearn.decomposition import PCA #主成分分析

## 设置属性防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore', category=FutureWarning)
# iris_feature_E = 'sepal length', 'sepal width', 'petal length', 'petal width'
# iris_feature_C = '花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'
# iris_class = 'Iris-setosa', 'Iris-versicolor', 'Iris-virginica'
#读取数据
path = './datas/iris.data'
data = pd.read_csv(path, header=None)
x=data[list(range(4))]             #获取X变量
y=pd.Categorical(data[4]).codes    #把Y转换成分类型的0,1,2
# print("总样本数目：%d;特征属性数目:%d" % x.shape)   #总样本数目：150;特征属性数目:4

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
#特征选择：从已有的特征中选择出影响目标值最大的特征属性
#常用方法：{ 分类：F统计量、卡方系数，互信息mutual_info_classif
        #{ 连续：皮尔逊相关系数 F统计量 互信息mutual_info_classif
#SelectKBest（卡方系数）
#在当前的案例中，使用SelectKBest这个方法从4个原始的特征属性，选择出来3个
ch2 = SelectKBest(chi2,k=3)
#K默认为10      如果指定了，那么就会返回你所想要的特征的个数
x_train = ch2.fit_transform(x_train, y_train)#训练并转换
x_test = ch2.transform(x_test)#转换

select_name_index = ch2.get_support(indices=True)
print ("对类别判断影响最大的三个特征属性分布是:",ch2.get_support(indices=False))   #[ True False  True  True]
print(select_name_index)  #[0 2 3]

#降维：对于数据而言，如果特征属性比较多，在构建过程中，会比较复杂，这个时候考虑将多维（高维）映射到低维的数据
#常用的方法：
#PCA：主成分分析（无监督）
#LDA：线性判别分析（有监督）类内方差最小，人脸识别，通常先做一次pca
pca = PCA(n_components=2)#构建一个pca对象，设置最终维度是2维
#这里是为了后面画图方便，所以将数据维度设置了2维，一般用默认不设置参数就可以
x_train = pca.fit_transform(x_train)  #训练并转换
x_test = pca.transform(x_test)       #转换

#模型的构建
model = DecisionTreeClassifier(criterion='entropy',random_state=0)    #另外也可选gini
#模型训练
model.fit(x_train, y_train)
#模型预测
y_test_hat = model.predict(x_test)
#模型结果的评估
y_test2 = y_test.reshape(-1)
result = (y_test2 == y_test_hat)
print ("准确率:%.2f%%" % (np.mean(result) * 100))   #准确率:96.67%
#实际可通过参数获取
print ("Score：", model.score(x_test, y_test))#准确率   #Score： 0.9666666666666667
print ("Classes:", model.classes_)     #Classes: [0 1 2]
print("获取各个特征的权重:", end='')   #获取各个特征的权重:[0.93420127 0.06579873]
print(model.feature_importances_)

#画图
N = 100  #横纵各采样多少个值
# print(x_train.T[0],x_train.T[1])
x1_min = np.min((x_train.T[0].min(), x_test.T[0].min()))
x1_max = np.max((x_train.T[0].max(), x_test.T[0].max()))
x2_min = np.min((x_train.T[1].min(), x_test.T[1].min()))
x2_max = np.max((x_train.T[1].max(), x_test.T[1].max()))
t1 = np.linspace(x1_min, x1_max, N)
t2 = np.linspace(x2_min, x2_max, N)
x1, x2 = np.meshgrid(t1, t2)              # 生成网格采样点
x_show = np.dstack((x1.flat, x2.flat))[0] #测试点
y_show_hat = model.predict(x_show)        #预测值

y_show_hat = y_show_hat.reshape(x1.shape)  #使之与输入的形状相同
# print(y_show_hat.shape)

#画图
plt_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
plt_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

plt.figure(facecolor='w')
## 画一个区域图
plt.pcolormesh(x1, x2, y_show_hat, cmap=plt_light)
# 画测试数据的点信息
plt.scatter(x_test.T[0], x_test.T[1], c=y_test.ravel(), edgecolors='k', s=150, zorder=10, cmap=plt_dark, marker='*')  # 测试数据
# 画训练数据的点信息
plt.scatter(x_train.T[0], x_train.T[1], c=y_train.ravel(), edgecolors='k', s=40, cmap=plt_dark)  # 全部数据
plt.xlabel(u'特征属性1', fontsize=15)
plt.ylabel(u'特征属性2', fontsize=15)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.grid(True)
plt.title(u'鸢尾花数据的决策树分类', fontsize=18)
plt.show()



