# -*- coding: UTF-8 -*-
import numpy as np
import operator
import collections

"""
函数说明:创建数据集

Parameters:
	无
Returns:
	group - 数据集
	labels - 分类标签
Modify:
	2017-07-13
"""
def createDataSet():
	#四组二维特征
	group = np.array([[1,101],[5,89],[108,5],[115,8]])
	#四组特征的标签
	labels = ['爱情片','爱情片','动作片','动作片']
	return group, labels

"""
函数说明:kNN算法,分类器

Parameters:
	inX - 用于分类的数据(测试集)
	dataSet - 用于训练的数据(训练集)
	labes - 分类标签
	k - kNN算法参数,选择距离最小的k个点
Returns:
	sortedClassCount[0][0] - 分类结果

Modify:
	2017-11-09 by Cugtyt 
		* GitHub(https://github.com/Cugtyt) 
		* Email(cugtyt@qq.com)
		Use list comprehension and Counter to simplify code
	2017-07-13
"""
def classify0(inx, dataset, labels, k):
	# 计算距离
	dist = np.sum((inx - dataset)**2, axis=1)**0.5
	# k个最近的标签
	k_labels = [labels[index] for index in dist.argsort()[0 : k]]
	# 出现次数最多的标签即为最终类别
	label = collections.Counter(k_labels).most_common(1)[0][0]
	return label

if __name__ == '__main__':
	#创建数据集
	group, labels = createDataSet()
	#测试集
	test = [101,20]
	#kNN分类
	test_class = classify0(test, group, labels, 3)
	#打印分类结果
	print(test_class)
