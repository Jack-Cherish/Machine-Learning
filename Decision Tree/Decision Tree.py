# -*- coding: UTF-8 -*-
from math import log


"""
函数说明:计算给定数据集的香农熵

Parameters:
	dataSet - 数据集
Returns:
	shannonEnt - 香农熵
Author:
	Jack Cui
Modify:
	2017-03-29
"""
def calcShannonEnt(dataSet):
	#返回数据集的行数
	numEntires = len(dataSet)
	#保存每个标签(Label)出现次数的字典
	labelCounts = {}
	#对每组特征向量进行统计
	for featVec in dataSet:
		#提取标签(Label)信息
		currentLabel = featVec[-1]
		#如果标签(Label)没有放入统计次数的字典,添加进去
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		#Label计数
		labelCounts[currentLabel] += 1
	#香农熵
	shannonEnt = 0.0
	#计算香农熵
	for key in labelCounts:
		#选择该标签(Label)的概率
		prob = float(labelCounts[key]) / numEntires
		#利用公式计算
		shannonEnt -= prob * log(prob, 2)
	#返回香农熵
	return shannonEnt

"""
函数说明:创建测试数据集

Parameters:
	无
Returns:
	dataSet - 数据集
	features - 特征
Author:
	Jack Cui
Modify:
	2017-03-29
"""
def createDataSet():
	#数据集
	dataSet = [[1, 1, 'maybe'],
			[1, 1, 'yes'],
			[1, 0, 'no'],
			[0, 1, 'no'],
			[0, 1, 'no']]
	#特征
	features = ['no surfacing', 'flippers']
	#返回数据集和标签
	return dataSet, features

"""
函数说明:按照给定特征划分数据集

Parameters:
	dataSet - 待划分的数据集
	axis - 划分数据集的特征
	value - 需要返回的特征的值
Returns:
	无
Author:
	Jack Cui
Modify:
	2017-03-30
"""
def splitDataSet(dataSet, axis, value):
	#创建返回的数据集列表
	retDataSet = []
	#遍历数据集
	for featVec in dataSet:
		if featVec[axis] == value:
			#去掉axis特征
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			#将符合条件的添加到返回的数据集
			retDataSet.append(reducedFeatVec)
	#返回划分后的数据集
	return retDataSet

"""
函数说明:选择最好的数据集划分方式

Parameters:
	dataSet - 数据集
Returns:
	bestFeature - 最好划分数据集的特征
Author:
	Jack Cui
Modify:
	2017-03-30
"""
def chooseBestFeatureToSplit(dataSet):
	#特征数量
	numFeatures = len(dataSet[0]) - 1
	#计算数据集的香农熵
	baseEntropy = calcShannonEnt(dataSet)
	#信息增益
	bestInfoGain = 0.0
	#划分特征
	bestFeature = -1
	#遍历所有特征
	for i in range(numFeatures):
	

if __name__ == '__main__':
	myDat, features = createDataSet()
	chooseBestFeatureToSplit(myDat)