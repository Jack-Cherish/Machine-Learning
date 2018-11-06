# -*- coding: UTF-8 -*-
import numpy as np
import collections
from os import listdir

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
	2017-11-14 by Cugtyt 
		* GitHub(https://github.com/Cugtyt) 
		* Email(cugtyt@qq.com)
		Use list comprehension, Counter, broadcasting instead of 
		tile in numpy to simplify code.
	2017-03-25
"""
def classify0(inX, dataSet, labels, k):
	# 计算距离
	dist = np.sum((inX - dataSet) ** 2, axis=1) ** 0.5
	# k个最近的标签
	k_labels = [labels[index] for index in dist.argsort()[0: k]]
	# 出现次数最多的标签即为最终类别
	label = collections.Counter(k_labels).most_common(1)[0][0]
	return label

"""
函数说明:将32x32的二进制图像转换为1x1024向量。

Parameters:
	filename - 文件名
Returns:
	returnVect - 返回的二进制图像的1x1024向量

Modify:
	2017-03-25
"""
def img2vector(filename):
	#创建1x1024零向量
	returnVect = np.zeros((1, 1024))
	#打开文件
	fr = open(filename)
	#按行读取
	for i in range(32):
		#读一行数据
		lineStr = fr.readline()
		#每一行的前32个元素依次添加到returnVect中
		for j in range(32):
			returnVect[0, 32 * i + j] = int(lineStr[j])
	#返回转换后的1x1024向量
	return returnVect

"""
函数说明:手写数字分类测试

Parameters:
	无
Returns:
	无

Modify:
	2017-11-14 by Cugtyt 
		* GitHub(https://github.com/Cugtyt) 
		* Email(cugtyt@qq.com)
		Simplify if condition.
		Remove float(), not need in python3
	2017-03-25
"""
def handwritingClassTest():
	#测试集的Labels
	hwLabels = []
	#返回trainingDigits目录下的文件名
	trainingFileList = listdir('trainingDigits')
	#返回文件夹下文件的个数
	m = len(trainingFileList)
	#初始化训练的Mat矩阵,测试集
	trainingMat = np.zeros((m, 1024))
	#从文件名中解析出训练集的类别
	for i in range(m):
		#获得文件的名字
		fileNameStr = trainingFileList[i]
		#获得分类的数字
		classNumber = int(fileNameStr.split('_')[0])
		#将获得的类别添加到hwLabels中
		hwLabels.append(classNumber)
		#将每一个文件的1x1024数据存储到trainingMat矩阵中
		trainingMat[i] = img2vector('trainingDigits/%s' % (fileNameStr))
	#返回testDigits目录下的文件名
	testFileList = listdir('testDigits')
	#错误检测计数
	errorCount = 0
	#测试数据的数量
	mTest = len(testFileList)
	#从文件中解析出测试集的类别并进行分类测试
	for i in range(mTest):
		#获得文件的名字
		fileNameStr = testFileList[i]
		#获得分类的数字
		classNumber = int(fileNameStr.split('_')[0])
		#获得测试集的1x1024向量,用于训练
		vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
		#获得预测结果
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
		errorCount += classifierResult != classNumber
	print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount/mTest))


"""
函数说明:main函数

Parameters:
	无
Returns:
	无

Modify:
	2017-03-25
"""
if __name__ == '__main__':
	handwritingClassTest()
