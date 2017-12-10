# -*- coding:utf-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(fileName):
	"""
	函数说明:加载数据
	Parameters:
		fileName - 文件名
	Returns:
		xArr - x数据集
		yArr - y数据集
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-11-19
	"""
	numFeat = len(open(fileName).readline().split('\t')) - 1
	xArr = []; yArr = []
	fr = open(fileName)
	for line in fr.readlines():
		lineArr =[]
		curLine = line.strip().split('\t')
		for i in range(numFeat):
			lineArr.append(float(curLine[i]))
		xArr.append(lineArr)
		yArr.append(float(curLine[-1]))
	return xArr, yArr

def lwlr(testPoint, xArr, yArr, k = 1.0):
	"""
	函数说明:使用局部加权线性回归计算回归系数w
	Parameters:
		testPoint - 测试样本点
		xArr - x数据集
		yArr - y数据集
		k - 高斯核的k,自定义参数
	Returns:
		ws - 回归系数
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-11-19
	"""
	xMat = np.mat(xArr); yMat = np.mat(yArr).T
	m = np.shape(xMat)[0]
	weights = np.mat(np.eye((m)))										#创建权重对角矩阵
	for j in range(m):                      							#遍历数据集计算每个样本的权重
		diffMat = testPoint - xMat[j, :]     							
		weights[j, j] = np.exp(diffMat * diffMat.T/(-2.0 * k**2))
	xTx = xMat.T * (weights * xMat)										
	if np.linalg.det(xTx) == 0.0:
		print("矩阵为奇异矩阵,不能求逆")
		return
	ws = xTx.I * (xMat.T * (weights * yMat))							#计算回归系数
	return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):  
	"""
	函数说明:局部加权线性回归测试
	Parameters:
		testArr - 测试数据集,测试集
		xArr - x数据集,训练集
		yArr - y数据集,训练集
		k - 高斯核的k,自定义参数
	Returns:
		ws - 回归系数
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-11-19
	"""
	m = np.shape(testArr)[0]											#计算测试数据集大小
	yHat = np.zeros(m)	
	for i in range(m):													#对每个样本点进行预测
		yHat[i] = lwlr(testArr[i],xArr,yArr,k)
	return yHat

def standRegres(xArr,yArr):
	"""
	函数说明:计算回归系数w
	Parameters:
		xArr - x数据集
		yArr - y数据集
	Returns:
		ws - 回归系数
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-11-19
	"""
	xMat = np.mat(xArr); yMat = np.mat(yArr).T
	xTx = xMat.T * xMat							#根据文中推导的公示计算回归系数
	if np.linalg.det(xTx) == 0.0:
		print("矩阵为奇异矩阵,不能求逆")
		return
	ws = xTx.I * (xMat.T*yMat)
	return ws

def rssError(yArr, yHatArr):
	"""
	误差大小评价函数
	Parameters:
		yArr - 真实数据
		yHatArr - 预测数据
	Returns:
		误差大小
	"""
	return ((yArr - yHatArr) **2).sum()

if __name__ == '__main__':
	abX, abY = loadDataSet('abalone.txt')
	print('训练集与测试集相同:局部加权线性回归,核k的大小对预测的影响:')
	yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
	yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
	yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
	print('k=0.1时,误差大小为:',rssError(abY[0:99], yHat01.T))
	print('k=1  时,误差大小为:',rssError(abY[0:99], yHat1.T))
	print('k=10 时,误差大小为:',rssError(abY[0:99], yHat10.T))

	print('')

	print('训练集与测试集不同:局部加权线性回归,核k的大小是越小越好吗？更换数据集,测试结果如下:')
	yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
	yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
	yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
	print('k=0.1时,误差大小为:',rssError(abY[100:199], yHat01.T))
	print('k=1  时,误差大小为:',rssError(abY[100:199], yHat1.T))
	print('k=10 时,误差大小为:',rssError(abY[100:199], yHat10.T))

	print('')

	print('训练集与测试集不同:简单的线性归回与k=1时的局部加权线性回归对比:')
	print('k=1时,误差大小为:', rssError(abY[100:199], yHat1.T))
	ws = standRegres(abX[0:99], abY[0:99])
	yHat = np.mat(abX[100:199]) * ws
	print('简单的线性回归误差大小:', rssError(abY[100:199], yHat.T.A))