# -*-coding:utf-8 -*-
import numpy as np
from bs4 import BeautifulSoup
import random

def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
	"""
	函数说明:从页面读取数据，生成retX和retY列表
	Parameters:
		retX - 数据X
		retY - 数据Y
		inFile - HTML文件
		yr - 年份
		numPce - 乐高部件数目
		origPrc - 原价
	Returns:
		无
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-12-03
	"""
	# 打开并读取HTML文件
	with open(inFile, encoding='utf-8') as f:
		html = f.read()
	soup = BeautifulSoup(html)

	i = 1
	# 根据HTML页面结构进行解析
	currentRow = soup.find_all('table', r = "%d" % i)

	while(len(currentRow) != 0):
		currentRow = soup.find_all('table', r = "%d" % i)
		title = currentRow[0].find_all('a')[1].text
		lwrTitle = title.lower()
		# 查找是否有全新标签
		if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
			newFlag = 1.0
		else:
			newFlag = 0.0

		# 查找是否已经标志出售，我们只收集已出售的数据
		soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
		if len(soldUnicde) == 0:
			print("商品 #%d 没有出售" % i)
		else:
			# 解析页面获取当前价格
			soldPrice = currentRow[0].find_all('td')[4]
			priceStr = soldPrice.text
			priceStr = priceStr.replace('$','') 
			priceStr = priceStr.replace(',','')
			if len(soldPrice) > 1:
				priceStr = priceStr.replace('Free shipping', '')
			sellingPrice = float(priceStr)

			# 去掉不完整的套装价格
			if  sellingPrice > origPrc * 0.5:
				print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
				retX.append([yr, numPce, newFlag, origPrc])
				retY.append(sellingPrice)
		i += 1
		currentRow = soup.find_all('table', r = "%d" % i)

def ridgeRegres(xMat, yMat, lam = 0.2):
	"""
	函数说明:岭回归
	Parameters:
		xMat - x数据集
		yMat - y数据集
		lam - 缩减系数
	Returns:
		ws - 回归系数
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-11-20
	"""
	xTx = xMat.T * xMat
	denom = xTx + np.eye(np.shape(xMat)[1]) * lam
	if np.linalg.det(denom) == 0.0:
		print("矩阵为奇异矩阵,不能求逆")
		return
	ws = denom.I * (xMat.T * yMat)
	return ws

def setDataCollect(retX, retY):
	"""
	函数说明:依次读取六种乐高套装的数据，并生成数据矩阵
	Parameters:
		无
	Returns:
		无
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-12-03
	"""
	scrapePage(retX, retY, './lego/lego8288.html', 2006, 800, 49.99)				#2006年的乐高8288,部件数目800,原价49.99
	scrapePage(retX, retY, './lego/lego10030.html', 2002, 3096, 269.99)				#2002年的乐高10030,部件数目3096,原价269.99
	scrapePage(retX, retY, './lego/lego10179.html', 2007, 5195, 499.99)				#2007年的乐高10179,部件数目5195,原价499.99
	scrapePage(retX, retY, './lego/lego10181.html', 2007, 3428, 199.99)				#2007年的乐高10181,部件数目3428,原价199.99
	scrapePage(retX, retY, './lego/lego10189.html', 2008, 5922, 299.99)				#2008年的乐高10189,部件数目5922,原价299.99
	scrapePage(retX, retY, './lego/lego10196.html', 2009, 3263, 249.99)				#2009年的乐高10196,部件数目3263,原价249.99

def regularize(xMat, yMat):
	"""
	函数说明:数据标准化
	Parameters:
		xMat - x数据集
		yMat - y数据集
	Returns:
		inxMat - 标准化后的x数据集
		inyMat - 标准化后的y数据集
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-12-03
	"""	
	inxMat = xMat.copy()														#数据拷贝
	inyMat = yMat.copy()
	yMean = np.mean(yMat, 0)													#行与行操作，求均值
	inyMat = yMat - yMean														#数据减去均值
	inMeans = np.mean(inxMat, 0)   												#行与行操作，求均值
	inVar = np.var(inxMat, 0)     												#行与行操作，求方差
	# print(inxMat)
	print(inMeans)
	# print(inVar)
	inxMat = (inxMat - inMeans) / inVar											#数据减去均值除以方差实现标准化
	return inxMat, inyMat

def rssError(yArr,yHatArr):
	"""
	函数说明:计算平方误差
	Parameters:
		yArr - 预测值
		yHatArr - 真实值
	Returns:
		
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-12-03
	"""
	return ((yArr-yHatArr)**2).sum()

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
		2017-11-12
	"""
	xMat = np.mat(xArr); yMat = np.mat(yArr).T
	xTx = xMat.T * xMat							#根据文中推导的公示计算回归系数
	if np.linalg.det(xTx) == 0.0:
		print("矩阵为奇异矩阵,不能求逆")
		return
	ws = xTx.I * (xMat.T*yMat)
	return ws

def crossValidation(xArr, yArr, numVal = 10):
	"""
	函数说明:交叉验证岭回归
	Parameters:
		xArr - x数据集
		yArr - y数据集
		numVal - 交叉验证次数
	Returns:
		wMat - 回归系数矩阵
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-11-20
	"""
	m = len(yArr)																		#统计样本个数                       
	indexList = list(range(m))															#生成索引值列表
	errorMat = np.zeros((numVal,30))													#create error mat 30columns numVal rows
	for i in range(numVal):																#交叉验证numVal次
		trainX = []; trainY = []														#训练集
		testX = []; testY = []															#测试集
		random.shuffle(indexList)														#打乱次序
		for j in range(m):																#划分数据集:90%训练集，10%测试集
			if j < m * 0.9: 
				trainX.append(xArr[indexList[j]])
				trainY.append(yArr[indexList[j]])
			else:
				testX.append(xArr[indexList[j]])
				testY.append(yArr[indexList[j]])
		wMat = ridgeTest(trainX, trainY)    											#获得30个不同lambda下的岭回归系数
		for k in range(30):																#遍历所有的岭回归系数
			matTestX = np.mat(testX); matTrainX = np.mat(trainX)						#测试集
			meanTrain = np.mean(matTrainX,0)											#测试集均值
			varTrain = np.var(matTrainX,0)												#测试集方差
			matTestX = (matTestX - meanTrain) / varTrain 								#测试集标准化
			yEst = matTestX * np.mat(wMat[k,:]).T + np.mean(trainY)						#根据ws预测y值
			errorMat[i, k] = rssError(yEst.T.A, np.array(testY))							#统计误差
	meanErrors = np.mean(errorMat,0)													#计算每次交叉验证的平均误差
	minMean = float(min(meanErrors))													#找到最小误差
	bestWeights = wMat[np.nonzero(meanErrors == minMean)]								#找到最佳回归系数

	xMat = np.mat(xArr); yMat = np.mat(yArr).T
	meanX = np.mean(xMat,0); varX = np.var(xMat,0)
	unReg = bestWeights / varX															#数据经过标准化，因此需要还原
	print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % ((-1 * np.sum(np.multiply(meanX,unReg)) + np.mean(yMat)), unReg[0,0], unReg[0,1], unReg[0,2], unReg[0,3]))	

def ridgeTest(xArr, yArr):
	"""
	函数说明:岭回归测试
	Parameters:
		xMat - x数据集
		yMat - y数据集
	Returns:
		wMat - 回归系数矩阵
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-11-20
	"""
	xMat = np.mat(xArr); yMat = np.mat(yArr).T
	#数据标准化
	yMean = np.mean(yMat, axis = 0)						#行与行操作，求均值
	yMat = yMat - yMean									#数据减去均值
	xMeans = np.mean(xMat, axis = 0)					#行与行操作，求均值
	xVar = np.var(xMat, axis = 0)						#行与行操作，求方差
	xMat = (xMat - xMeans) / xVar						#数据减去均值除以方差实现标准化
	numTestPts = 30										#30个不同的lambda测试
	wMat = np.zeros((numTestPts, np.shape(xMat)[1]))	#初始回归系数矩阵
	for i in range(numTestPts):							#改变lambda计算回归系数
		ws = ridgeRegres(xMat, yMat, np.exp(i - 10))	#lambda以e的指数变化，最初是一个非常小的数，
		wMat[i, :] = ws.T 								#计算回归系数矩阵
	return wMat

def useStandRegres():
	"""
	函数说明:使用简单的线性回归
	Parameters:
		无
	Returns:
		无
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-11-12
	"""
	lgX = []
	lgY = []
	setDataCollect(lgX, lgY)
	data_num, features_num = np.shape(lgX)
	lgX1 = np.mat(np.ones((data_num, features_num + 1)))
	lgX1[:, 1:5] = np.mat(lgX)
	ws = standRegres(lgX1, lgY)
	print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % (ws[0],ws[1],ws[2],ws[3],ws[4]))	

def usesklearn():
	"""
	函数说明:使用sklearn
	Parameters:
		无
	Returns:
		无
	Website:
		http://www.cuijiahua.com/
	Modify:
		2017-12-08
	"""
	from sklearn import linear_model
	reg = linear_model.Ridge(alpha = .5)
	lgX = []
	lgY = []
	setDataCollect(lgX, lgY)
	reg.fit(lgX, lgY) 
	print('%f%+f*年份%+f*部件数量%+f*是否为全新%+f*原价' % (reg.intercept_, reg.coef_[0], reg.coef_[1], reg.coef_[2], reg.coef_[3]))	

if __name__ == '__main__':
	usesklearn()