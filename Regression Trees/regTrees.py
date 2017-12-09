#-*- coding:utf-8 -*-
import numpy as np

def loadDataSet(fileName):
    """
    函数说明:加载数据
    Parameters:
        fileName - 文件名
    Returns:
		dataMat - 数据矩阵
    Website:
        http://www.cuijiahua.com/
    Modify:
        2017-12-09
    """
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))					#转化为float类型
        dataMat.append(fltLine)
    return dataMat

if __name__ == '__main__':
	print(loadDataSet('ex00.txt'))