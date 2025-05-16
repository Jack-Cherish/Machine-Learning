import numpy as np
from collections import Counter

def knn_classify(input_vector, training_data, training_labels, k_neighbors):
    """
    K-近邻分类器（优化版）
    
    参数:
    input_vector -- 待分类的输入向量，形状为(1, n_features)
    training_data -- 训练集特征数组，形状为(n_samples, n_features)
    training_labels -- 训练集标签数组，形状为(n_samples,)
    k_neighbors -- 选择的最近邻居数量
    
    返回:
    预测结果标签
    
    优化点:
    1. 使用向量化运算替代显式循环，提升计算效率
    2. 采用np.argpartition进行部分排序，时间复杂度从O(nlogn)降到O(n)
    3. 使用Counter替代手动计数，提高代码可读性
    """
    # 计算欧氏距离（向量化运算）
    distances = np.linalg.norm(training_data - input_vector, axis=1)
    
    # 使用分区算法获取前k个最小值的索引（比完全排序更高效）
    k_indices = np.argpartition(distances, k_neighbors)[:k_neighbors]
    
    # 统计最近邻标签
    top_k_labels = [training_labels[i] for i in k_indices]
    return Counter(top_k_labels).most_common(1)[0][0]

def file2matrix(filename):
    """
    从文件加载数据集（保持原始路径处理）
    
    参数:
    filename -- 数据文件路径（使用原始相对路径方式）
    
    返回:
    feature_matrix -- 特征矩阵，形状为(n_samples, 3)
    labels -- 标签向量，形状为(n_samples,)
    
    优化点:
    1. 使用字典映射替代多重if判断，提高可维护性
    2. 使用with语句自动管理文件资源
    3. 添加明确的异常处理
    """
    label_map = {
        'didntLike': 1,
        'smallDoses': 2,
        'largeDoses': 3
    }
    
    try:
        # 保持原始文件打开方式（相对路径）
        with open(filename) as file:
            lines = file.readlines()
    except FileNotFoundError as e:
        print(f"错误：文件 '{filename}' 未找到")
        print("请确认：")
        print(f"1. 文件是否存在于当前工作目录: {os.getcwd()}")
        print(f"2. 文件名是否完全匹配（注意大小写）")
        raise e

    # 初始化特征矩阵（更高效的预分配方式）
    feature_matrix = np.zeros((len(lines), 3))
    labels = []
    
    for index, line in enumerate(lines):
        parts = line.strip().split('\t')
        feature_matrix[index] = parts[:3]  # 前3列为特征
        
        # 使用字典映射提高可读性和可维护性
        labels.append(label_map.get(parts[-1], 0))  # 最后1列为标签
        
    return feature_matrix, labels

def autoNorm(dataSet):
    """
    数据归一化（0-1标准化）
    
    参数:
    dataSet -- 原始数据集，形状为(n_samples, n_features)
    
    返回:
    norm_dataset -- 归一化后的数据集
    ranges -- 特征范围数组
    min_vals -- 特征最小值数组
    
    优化点:
    1. 使用向量化运算替代显式循环
    2. 添加防止除零保护机制
    3. 简化归一化计算步骤
    """
    min_vals = dataSet.min(axis=0)
    max_vals = dataSet.max(axis=0)
    ranges = max_vals - min_vals
    
    # 处理常数值特征（防止除零错误）
    ranges[ranges == 0] = 1  # 将零范围特征设为1避免除法错误
    
    # 向量化归一化计算
    norm_dataset = (dataSet - min_vals) / ranges
    return norm_dataset, ranges, min_vals

def classifyPerson():
    """
    用户交互式分类函数
    
    流程说明:
    1. 收集用户输入的特征数据
    2. 加载并预处理训练数据
    3. 对输入数据进行归一化
    4. 使用KNN进行分类预测
    5. 输出可视化结果
    """
    result_map = {
        1: '不喜欢',
        2: '有点喜欢',
        3: '喜欢'
    }
    
    print("请输入以下特征数据：")
    ffMiles = float(input("每年获得的飞行常客里程数: "))
    precentTats = float(input("玩视频游戏所耗时间百分比: "))
    iceCream = float(input("每周消费的冰激淋公升数: "))

    filename = "datingTestSet.txt"
    datingDataMat, datingLabels = file2matrix(filename)
    
    # 数据归一化（使用优化后的实现）
    normMat, ranges, minVals = autoNorm(datingDataMat)
    
    # 构建输入向量并归一化
    inArr = np.array([ffMiles, precentTats, iceCream])
    norminArr = (inArr - minVals) / ranges
    
    # 进行分类预测（使用优化后的KNN）
    classifierResult = knn_classify(norminArr, normMat, datingLabels, 3)
    
    # 结果输出（添加类型转换保障）
    print(f"\n预测结果：你可能会{result_map.get(int(classifierResult), '未知')}这个人")

if __name__ == '__main__':   
    classifyPerson()