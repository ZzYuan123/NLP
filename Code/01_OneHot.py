''''
One-Hot编码（独热编码）是一种将类别数据转换为数值数据的编码方式，便于机器学习算法处理。
核心思想
用二进制向量表示类别，每个类别对应一个位置，该位置为1，其他位置为0。
'''
# 例子:
# 城市类别：['北京', '上海', '深圳']
# 北京 → [1, 0, 0]  # 只有北京位置为1
# 上海 → [0, 1, 0]  # 只有上海位置为1
# 深圳 → [0, 0, 1]  # 只有深圳位置为1

'''
如果 北京 = 0, 上海 = 1, 深圳 = 2
问题：模型会认为 0 < 1 < 2，即 北京 < 上海 < 深圳，产生虚假的序关系！
采用One-Hot编码
所有类别平等，没有大小之分 每个特征独立，互不影响 适合大多数机器学习算法

缺点 OneHot只能表示这个词存在 不能表示这个词意思 切会维度爆炸假设一千万个词 就会开一个一千万维的数组
'''

from sklearn.preprocessing import OneHotEncoder
import numpy as np

# 创建数据
words = np.array([
    ["我"],
    ["喜欢"],
    ["吃"],
    ["苹果"]
])
# 创建编码器
encoder = OneHotEncoder()
# 转换
result = encoder.fit_transform(words)
print(result.toarray())
print(encoder.get_feature_names_out())