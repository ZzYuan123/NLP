from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# 设置全局字体为黑体（SimHei）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 'Microsoft YaHei' (微软雅黑)
# 解决负号 '-' 显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False
# 训练数据
sentences = [

["苹果","香蕉","水果","好吃"],
["苹果","水果","甜"],
["香蕉","水果","甜"],
["苹果","可以","吃"],
["香蕉","可以","吃"],

["老虎","狮子","动物"],
["老虎","狮子","森林"],
["老虎","肉食动物"],
["狮子","肉食动物"],

["汽车","车辆","驾驶"],
["汽车","轮子","发动机"],

]

# Word2Vec训练
model = Word2Vec(
    sentences,
    vector_size=20,
    window=2,
    min_count=1,
    sg=1,
    epochs=5000
)
# 取出词向量
words = [
    "苹果",
    "香蕉",
    "老虎",
    "狮子",
    "汽车",
    "水果",
    "动物"
]
vectors = []
for word in words:
    vectors.append(model.wv[word])
# PCA降维
# 不会改变结果 只是吧维度降低 方便绘画
pca = PCA(
    n_components=2
)
result = pca.fit_transform(vectors)
# 画图
plt.figure(figsize=(8,6))
for i, word in enumerate(words):
    x = result[i][0]
    y = result[i][1]
    plt.scatter(x,y)
    plt.text(
        x,
        y,
        word,
        fontsize=12
    )
plt.title("Word2Vec PCA Visualization")
plt.show()