'''
Attention 不依赖历史记忆也就是不依赖hidden state和cell state
而是直接计算当前词与所有词之间的相关性 再重点关注最相关的信息 从而获得最有用的信息

Query (Q): 表示 我要找什么?
Key (K): 表示 我是什么?
Value (V): 表示 我真正提供的信息

什么是自注意力机制? (Self - Attention)
    所有词都在句子内部互相查询 不是去查数据库 也不是去查互联网 而是自己查自己

Attention 本质上是在给每一个词重新生成一个包含上下文的新向量
    Q和所有的K计算相似度 -> 得到Attention的权重 -> 用权重加权所有的V -> 得到一个包含上下文的新向量Z
    所以说不是更新Q 而是得到了一个新的Q'
    新的到的Q'会在下一层的transformer中使用

核心:
    每一个词都会利用自己的 Query 去关注所有的 Key 得到注意力权重 再对所有的词的 Value 做加权求和
    生成一个融合了上下信息的新向量 这个新向量不会在当前的 Attention 中再次查询
    而是作为下一层 Transformer 的输入进行处理

和 Word2Vec 的区别:
    Word2Vec 为每个词只学习一个固定的向量 因此 苹果公司 和 苹果水果 中的 苹果 始终是一个表示
    而 Transformer 中 每个词都会通过 Self - Attention 融合上下文信息 生产新的上下文向量
    因此不同语境下的 苹果 会得到不同的表示
'''
import numpy as np

# 准备三个词向量
# 我 喜欢 AI
embeddings = np.array([
    [1, 0],   # 我
    [0, 1],   # 喜欢
    [1, 1]    # AI
])
words = ["我", "喜欢", "AI"]
print(f"原始向量: {embeddings}")

# 生成 Q K V
# 真正Transformer：
# Q = X @ Wq
# K = X @ Wk
# V = X @ Wv

# 为了方便理解
# 这里直接让: Q = K = V = Embedding
Q = embeddings
K = embeddings
V = embeddings

# 计算 Attention Score
# Score = Q × K.T
# 每个词都和所有词计算相关性
# 矩阵相乘
score = np.dot(Q, K.T)
print(f"Attention Score: {score}")

# Softmax Softmax 的作用是把 Attention Score 转换为概率分布
# 使所有注意力权重位于 0~1 之间且总和为 1，从而能够作为加权平均的权重。
def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)

weights = softmax(score)
print(f"权重: {weights}")

# 加权 Value
# 新向量 Z = Attention × V
output = np.dot(weights, V)
print(f"新的词向量: {output}")

# 看看每个词最关注谁
for i in range(len(words)):
    print(f"\n【{words[i]}】关注程度: ")
    for j in range(len(words)):
        print(f"{words[j]} : {weights[i][j]:.3f}")