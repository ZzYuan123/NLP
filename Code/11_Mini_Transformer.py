'''
真正的Transformer流程
Embedding
      │
      ▼
Position Encoding
      │
      ▼
Embedding + Position
      │
      ▼
生成 Q K V
      │
      ▼
Q × Kᵀ
      │
      ▼
Softmax
      │
      ▼
Attention Weight
      │
      ▼
Weight × V
      │
      ▼
新的词向量

Attention 能知道和谁有关 但是不知道谁在前谁在后 所以如果没有 Position Encoding
那么" 狗 咬 人 "和 " 人 咬 狗 " 拥有完全相同的词 只是排列顺序不同 模型无法区分他们 所以 必须加入位置编码
'''
import numpy as np

# 词向量(Embedding)
# 手动对应 但在真正的Transformer中这些向量是训练出来的
words = ["我", "喜欢", "AI"]
enbedding = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])
print(f"enbedding: {enbedding}")

# Position Encoding
# 自己造一个位置向量
position = np.array([
    [0.1, 0],
    [0.2, 0],
    [0.3, 0]
])
print(f"position: {position}")

# 真正输入Transformer
# 真正输入的是 词的向量 + 位置向量
X = enbedding + position
print(f"\nTransformer输入: {X}")

# 模拟生成 Q K V
Wq = np.array([
    [1,0],
    [0,1]
])
Wk = np.array([
    [1,0],
    [0,1]
])
Wv = np.array([
    [1,0],
    [0,1]
])
# 用输入的X分别乘三个权重矩阵 得到各自的向量
# 只是为了制造出三个矩阵 而不是计算注意力分数
Q = X @ Wq
K = X @ Wk
V = X @ Wv

# Attention Score
# 计算注意力分数
score = Q @ K.T
print(f"\nscore: {score}")

# Softmax
# 将上一步计算的注意力分数转化为概率
def softmax(x):
    # 指数化，让所有数变成正数
    exp = np.exp(x)
    # axis=1：对每一行分别求和
    # keepdims=True：保持维度，便于广播
    return exp / np.sum(exp, axis=1, keepdims=True)
weights = softmax(score)
print(f"\nweight: {weights}")

# Attention 输出
output = weights @ V
# 这里输出的就是融合上下文之后的新的向量
print(f"\noutput: {output}")

# 看看每个词都关注谁
print("\n==============================")
for i in range(len(words)):
    print(f"\n{words[i]} 的关注程度: ")
    for j in range(len(words)):
        print(f"{words[j]} : {weights[i][j]:.3f}")

