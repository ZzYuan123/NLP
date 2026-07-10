'''
             输入 X
                │
        ┌───────┴────────┐
        │                │
      Head1            Head2
        │                │
   Self Attention   Self Attention
        │                │
        └───────┬────────┘
                │
             Concat
                │
          Linear（Wo）
                │
             最终输出

Multi-Head Attention的本质 就是让多个 Self-Attention 并行工作 每个Head 拥有自己独立的 Wq Wk Wv
因此可以学习不同类型的语义关系 所有 Head的输出经过 Concat 拼接后 再通过一个线性层 Wo 融合 得到最终输出
'''
import numpy as np

# Embedding + Position
words = ["我", "喜欢", "AI"]
embedding = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])
position = np.array([
    [0.1,0],
    [0.2,0],
    [0.3,0]
])
X = embedding + position

# Self - Attention 封装函数
def softmax(x):
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)

def attention(Q, K, V):
    scores = Q @ K.T
    weights = softmax(scores)
    output = weights @ V
    return output

# Head1
Wq1 = np.array([
    [1,0],
    [0,1]
])
Wk1 = np.array([
    [1,0],
    [0,1]
])
Wv1 = np.array([
    [1,0],
    [0,1]
])
# 计算Head1 Q K V
Q1 = X @ Wq1
K1 = X @ Wk1
V1 = X @ Wv1
head1 = attention(Q1,K1,V1)
print(f"Head1输出: {head1}")

# Head2
Wq2 = np.array([
    [2,1],
    [1,2]
])
Wk2 = np.array([
    [1,2],
    [2,1]
])
Wv2 = np.array([
    [2,0],
    [0,2]
])
# 计算Head2 Q K V
Q2 = X @ Wq2
K2 = X @ Wk2
V2 = X @ Wv2
head2 = attention(Q2,K2,V2)
print(f"Head2输出: {head2}")

# 将Head1 和 Head2拼接起来
concat = np.concatenate(
    [head1,head2],axis=1
)
print(f"\nConcat: {concat}")

'''
Wo 的作用:
    作用1 把多个Head拼接后的高维向量重新映射回模型需要的维度
    作用2 每一个Head学习不同的关系 Wo负责把多个Head真正融合成一个新的表示

Multi-Head-Attention 中 就是让多个 Self-Attention 并行工作 每个Head 拥有自己独立的 Wq Wk Wv
因此可以学习不同类型的语义关系 例如人物关系 时间关系 地点关系 动作关系等 
Head越多 模型能够从更多角度理解同一句话 最后通过Concat 和 Wo融合这些信息 
使文本表示更加全面 从而提升模型的表达能力和预测能力
'''

Wo = np.array([
    [1,0],
    [0,1],
    [1,0],
    [0,1]
])
output = concat @ Wo

print("\n最终输出：")
print(output)