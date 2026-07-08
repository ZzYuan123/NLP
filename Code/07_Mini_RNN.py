'''
RNN 的核心就是 Hidden State。
Hidden State 保存了之前的信息，如果每次都重新置为 0，
那么模型就无法利用历史信息，退化成每个词独立处理，与 Word2Vec 类似，只能学习单词，不能理解句子。

'''

import math

# 【核心大白话总结】：
# 1. Word2Vec 只能认识孤立的词，无法理解“我爱你”和“你爱我”的区别。
# 2. RNN 的诞生就是为了解决【语序】和【句子级记忆】的问题。
# 3. RNN 的本质就是一个 for 循环！每读一个词，都在更新“脑子里的记忆”。
# 4. 记忆的载体叫 Hidden State（隐藏状态），它不断往后传，实现了“边读边记”。

# 1. 模拟一句话（把文本数字化，变成词向量/ Embedding 的简化版）
# 2 -> 我,  3 -> 喜欢,  1 -> AI
sentence = [2, 3, 1]

# 2. 模拟训练好的网络参数（权重和偏置）
# Wh: 记忆权重 —— 决定了“上一时刻的记忆”有多重要
# Wx: 输入权重 —— 决定了“当前读到的词”有多重要
# b : 偏置项  —— 调整计算的基准
Wh = 0.5
Wx = 1.0
b = 0.0
# 3. 初始化隐藏状态（脑子刚开始是空空的，所以是 0）
hidden = 0.0
print("🔥 开始运行原哥的 Mini-RNN 记忆引擎 🔥")
print("=" * 50)
# 4. 开始像人类一样，按顺序一个词一个词地读句子
for i, x in enumerate(sentence):
    print(f"\n👉 第 {i + 1} 步：当前读到的词输入 x = {x}")
    print(f"   [此时脑子里的旧记忆 hidden] = {hidden:.4f}")

    # 【最核心的数学逻辑 - 纯手算公式】：
    # h_t = tanh( Wh * h_(t-1) + Wx * x_t + b )
    #
    # ⚠️ 注意点（初学者必踩坑）：
    # 下一步传入的不是算出来的“原始数字”，而是经过 tanh 激活函数压缩后的值！
    # tanh 会把任意庞大的数字压缩到 -1 ~ 1 之间，防止数字滚雪球一样炸掉。
    # 混合“旧记忆”和“新输入”，并用 tanh 激活
    raw_mix = Wh * hidden + Wx * x + b
    hidden = math.tanh(raw_mix)
    print(f"   [计算过程]：tanh({Wh} * {hidden:.4f} + {Wx} * {x} + {b}) = tanh({raw_mix:.4f})")
    print(f"   [✨ 脑子更新后的新记忆 hidden] = {hidden:.4f}")

print("\n" + "=" * 50)
print("🏆 运行结束！")
print("💡 原哥复习结论：如果把 [2, 3, 1] 改成 [1, 3, 2]，")
print("   因为每次循环扔进 tanh 的旧记忆和新输入顺序换了，最终算出来的 Hidden State 会完全不同！")
print("   这就是 RNN 能够“分辨语序”的终极秘密！")

# 模拟文字
# words = {
#     "我":2,
#     "喜欢":3,
#     "AI":1
# }
# sentences = ["我","喜欢","AI"]
# # 参数
# wh = 0.5
# wx = 1
# b = 0
# # 初始隐藏状态
# hidden = 0
# print("开始计算RNN")
# # RNN循环
# for word in sentences:
#     x = words[word]
#     hidden = math.tanh(
#         wh * hidden +
#         wx * x
#     )
#     print(f"{word:<4} -> Hidden State = {hidden:.4f}")
