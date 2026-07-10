import math

# Sigmoid 激活函数
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Tanh 激活函数
def tanh(x):
    return math.tanh(x)

# 长期记忆 (Cell State)
cell = 0
# 短期记忆 (Hidden State)
hidden = 0
# 输入数据
sentences = [2, 3, 1]

# 指定参数
# 保留80%
forget_gate = 0.8
# 加入60%
input_gate = 0.6
# 输出90%
output_gate = 0.9

for i, x in enumerate(sentences):
    print("😀" * 30)
    print(f"读取到第{i + 1}个词: {x}")
    # Forget Gate
    forget = forget_gate
    # Input Gate
    input = input_gate
    # 新信息
    new_info = tanh(x)
    # 更新长期记忆
    # 新的记忆 = 旧的记忆 * 保留多少 + 新的输入 * 学习多少
    cell = forget * cell + input * new_info
    # Output Gate
    output = output_gate
    # 更新输出
    hidden = output * tanh(cell)

    print(f"Forget Gate : {forget}")
    print(f"Input Gate  : {input_gate}")
    print(f"Cell State  : {cell:.4f}")
    print(f"Hidden State: {hidden:.4f}")