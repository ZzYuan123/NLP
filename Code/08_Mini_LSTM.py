'''
RNN：
只有 Hidden State（短期记忆）
LSTM：
增加了 Cell State（长期记忆）
Hidden State：
负责当前时刻的信息输出。
Cell State：
负责长期保存重要信息，避免长文本中早期信息丢失。
'''
cell_state = ["篮球", "足球", "Java"]
print(f"原来的长期记忆: {cell_state}")

# Forget Gate（遗忘门）
# 不是执行删不删除 而是输出一个0~1区间的值 来确定要不要保留这个数据
# 如果发现: 篮球不重要就删除
forget_word = '篮球'
if forget_word in cell_state:
    cell_state.remove(forget_word)
print(f"Forget Gate后: {cell_state}")

print("😀" * 30)

# Input Gate（输入门）
# 新知识: Python很重要 需要加入长期记忆
new_word = 'Python'
cell_state.append(new_word)
print(f"Input Gate后: {cell_state}")

print("😀" * 30)

# Output Gate（输出门）
# 当前真正输出的真正信息
# 演示输出最后两个重要知识
hidden_state = cell_state[-2 : ]
print(f"Output Gate: {hidden_state}")