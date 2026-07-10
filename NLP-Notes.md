# NLP 学习笔记（Natural Language Processing）

> 学习路线：
>
> One-Hot
> ↓
> Bag Of Words
> ↓
> TF-IDF
> ↓
> Word2Vec
> ↓
> RNN
> ↓
> LSTM
> ↓
> Attention
> ↓
> Transformer

---

# 第一章 One-Hot 编码

## 为什么需要 One-Hot

One-Hot编码（独热编码）是一种将类别数据转换为数值数据的一种编码方式 便于机器学习和处理

## 核心思想

用二进制的思想表示类别 每个类别对应一个位置 如果该位置存在则标为1 否则就为0

## 优点

所有类别是平等的 没有大小之分 每个特征相互独立 互不影响 适合大多数机器学习算法

## 缺点

One-Hot只能表示这个词存在 不能表示某一个词的意思 而且如果有一千万个词 会开一个一千万维的数组 导致维度爆炸

## 代码

01_OneHot.py

---

# 第二章 Bag Of Words

## 为什么出现

One-Hot 只能表示一个词是否存在，无法表示一篇文章。

Bag Of Words（词袋模型）把一篇文章表示成每个词出现次数的统计结果。

例如：

我 喜欢 苹果
我 喜欢 香蕉

↓

苹果：1
香蕉：1
喜欢：2
我：2

## 优点

- 实现简单
- 能表示整篇文章
- 常用于文本分类

## 缺点

- 不考虑词语顺序
- 无法表示词义
- 无法区分"我爱你"和"你爱我"

代码：

02_BOW.py

---

# 第三章 TF-IDF

TF-IDF 用来衡量一个词对一篇文章的重要程度 它想回答的问题是 那些词最能代表这篇文章

TF (Term Frequency) 词频 
    表示一个词在当前文章中出现了多少次 描述局部重要性
    TF = 某个词出现的次数 / 文章中所有的词个数
IDF (Inverse Document Frequency) 逆文档频率
    表示一个词是不是很稀有 描述全局重要性
    IDF = log以10为底 然后取 所有文章库中的文章数 / 包含这个词的文章数
    IDF 衡量的是一个词在整个语料库中的稀有程度
    一个词出现在越少的文档中 说明他区分能力就越强 因此IDF越大 反之如果一个词几乎所有文档都有 那它的IDF就越小
eg:
    1000篇文章---语料库 100篇文章包含"非常" 10篇文章包含"经济"
    现在有两篇文章
                文章A(100词) 出现10次 经济
                TF = 10 / 100 = 0.1
                IDF = log 10 (1000 / 10) = 2
                TF-IDF = 0.1 * 2 = 0.2
                文章A(100词) 出现10次 非常
                TF = 10 / 100 = 0.1
                IDF = log 10 (1000 / 10) = 1
                TF-IDF = 0.1 * 2 = 0.1
    经济的 TF-IDF > 非常的 TF-IDF 所以也就是说 "经济" 这个次的重要性要大于 "非常"

为什么要用乘法?
    因为只有同时满足这两个条件 分数才会变高
    TF-IDF使用乘法 是因为只有一个词在当前文章中经常出现且这个词在整个语料库中又非常稀有时 才真正具有代表性
    乘法能够同时体现这两个条件 任何一个条件不满足 最终的得分都会降低

总结：
    一个词出现的次数越多---TF越大
    一个词出现的次数越少---IDF越大
    最终TF-IDF越大---说明这个词越可以代表这篇文章
    
---

# 第四章 Word2Vec

## 为什么出现

Word2Vec 学出来的是稠密向量（Dense Vector） 相比 One-Hot 的稀疏向量（Sparse Vector）能够大幅降低维度并表达语义关系。

One-Hot 和 TF-IDF 只能表示词是否出现，无法表示词义之间的关系，例如"苹果"和"香蕉"在向量空间中完全没有联系。
因此 Google 提出了 Word2Vec，希望让模型根据上下文自动学习词语之间的语义关系

核心思想：
    利用上下文训练神经网络，使语义相近的词在向量空间中距离更近。
    Word2Vec并不知道"苹果是水果"这个知识 他只是不断观察大量文本 如果两个词经常出现在相似的上下文中
    例如"吃、甜、水果"一起出现 那么模型就会把他们的向量不断调整的更接近 最终 具有语义相似的词自然会集聚在一起

## CBOW

CBOW Content -> Word 利用上下文预测中心词单词

## Skip-Gram

Skip-Gram Word -> Content 利用中心词单词预测上下文内容

## Gensim训练

model = Word2Vec(
    sentences,
    # 决定每一个词用多少维度来表示 现在用20维 就是20个数字表示一个词
    vector_size=20,         # 向量维度 为了方便看 设定小一点
    # 决定上下文的范围 等于2 说明当前词的前后各两个词都属于上下文
    window=2,               # 设置上下文窗口
    # 过滤低频词只保留>=min_count的词
    # 值太大会包含罕见词 可能学到噪声词
    # 值太小会过滤掉罕见词 但可能会丢失重要信息
    min_count=1,            # 至少出现一次
    sg=0,                   # 0=CBOW 1=Skip-Gram
    # 训练轮次
    # 轮次太少 欠拟合
    # 轮次太多 过拟合
    epochs=1000
)

## PCA可视化

PCA 用于降维，将高维词向量映射到二维或三维空间，方便可视化。
它会保留数据的大部分信息，但会损失一部分细节，因此降维后的数据不能完全等同于原始数据。

## 词向量运算

词向量也可以做类似的数学运算

## 总结

Word2Vec 第一次让计算机学会了"词义"。

代码：

04_Word2Vec.py

05_Word2Vec_PCA.py

06_Word2Vec_Arithmetic.py

---

# 第五章 RNN

## 为什么出现

Word2Vec 每个词都是独立表示 无法学习词语之间的顺序关系
没有顺序 没有位置感 并不能让模型有记忆 可以按照上传的顺序 区分语序
RNN不像Word2Vec那样独立的处理每一个单词 而是一个词一个词的读取 并把上一个时刻的记忆(Hidden State)
传递给下一时刻 因此RNN可以理解句子的顺序和上下文 RNN 可以像人类一样按顺序一个词一个词的读句子

## Hidden State

RNN的核心就是Hidden State 保存了之前的信息

## 前向传播

用现有的知识和参数，对输入数据做一遍正向的计算推理，得出一个结论（或预测值）
就是每一次输入 = 上一次的记忆 + 当前要读取的内容

## 手写Mini RNN

RNN：
    Hidden state 每一次会随着读入的数据不断更新 因此会得到最终的结果
公式：
     h_t = tanh( Wh * h_(t-1) + Wx * x_t + b )
     当前的记忆 = thanh(Wh * 上一次的记忆 + Wx * 每一个数据的向量 + 偏置)

## 总结

Word2Vec：
  学习"词"的表示（Embedding）
RNN：
  学习"句子"的表示（Sequence）
举例：
  我 爱 你
  你 爱 我
Word2Vec：
  认为都是三个词，几乎无法区分语序
RNN：
  Hidden State 会随着顺序不断更新，
  因此最终得到的结果不同。

代码：

07_Mini_RNN.py

---

# 第六章 LSTM

## 为什么出现

RNN只有短期记忆 如果一句话很长 每次读入的数据有很多的时候 RNN最后可能忘记最初读入的数据 会产生遗忘
因此 出现了LSTM(long short Term Memory) 通过Cell State可以保证有长短期存储记忆

## Cell State

负责长期保存重要信息 避免因为长文本而产生信息丢失

## Forget Gate

遗忘门 并不执行删除操作 而是会输出一个0~1区间的值 用来表示保留多少

## Input Gate

输入门 用来表示新的数据要读入多少

## Output Gate

输出门 用来表示最后真正输出的东西占比有多少

## 手写Mini LSTM

公式：
    cell = forget_gate * cell + input_gate * new_info
    新的记忆 = 旧的记忆 * 保留多少 + 新的输入 * 读取多少(学习多少)
    hidden = output_gate * tanh(cell) 
    最后的输出 = 输出门 × tanh(细胞状态)

## 总结

1.遗忘:旧记忆 x保留率 → 扔掉过时的
2.输入:新知识 x吸收率 → 学进重要的
3.输出:更新后的记忆 x提炼率 → 吐出当前结果

代码：

08_Mini_LSTM.py

---

# 第七章 Attention

## 为什么出现

因为 RNN 和 LSTM 的记忆都是需要依赖Hidden State 和 Cell State 
RNN 容易产生遗忘 虽然LSTM不容易产生遗忘 但是也依旧时一个词一个词的读入数据 速度慢 而且如果距离越来越远 越难记
于是有了Attention 它不依赖历史记忆 而是直接计算出当前词和所有词之间的相关性 在着重关心最相关的信息

## Query

Query (Q): 表示 我要找什么?

## Key

Key (K): 表示 我是什么?

## Value

Value (V): 表示 我真正提供的信息

## Self Attention

自注意力机制 所有词都是在内部互相查询 而不是通过数据库或者互联网查询 而是自己查自己

## Softmax

Attention 会让每一个 Query 与所有 Key 计算相似度（点积），
得到 Attention Score，再经过 Softmax 转换成权重。

## 手写Attention

计算流程：
    每个词的Q去点积所有K → Softmax得权重 → 加权所有V → 生成融合上下文的新向量，传给下一层。

每一个词都会利用自己的 Query 去关注所有的 Key 得到注意力权重 再对所有的词的 Value 做加权求和
生成一个融合了上下信息的新向量 这个新向量不会在当前的 Attention 中再次查询
而是作为下一层 Transformer 的输入进行处理

代码：

09_Self_Attention.py

---

# 第八章 Transformer

## 为什么需要 Position Encoding

Attention 只能知道一个词和谁有关但是不知道谁在前谁在后 所以引入Position Encoding
如果没有Position Encoding 那么" 狗 咬 人 "和 " 人 咬 狗 " 拥有完全相同的词 
只是排列顺序不同 模型无法区分他们 所以必须加入位置编码

## 为什么需要 Multi-Head

Multi-Head Attention 的本质就是为了让多个 Self-Attention并行工作 每一个Head拥有自己独立的Wq Wk Wv
因此他们可以同时学习不同类型的语义关系 所有的Head最后经过Concat拼接后 再通过一层Wo融合 得到最终结果

## Transformer整体流程

输入句子: "我 爱 你"
    ↓
[1. 输入嵌入] 每个词转成向量
    ↓
[2. 位置编码] 给每个词加上位置信息（正弦/余弦）
    ↓
[3. 多头自注意力] 每个词看所有词，算相关性（Q×K→Softmax→×V）
    ↓
[4. 残差连接 + LayerNorm] 加回原始输入 + 归一化
    ↓
[5. 前馈网络 (FFN)] 两层全连接层（MLP），进一步变换
    ↓
[6. 残差连接 + LayerNorm] 再次加回 + 归一化
    ↓
[7. 输出] 编码器输出（如果只有编码器，比如BERT）
        或 送入解码器（如果是GPT/翻译模型）

## Multi-Head Attention

假设：输入维度 d_model = 512，8个Head

Step 1: 每个Head拿到属于自己的Wq, Wk, Wv
    Q_i = X × Wq_i    → 维度: [seq_len, 64]  (512/8=64)
    K_i = X × Wk_i    → 维度: [seq_len, 64]
    V_i = X × Wv_i    → 维度: [seq_len, 64]

Step 2: 每个Head独立算Attention
    head_i = Attention(Q_i, K_i, V_i)  → 维度: [seq_len, 64]

Step 3: 所有Head拼接（Concat）
    concat = [head_1, head_2, ..., head_8]  → 维度: [seq_len, 512]

Step 4: 通过输出权重矩阵 Wo 融合
    output = concat × Wo  → 维度: [seq_len, 512]

## Wo的作用
有两个作用：
    作用1 用来将Concat的高维度向量转换为模型本身需要的维度
    作用2 每一个Head学习不同的关系 Wo负责把多个Head真正的融合成一个新的表示 也就是产出一个最终的向量

## 总结
Multi-Head-Attention 中 就是让多个 Self-Attention 并行工作 每个Head 拥有自己独立的 Wq Wk Wv
因此可以学习不同类型的语义关系 例如人物关系 时间关系 地点关系 动作关系等 
Head越多 模型能够从更多角度理解同一句话 最后通过Concat 和 Wo融合这些信息 
使文本表示更加全面 从而提升模型的表达能力和预测能力
    
### 相比 RNN：

✅ 支持并行计算
RNN 必须一个词一个词处理。
Transformer 可以一次处理整个句子。
---
✅ 更容易捕获长距离依赖
Attention 可以直接计算任意两个词之间的关系。
不会因为距离过远导致信息遗忘。
---
✅ 更容易扩展
可以不断增加 Head、层数和参数。
因此成为 GPT、BERT、Llama、Qwen、DeepSeek 等现代大模型的基础。

代码：

10_Mini_Transformer.py

11_MultiHead_Attention.py

---

# NLP 整体发展总结

One-Hot
↓

让计算机认识词

↓

Bag Of Words

↓

让计算机认识文章

↓

TF-IDF

↓

知道哪些词重要

↓

Word2Vec

↓

学会词义

↓

RNN

↓

学会语序

↓

LSTM

↓

学会长短期记忆

↓

Attention

↓

不用记忆，直接关注

↓

Transformer

↓

并行理解整个句子

# 学习总结

机器学习主要解决的是： 如何利用数据建立预测模型。

NLP 主要解决的是： 如何让计算机理解和生成自然语言。

整个 NLP 的发展过程可以理解为：

One-Hot 让计算机认识单词；

Word2Vec 让计算机理解词义；

RNN 和 LSTM 让计算机理解语序；

Attention 让计算机学会关注重要信息；

Transformer 则综合 Position Encoding 和 Multi-Head Attention，实现了高效、并行的上下文建模，并成为现代大语言模型（GPT、BERT、Llama、Qwen、DeepSeek 等）的基础架构。
