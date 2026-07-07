'''
TF-IDF 用来衡量一个词对一篇文章的重要程度。
它想回答的问题就是："哪些词最能代表这篇文章？"

TF Term Frequency（词频） 表示一个词在当前文章出现了多少次
    TF = 某词出现次数 / 总词数
    TF描述局部重要性

IDF Inverse Document Frequency（逆文档频率） 表示一个词是不是很稀有
    IDF = log以10为底 然后取一个词在所有文章库中的文章/包含这个词的文章
    IDF描述全局重要性
    IDF 衡量的是一个词在整个语料库中的稀有程度。
    一个词出现在越少的文档中，说明它的区分能力越强，因此 IDF 越大；反之，如果一个词几乎所有文档都有，它的 IDF 就越小。

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
    为什么要使用乘法呢？
        因为只有同时满足这两个条件分数才会高
        TF-IDF 使用乘法，是因为一个词只有在“当前文章中经常出现”且“整个语料中比较稀有”时，
        才真正具有代表性。乘法能够同时体现这两个条件，任何一个条件不足，最终得分都会降低。

一个词出现的次数越多 TF越大
一个词出现的次数越少 IDF越大
最终TF-IDF越大 说明这个词越能代表这篇文章
'''

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# 文档
documents = [
    "apple banana apple",
    "banana orange",
    "apple orange orange"
]
# 创建TF-IDF
vectorizer = TfidfVectorizer()
# 转换
X = vectorizer.fit_transform(documents)
# 查看词表
print(f"词表:{vectorizer.get_feature_names_out()}")
# TF_IDF矩阵
print(f"\nTF-IDF: {X.toarray()}")
