from sklearn.feature_extraction.text import CountVectorizer
import jieba

# 文档
documents = [
    "我喜欢吃苹果",
    "我喜欢吃香蕉",
    "我不喜欢吃苹果"
]

# 使用 jieba 进行分词，并将结果拼接为空格分隔的字符串
def cut_documents(docs):
    cut_docs = []
    for doc in docs:
        # jieba.cut 返回生成器，用 ' '.join() 拼接
        cut_docs.append(' '.join(jieba.cut(doc)))
    return cut_docs

# 对全部文档进行分词
cut_documents_list = cut_documents(documents)
print("分词后的文档：", cut_documents_list)

# 创建词袋模型
vectorizer = CountVectorizer()

# 转换（传入分词后的文档列表）
X = vectorizer.fit_transform(cut_documents_list)

# 查看词表（特征名称）
print("\n词表（特征名称）：")
print(vectorizer.get_feature_names_out())

# 查看向量（每个文档的词频统计）
print("\n词频矩阵：")
print(X.toarray())