from gensim.models import Word2Vec

sentences = [
    # 水果关系
    ["苹果", "香蕉", "水果"],
    ["苹果", "香蕉", "水果"],
    ["苹果", "很好吃"],
    ["香蕉", "很好吃"],
    # 动物关系
    ["老虎", "狮子", "动物"],
    ["老虎", "狮子", "动物"],
    ["老虎", "森林"],
    ["狮子", "森林"],
    # 性别关系（模拟）
    ["男人", "女人", "人类"],
    ["男人", "女人", "人类"],
    ["爸爸", "妈妈", "父母"],
    ["哥哥", "姐姐", "兄弟姐妹"],

]
model = Word2Vec(
    sentences,
    vector_size=20,
    window=2,
    min_count=1,
    sg=1,
    epochs=3000
)
# 向量运算
result = (
        model.wv["男人"]
        -
        model.wv["爸爸"]
        +
        model.wv["妈妈"]
)
print(
    model.wv.most_similar(
        [result],
        topn=5
    )
)
