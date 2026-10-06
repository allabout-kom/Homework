import pandas as pd
import numpy as np

from nltk import word_tokenize

from gensim.models import Word2Vec

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score



# ==========================
# 1. 读取数据
# ==========================


# AG News
ag_df = pd.read_csv(
    "data/ag.csv"
)


# NYT
train_df = pd.read_csv(
    "data/train.csv"
)

test_df = pd.read_csv(
    "data/test.csv"
)



print("AG News:")
print(ag_df.shape)


print("NYT:")
print(
    train_df.shape,
    test_df.shape
)



# ==========================
# 2. 文本分词
# ==========================


def tokenize(text):

    words = word_tokenize(
        text.lower()
    )

    return words



# ==========================
# 3. 使用AG News训练Word2Vec
# ==========================


print("Preparing AG News corpus...")


ag_sentences = [

    tokenize(text)

    for text in ag_df["text"]

]


print("Training Word2Vec...")


w2v_model = Word2Vec(

    sentences=ag_sentences,

    vector_size=100,   # 100维词向量

    window=5,

    min_count=2,

    workers=4

)



print("Word2Vec training finished")


print(
    "Vocabulary size:",
    len(w2v_model.wv)
)



# ==========================
# 4. NYT文本 -> 文档向量
# ==========================


def document_vector(text):

    words = tokenize(text)


    vectors = []


    for word in words:

        if word in w2v_model.wv:

            vectors.append(
                w2v_model.wv[word]
            )


    # 如果没有找到任何词
    if len(vectors) == 0:

        return np.zeros(100)


    # 对所有词向量求平均
    return np.mean(
        vectors,
        axis=0
    )



# ==========================
# 5. 转换NYT文本
# ==========================


print("Transform NYT train...")


X_train = np.array(

    [
        document_vector(text)

        for text in train_df["text"]

    ]

)



print("Transform NYT test...")


X_test = np.array(

    [
        document_vector(text)

        for text in test_df["text"]

    ]

)



y_train = train_df["label"]

y_test = test_df["label"]



print(
    "Train vector shape:",
    X_train.shape
)

print(
    "Test vector shape:",
    X_test.shape
)

# 应该类似：
# Train vector shape: (9215,100)
# Test vector shape: (1152,100)



# ==========================
# 6. Logistic Regression
# ==========================


model = LogisticRegression(
    max_iter=1000
)


print("Training classifier...")


model.fit(
    X_train,
    y_train
)



# ==========================
# 7. Test Evaluation
# ==========================


test_pred = model.predict(
    X_test
)



test_acc = accuracy_score(
    y_test,
    test_pred
)


test_f1 = f1_score(
    y_test,
    test_pred,
    average="macro"
)



print("================")
print("Test Result")
print("================")


print(
    "Accuracy:",
    test_acc
)


print(
    "Macro-F1:",
    test_f1
)