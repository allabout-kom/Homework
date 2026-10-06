import pandas as pd
import numpy as np

from nltk import word_tokenize

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score



# ======================
# 1. 读取数据
# ======================


train_df = pd.read_csv(
    "data/train.csv"
)

test_df = pd.read_csv(
    "data/test.csv"
)


X_train_text = train_df["text"]

X_test_text = test_df["text"]


y_train = train_df["label"]

y_test = test_df["label"]



# ======================
# 2. 加载GloVe
# ======================


def load_glove(path):

    embeddings = {}

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            values = line.split()

            word = values[0]

            vector = np.asarray(
                values[1:],
                dtype="float32"
            )

            embeddings[word] = vector


    return embeddings



print("Loading GloVe...")


glove = load_glove(
    "glove.6B.100d.txt"
)


print(
    "Words:",
    len(glove)
)



# ======================
# 3. 文本 -> 文档向量
# ======================


def document_vector(text):

    words = word_tokenize(
        text.lower()
    )


    vectors = []


    for word in words:

        if word in glove:

            vectors.append(
                glove[word]
            )


    # 如果一个词都没有找到
    if len(vectors)==0:

        return np.zeros(100)


    # 求平均
    return np.mean(
        vectors,
        axis=0
    )



# ======================
# 4. 转换数据
# ======================


print("Transform train...")


X_train = np.array(
    [
        document_vector(text)
        for text in X_train_text
    ]
)


print("Transform test...")


X_test = np.array(
    [
        document_vector(text)
        for text in X_test_text
    ]
)



print(
    X_train.shape
)

# 应该:
# (9215,100)



# ======================
# 5. Logistic Regression
# ======================


model = LogisticRegression(
    max_iter=1000
)


print("Training...")


model.fit(
    X_train,
    y_train
)



# ======================
# 6. Evaluation
# ======================


pred = model.predict(
    X_test
)



accuracy = accuracy_score(
    y_test,
    pred
)


macro_f1 = f1_score(
    y_test,
    pred,
    average="macro"
)



print("================")
print("GloVe Result")
print("================")


print(
    "Accuracy:",
    accuracy
)


print(
    "Macro-F1:",
    macro_f1
)