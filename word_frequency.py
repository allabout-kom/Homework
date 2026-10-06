import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score


# ======================
# 1. 读取数据
# ======================

train_df = pd.read_csv(
    "data/train.csv"
)

val_df = pd.read_csv(
    "data/val.csv"
)

test_df = pd.read_csv(
    "data/test.csv"
)



print("train:", train_df.shape)
print("val:", val_df.shape)
print("test:", test_df.shape)



X_train_text = train_df["text"]

X_test_text = test_df["text"]


y_train = train_df["label"]

y_test = test_df["label"]



# ======================
# 2. Word Frequency
# ======================


vectorizer = CountVectorizer()



# 建立词表 + 训练集转换

X_train = vectorizer.fit_transform(
    X_train_text
)



# 测试集转换

X_test = vectorizer.transform(
    X_test_text
)



print(
    "Vocabulary size:",
    len(vectorizer.vocabulary_)
)


print(
    "X_train shape:",
    X_train.shape
)



# ======================
# 3. Logistic Regression
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
# 4. Evaluation
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



print("====================")
print("Word Frequency Result")
print("====================")


print(
    "Accuracy:",
    accuracy
)


print(
    "Macro-F1:",
    macro_f1
)