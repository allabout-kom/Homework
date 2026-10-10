import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score

train_df = pd.read_csv(
    "data/train.csv"
)

val_df = pd.read_csv(
    "data/val.csv"
)

test_df = pd.read_csv(
    "data/test.csv"
)


print(train_df.shape)
print(val_df.shape)
print(test_df.shape)



X_train_text = train_df["text"]

X_val_text = val_df["text"]

X_test_text = test_df["text"]


y_train = train_df["label"]

y_test = test_df["label"]



# Binary BoW

vectorizer = CountVectorizer(
    binary=True
)

X_train = vectorizer.fit_transform(
    X_train_text
)

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




#Logistic Regression

model = LogisticRegression(
    max_iter=1000
)


print("Training...")


model.fit(
    X_train,
    y_train
)




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
print("Binary BoW")
print("================")

print(
    "Accuracy:",
    accuracy
)


print(
    "Macro-F1:",
    macro_f1
)