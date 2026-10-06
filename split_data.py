import pandas as pd
from sklearn.model_selection import train_test_split


# 读取数据
df = pd.read_csv("data/nyt.csv")


print(df.head())
print(df.shape)
print(df["label"].value_counts())


# 第一次划分
# train 80%, temp 20%
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True,
    stratify=df["label"]
)


# 第二次划分
# temp里面一半val，一半test
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    shuffle=True,
    stratify=temp_df["label"]
)


print("train:", train_df.shape)
print("val:", val_df.shape)
print("test:", test_df.shape)


print("\ntrain label:")
print(train_df["label"].value_counts())

print("\nval label:")
print(val_df["label"].value_counts())

print("\ntest label:")
print(test_df["label"].value_counts())


# 保存
train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)