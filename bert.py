import pandas as pd
import torch
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from transformers import (
    BertTokenizer,
    BertForSequenceClassification
)

from torch.optim import AdamW

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, f1_score



train_df = pd.read_csv(
    "data/train.csv"
)


test_df = pd.read_csv(
    "data/test.csv"
)



texts_train = train_df["text"].tolist()

texts_test = test_df["text"].tolist()


labels_train = train_df["label"].tolist()

labels_test = test_df["label"].tolist()



# 标签数字化

encoder = LabelEncoder()


y_train = encoder.fit_transform(
    labels_train
)


y_test = encoder.transform(
    labels_test
)



class NYTDataset(Dataset):

    def __init__(
        self,
        texts,
        labels,
        tokenizer
    ):

        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer


    def __len__(self):

        return len(self.texts)


    def __getitem__(self,index):

        text = self.texts[index]


        encoding = self.tokenizer(

            text,

            max_length=64,

            padding="max_length",

            truncation=True,

            return_tensors="pt"

        )


        return {

            "input_ids":
            encoding["input_ids"].squeeze(0),

            "attention_mask":
            encoding["attention_mask"].squeeze(0),

            "labels":
            torch.tensor(
                self.labels[index]
            )

        }




# BERT tokenizer

tokenizer = BertTokenizer.from_pretrained(
    "google-bert/bert-base-uncased"
)



train_dataset = NYTDataset(
    texts_train,
    y_train,
    tokenizer
)



test_dataset = NYTDataset(
    texts_test,
    y_test,
    tokenizer
)



train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)


test_loader = DataLoader(
    test_dataset,
    batch_size=16
)



device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = BertForSequenceClassification.from_pretrained(

    "google-bert/bert-base-uncased",

    num_labels=3

)


model.to(device)


optimizer = AdamW(

    model.parameters(),

    lr=2e-5

)



epochs = 3


print("start training.")


for epoch in range(epochs):

    model.train()

    total_loss = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{epochs}"
    )

    for batch in progress_bar:


        optimizer.zero_grad()


        input_ids = batch["input_ids"].to(device)

        mask = batch["attention_mask"].to(device)

        labels = batch["labels"].to(device)



        output = model(

            input_ids,

            attention_mask=mask,

            labels=labels

        )


        loss = output.loss


        loss.backward()


        optimizer.step()



        total_loss += loss.item()

        progress_bar.set_postfix(
                loss=loss.item()
        )



    print(
        "Epoch:",
        epoch+1,
        "Loss:",
        total_loss
    )


model.eval()


preds=[]

true=[]


with torch.no_grad():


    for batch in test_loader:


        input_ids=batch["input_ids"].to(device)

        mask=batch["attention_mask"].to(device)



        output=model(

            input_ids,

            attention_mask=mask

        )


        prediction=torch.argmax(

            output.logits,

            dim=1

        )


        preds.extend(
            prediction.cpu().numpy()
        )


        true.extend(
            batch["labels"].numpy()
        )



acc=accuracy_score(
    true,
    preds
)


f1=f1_score(
    true,
    preds,
    average="macro"
)



print("================")
print("BERT Result")
print("================")


print(
    "Accuracy:",
    acc
)


print(
    "Macro-F1:",
    f1
)