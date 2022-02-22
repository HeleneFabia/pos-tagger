import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau


class POSDataset(Dataset):
    def __init__(self, texts, targets):
        self.texts = texts
        self.targets = targets

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = torch.tensor(self.texts[idx], dtype=torch.long)
        target = torch.tensor(self.targets[idx], dtype=torch.long)

        return text, target


class Model(nn.Module):
    def __init__(
            self,
            output_size,
            hidden_dim,
            n_layers,
            embedding_weights,
            embedding_size
    ):
        super(Model, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_weights)
        self.rnn = nn.RNN(embedding_size, hidden_dim, n_layers)
        self.tdd = nn.Conv2d(1, output_size, (1, hidden_dim))

    def forward(self, x):  # (BS, SL)

        # print("Input", x.shape)  # (bs, sl)
        embedding = self.embedding(x)  # (bs, sl) --> (bs, sl, es)
        # print("Embedding", embedding.shape)
        output, hidden = self.rnn(embedding)  # (bs, sl, es) --> (bs, sl, hd)
        # print("RNN Output", output.shape)
        output = output.unsqueeze(1)  # (bs, sl, hd) --> (bs, 1, sl, hd) (add channel dim)
        # print("TDD Input", output.shape)
        output = self.tdd(output)  # (bs, 1, sl, hd) (bs, ch, w, h) --> (bs, os, sl, 1) (bs, ch, w, h)
        # print("TDD Output", output.shape)
        output = output.squeeze(-1)  # (bs, os, sl) --> postprecessing: argmax(dim=1) --> (bs, sl)
        # print("Return", output.shape)

        return output


def train_eval(
        train_dataloader,
        valid_dataloader,
        model,
        num_epochs,
        early_stopping,
        learning_rate,
        weight_decay,
        device
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=10,
        factor=0.5,
        verbose=True
    )

    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()

    best_valid_loss = 10
    early_stopping_count = 0

    for epoch in range(num_epochs):

        print(f"----- Epoch {epoch + 1} ----- ")

        # ----- training

        model.train()
        optimizer.zero_grad()

        train_losses = list()
        valid_losses = list()

        for idx, batch in enumerate(train_dataloader):
            texts = batch[0].to(device)
            targets = batch[1].to(device)

            preds = model(texts)

            loss = loss_func(preds, targets)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        scheduler.step(np.mean(valid_losses))

        print(f"Avg. train loss {np.mean(train_losses):.3f}")

        # ----- validation

        model.eval()

        with torch.no_grad():
            for idx, batch in enumerate(valid_dataloader):
                texts = batch[0].to(device)
                targets = batch[1].to(device)

                preds = model(texts)
                loss = loss_func(preds, targets)
                valid_losses.append(loss.item())

        print(f"Avg. valid loss {np.mean(valid_losses):.3f}")
        if np.mean(valid_losses) < best_valid_loss:
            best_valid_loss = np.mean(valid_losses)
            early_stopping_count = 0
        else:
            early_stopping_count += 1

        if early_stopping_count >= early_stopping:
            break

        return train_losses, valid_losses

