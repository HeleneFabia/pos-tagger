import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score


class POSDataset(Dataset):
    def __init__(self, texts, targets, targets_unpadded):
        self.texts = texts
        self.targets = targets
        self.targets_unpadded = targets_unpadded
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = torch.tensor(self.texts[idx], dtype=torch.long).unsqueeze(0)
        target = torch.tensor(self.targets[idx], dtype=torch.long)
        target_unpadded = np.array(self.targets_unpadded[idx])
        
        return text, target, target_unpadded
    
    
class Model(nn.Module):
    def __init__(
        self, 
        input_size,
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

    def forward(self, x): # (BS, SL)
        
        print("Input", x.shape) # (bs, sl)
        embedding = self.embedding(x) # (bs, sl) --> (bs, sl, es)
        print("Embedding", embedding.shape)
        output, hidden = self.rnn(embedding) # (bs, sl, es) --> (bs, sl, hd)
        print("RNN Output", output.shape)
        output = output.unsqueeze(1) # (bs, sl, hd) --> (bs, 1, sl, hd) (add channel dim)
        print("TDD Input", output.shape)
        output = self.tdd(output) # (bs, 1, sl, hd) (bs, ch, w, h) --> (bs, os, sl, 1) (bs, ch, w, h)
        print("TDD Output", output.shape)
        output = output.squeeze(-1) # (bs, os, sl) --> postprecessing: argmax(dim=1) --> (bs, sl)
        print("Return", output.shape)

        return output
    
    
def _get_f1_score(targets, preds, classes):
    preds = preds[:len(targets)]
    f1 = f1_score(targets, preds, labels=classes, average="micro")
    return f1
    
    
def train_eval(
    train_dataloader, 
    valid_dataloader, 
    classes,
    model,
    num_epochs,
    learning_rate,
    weight_decay,
    device
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate, 
        weight_decay=weight_decay
        )
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss()
    
    epoch_metrics_train = list()
    epoch_metrics_valid = list()
    best_valid_metric =  0
    
    for epoch in range(self.num_epochs):
        
        print(f"----- Epoch {epoch + 1} ----- ")
        
        # training
        model.train()
        optimizer.zero_grad()
        f1_batch_train = list()
        f1_batch_valid = list()
        
        for idx, batch in enumerate(train_dataloader):
            
            texts = batch[0].to(device)
            targets = batch[1].to(device)
            targets_unpadded = batch[2]
            
            preds = model(texts)
            
            loss = loss_func(preds, targets)
            print(f"Batch Loss: {loss:.4f}")
            loss.backward()
            optimizer.step()
            
            f1 = _get_f1_score(
                preds.cpu().detach().numpy(), 
                targets_unpadded, 
                classes
            )
            f1_batch_train.append(f1)
        
        epoch_metrics_train.append(np.mean(f1_batch_train)) 
        print(f"Training F1 Score: {epoch_metrics_train[-1]:.4f}")
        
        # validation
        model.eval()
        
        with torch.no_grad():
            for idx, batch in enumerate(valid_dataloader):
            
                texts = batch[0].to(device)
                targets = batch[1].to(device)
                targets_unpadded = batch[2]

                preds = model(texts)
                
                loss = loss_func(preds, targets)
                f1 = _get_f1_score(
                preds.cpu().detach().numpy(), 
                targets_unpadded, 
                classes
                )
                f1_batch_valid.append(f1)
        
        epoch_metrics_valid.append(np.mean(f1_batch_valid)) 
        print(f"Validation F1 Score: {epoch_metrics_valid[-1]:.4f}")
        
        if epoch_metrics_valid[-1] > best_valid_metric:
            best_valid_metric = epoch_metrics_valid[-1]
