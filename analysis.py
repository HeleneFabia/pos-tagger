import torch

def inference_with_metrics(model, dataloader, classes):

    correct = {cls: 0 for cls in classes[1:]}
    correct["all"] = 0
    total = {cls: 0 for cls in classes[1:]}
    total["all"] = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:

            texts, targets = batch[0], batch[1].squeeze(0)

            preds = model(texts)
            preds = torch.argmax(preds, dim=1).squeeze(0)

            for p, t in zip(preds, targets):
                if t.item() != 0:
                    total["all"] += 1
                    total[t.item()] += 1
                    if p == t:
                        correct["all"] += 1
                        correct[t.item()] += 1

    return correct, total