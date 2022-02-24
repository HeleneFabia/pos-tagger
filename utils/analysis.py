import matplotlib.pyplot as plt
import pandas as pd

import torch


def plot_learning_curve(train_losses, valid_losses):
    plt.figure(figsize=(7, 4))
    plt.plot(range(len(train_losses)), train_losses, label="train")
    plt.plot(range(len(valid_losses)), valid_losses, label="valid")
    plt.xticks(list(range(len(train_losses))), labels=list(range(1, len(train_losses) + 1)))
    plt.ylim(bottom=0, top=3)
    plt.legend(loc="upper right")
    plt.title("Loss per epoch for train/valid set")
    plt.show()


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

def get_analysis_dataframe(
        value2pos,
        correct_valid,
        correct_test,
        total_valid,
        total_test
):
    cls_metrics_valid, cls_metrics_test = list(), list()
    cls_total_valid, cls_total_test = list(), list()
    tags = list()
    # create lists with scores and totals per dataset
    for cls, tag in value2pos.items():
        if cls != 0:
            tags.append(tag)
            cls_metrics_test.append(correct_test[cls] / (total_test[cls] + 1e-10))
            cls_metrics_valid.append(correct_valid[cls] / (total_valid[cls] + 1e-10))
            cls_total_test.append(total_test[cls])
            cls_total_valid.append(total_valid[cls])
    # sort list according to test scores
    test_metrics, valid_metrics, test_total, valid_total, tags = zip(*sorted(zip(
        cls_metrics_test,
        cls_metrics_valid,
        cls_total_test,
        cls_total_valid,
        tags
    ), reverse=True))
    # create dataframe
    metrics = pd.DataFrame(data={
        "tag": tags,
        "test_score": test_metrics,
        "valid_score": valid_metrics,
        "test_samples": test_total,
        "valid_samples": valid_total
    })
    return metrics