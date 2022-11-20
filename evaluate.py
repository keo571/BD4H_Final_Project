from sklearn.metrics import classification_report
import numpy as np
import torch


class Eval(object):
    def __init__(self, config):
        self.device = config.device

    def evaluate(self, model, criterion, data_loader, id2ast):
        predict_label = []
        true_label = []
        total_loss = 0.0
        with torch.no_grad():
            model.eval()
            for _, (data, label) in enumerate(data_loader):
                data = data.to(self.device)
                label = label.to(self.device)

                logits = model(data)
                loss = criterion(logits, label)
                total_loss += loss.item() * logits.shape[0]

                _, pred = torch.max(logits, dim=1)  # replace softmax with max function, same impacts
                pred = pred.cpu().detach().numpy().reshape((-1, 1))
                label = label.cpu().detach().numpy().reshape((-1, 1))
                predict_label.append(pred)
                true_label.append(label)
        predict_label = np.concatenate(predict_label, axis=0).reshape(-1).astype(np.int64)
        true_label = np.concatenate(true_label, axis=0).reshape(-1).astype(np.int64)
        eval_loss = total_loss / predict_label.shape[0]

        true_str_label = []
        predict_str_label = []
        for tl in true_label:
            true_str_label.append(id2ast[tl])
        for pl in predict_label:
            predict_str_label.append(id2ast[pl])

        class_dict = classification_report(true_str_label, predict_str_label, zero_division=0, output_dict=True)
        return class_dict, eval_loss, predict_label


