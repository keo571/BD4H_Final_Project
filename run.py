import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
from config import Config
from utils import WordEmbeddingLoader, AssertionLoader, MyDataLoader
from model import Att_BLSTM
from evaluate import Eval


def print_result(predict_label, id2ast, start_idx):
    with open('predicted_result.txt', 'w', encoding='utf-8') as fw:
        for i in range(0, predict_label.shape[0]):
            fw.write('{}\t{}\n'.format(
                start_idx+i, id2ast[int(predict_label[i])]))


def train(model, criterion, loaders, config, str_labels):
    optimizer = optim.Adadelta(
        model.parameters(), lr=config.lr, weight_decay=config.L2_decay)
    eval_tool = Eval(config)
    max_f1 = -float('inf')

    print(model)
    print('traning model parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('%s :  %s' % (name, str(param.data.shape)))
    print('--------------------------------------')
    print('start to train the model ...')

    for epoch in range(1, config.epoch+1):
        precision_dict = defaultdict(list)
        recall_dict = defaultdict(list)
        f1_dict = defaultdict(list)
        print('Epoch %d' %(epoch))
        for fold, (train_loader, vali_loader) in enumerate(loaders):   
            for data, label in train_loader:
                model.train()
                data = data.to(config.device)
                label = label.to(config.device)

                optimizer.zero_grad()
                logits = model(data)
                loss = criterion(logits, label)
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), clip_value=5)
                optimizer.step()

            _, train_loss, _ = eval_tool.evaluate(model, criterion, train_loader, id2ast)
            class_dict, vali_loss, _ = eval_tool.evaluate(model, criterion, vali_loader, id2ast)

            # result of each fold
            print('Fold %d  train_loss: %.3f | vali_loss: %.3f'% (fold+1, train_loss, vali_loss))

            for label in str_labels:
                scores = class_dict[label]
                precision_dict[label].append(scores['precision'])
                recall_dict[label].append(scores['recall'])
                f1_dict[label].append(scores['f1-score'])

        # result of each epoch
        for label in str_labels:
            precision = np.mean(precision_dict[label])
            recall = np.mean(recall_dict[label])
            f1 = np.mean(f1_dict[label])
            print('Label [%s]  precision: %.3f | recall: %.3f | micro f1: %.4f'% (label, precision, recall, f1))

        present_f1 = np.mean(f1_dict['present'])
        if present_f1 > max_f1:
            max_f1 = present_f1
            torch.save(model.state_dict(), os.path.join(
                config.model_dir, 'model.pkl'))
            print('>>> save models!')
        else:
            print()


def test(model, criterion, test_loader, config, str_labels):
    print('--------------------------------------')
    print('start test ...')

    model.load_state_dict(torch.load(
        os.path.join(config.model_dir, 'model.pkl')))
    eval_tool = Eval(config)
    class_dict, _, predict_label = eval_tool.evaluate(model, criterion, test_loader, id2ast)
    
    for label in str_labels:
        scores = class_dict[label]
        print('Label [%s]  precision: %.3f | recall: %.3f | micro f1: %.4f'
            % (label, scores['precision'], scores['recall'], scores['f1-score']))
    
    return predict_label


if __name__ == '__main__':
    config = Config()
    print('--------------------------------------')
    print('some config:')
    config.print_config()

    print('--------------------------------------')
    print('start to load data ...')
    word2id, word_vec = WordEmbeddingLoader(config).random_embedding('data/BID_PH.json')
    ast2id, id2ast, class_num = AssertionLoader(config).get_assertion()
    myloader = MyDataLoader(ast2id, word2id, config)
    loaders = myloader.get_train_vali('BID_PH.json')
    test_loader = myloader.get_test('UPMC.json')
    print('finish!')

    print('--------------------------------------')
    model = Att_BLSTM(word_vec=word_vec, class_num=class_num, config=config)
    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()
    str_labels = ast2id.keys()

    if config.mode == 1:  # train mode
        train(model, criterion, loaders, config, str_labels)

    predict_label = test(model, criterion, test_loader, config, str_labels)
    print_result(predict_label, id2ast, start_idx=7074)