import os
import json
import torch
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


class WordEmbeddingLoader(object):
    """
    A loader for pre-trained word embedding
    """
    def __init__(self, config):
        self.word_path = config.embedding_path  # path of pre-trained word embedding
        self.word_dim = config.word_dim  # dimension of word embedding

    def random_embedding(self, json_path):
        word2id = {}
        word_vec = []  # wordID to word embedding

        word2id['PAD'] = len(word2id)  # PAD character
        word2id['UNK'] = len(word2id)  # out of vocabulary

        for line in open(json_path, 'r'):
            for word in json.loads(line)['sentence']:
                lword = word.lower()
                if lword not in word2id:
                    word2id[lword] = len(word2id)
                    word_vec.append(np.random.rand(self.word_dim).astype(np.float32))
        
        word_vec = np.stack(word_vec)
        vec_mean, vec_std = word_vec.mean(), word_vec.std()
        special_emb = np.random.normal(vec_mean, vec_std, (2, self.word_dim))
        special_emb[0] = 0  # <pad> is initialize as zero

        word_vec = np.concatenate((special_emb, word_vec), axis=0)
        word_vec = word_vec.astype(np.float32).reshape(-1, self.word_dim)
        word_vec = torch.from_numpy(word_vec)

        return word2id, word_vec

    def load_embedding(self):
        word2id = {}  # word to wordID
        word_vec = []  # wordID to word embedding

        word2id['PAD'] = len(word2id)  # PAD character
        word2id['UNK'] = len(word2id)  # out of vocabulary
        word2id['<c>'] = len(word2id)
        word2id['</c>'] = len(word2id)

        with open(self.word_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip().split()
                if len(line) != self.word_dim + 1:
                    continue
                word2id[line[0]] = len(word2id)
                word_vec.append(np.asarray(line[1:], dtype=np.float32))

        word_vec = np.stack(word_vec)
        vec_mean, vec_std = word_vec.mean(), word_vec.std()
        special_emb = np.random.normal(vec_mean, vec_std, (4, self.word_dim))
        special_emb[0] = 0  # <pad> is initialize as zero

        word_vec = np.concatenate((special_emb, word_vec), axis=0)
        word_vec = word_vec.astype(np.float32).reshape(-1, self.word_dim)
        word_vec = torch.from_numpy(word_vec)
        return word2id, word_vec


class AssertionLoader(object):
    def __init__(self, config):
        self.data_dir = config.data_dir
    
    def __load_assertion(self):
        assertion_file = os.path.join(self.data_dir, 'ast2id.txt')
        ast2id = {}
        id2ast = {}
        with open(assertion_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                assertion, idx = line.strip().split()
                idx = int(idx)
                ast2id[assertion] = idx
                id2ast[idx] = assertion
        return ast2id, id2ast, len(ast2id)

    def get_assertion(self):
        return self.__load_assertion()


class MyDataset(Dataset):
    def __init__(self, filename, ast2id, word2id, config):
        self.filename = filename
        self.ast2id = ast2id
        self.word2id = word2id
        self.max_len = config.max_len
        self.data_dir = config.data_dir
        self.dataset, self.label = self.__load_data()

    def __symbolize_sentence(self, sentence):
        """
        sentence is a list of words (strings).
        """
        mask = [1] * len(sentence)
        words = []
        length = min(self.max_len, len(sentence))
        mask = mask[:length]

        for i in range(length):
            words.append(self.word2id.get(sentence[i].lower(), self.word2id['UNK']))

        if length < self.max_len:
            for i in range(length, self.max_len):
                mask.append(0)  # 'PAD' mask is zero
                words.append(self.word2id['PAD'])

        unit = np.asarray([words, mask], dtype=np.int64)
        unit = np.reshape(unit, newshape=(1, 2, self.max_len))
        return unit

    def __load_data(self):
        data_file_path = os.path.join(self.data_dir, self.filename)
        data =[]
        labels = []
        with open(data_file_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = json.loads(line.strip())
                label = line['assertion']
                label_idx = self.ast2id[label]
                sentence = line['sentence']

                one_sentence = self.__symbolize_sentence(sentence)
                data.append(one_sentence)
                labels.append(label_idx)
        return data, labels

    def __getitem__(self, index):
        data = self.dataset[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)


class MyDataLoader(object):
    def __init__(self, ast2id, word2id, config):
        self.ast2id = ast2id
        self.word2id = word2id
        self.config = config
        
    def __collate_fn(self, batch):
        data, label = zip(*batch)  # unzip the batch data
        data = list(data)
        label = list(label)
        data = torch.from_numpy(np.concatenate(data, axis=0))
        label = torch.from_numpy(np.asarray(label, dtype=np.int64))
        return data, label

    def get_train_vali(self, filename, k=5, shuffle=True): # used for kFold validation
        dataset = MyDataset(filename, self.ast2id, self.word2id, self.config)
        splits = KFold(n_splits=k, shuffle=shuffle, random_state=42)
        loaders = []
        
        for train_idx, val_idx in splits.split(np.arange(len(dataset))):
            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(
                dataset=dataset, 
                batch_size=self.config.batch_size, 
                num_workers=0, # change to 2
                collate_fn=self.__collate_fn, 
                sampler=train_sampler)
            test_loader = DataLoader(
                dataset=dataset, 
                batch_size=self.config.batch_size, 
                num_workers=0, # change to 2
                collate_fn=self.__collate_fn, 
                sampler=test_sampler)
            loaders.append((train_loader, test_loader))
            
        return loaders

    def get_test(self, filename, shuffle=False):
        dataset = MyDataset(filename, self.ast2id, self.word2id, self.config)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=self.__collate_fn
        )
        return loader


if __name__ == '__main__':
    from config import Config
    config = Config()
    word2id, word_vec = WordEmbeddingLoader(config).random_embedding('data/BID_PH.json')
    ast2id, id2ast, class_num = AssertionLoader(config).get_assertion()
    loaders = MyDataLoader(ast2id, word2id, config).get_train_vali('BID_PH.json')
    test_loader = loaders[0][0] # first fold's train loader

    for step, (data, label) in enumerate(test_loader):
        print(type(data), data.shape)
        print(type(label), label.shape)
        break

