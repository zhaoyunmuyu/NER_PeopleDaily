'''

 Filename:  make_dataset.py
 Description:  处理并分割数据集，生成文件
 Created:  2022年04月28日 15时06分
 Author:  Li Shao

'''

import pickle
import torch
from torch.utils.data import Dataset
import collections
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
# from tensorflow.keras.preprocessing.sequence import pad_sequences
from config.config import Config

class NERDataset(Dataset):
    def __init__(self, X, Y):
        self.data = [{'x': X[i], 'y': Y[i]} for i in range(X.shape[0])]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# 保存所有实体
def entity2pkl(data_path):
    input_data = open(data_path+'renmin_result.txt','r')
    entities = collections.defaultdict(set)
    for line in input_data.readlines():
        line = line.strip().split()
        tokens = [line[i].split('/')[0] for i in range(len(line))]
        tags = [line[i].split('/')[1] for i in range(len(line))]
        end = 0
        for i in range(len(line)):
            tag = tags[i]
            if tag.startswith("B"):
                begin = i
            elif tag.startswith("E"):
                end = i
                word = ''.join(tokens[begin:end+1])
                label = tag[2:]
                entities[label].add(word)
    with open(data_path+'entities.pkl', 'wb') as out:
        pickle.dump(entities, out)
        # entities['nr'], entities['ns'], entities['nt']    
        config.logger.info("实体文件entities.pkl保存完成")

# 处理分割数据并保存
def data2pkl(data_path):
    datas = list()
    labels = list()
    all_words = []
    tags = set()
    input_data = open(data_path+'renmin_result.txt', 'r')
    # 将标注子句拆分成字列表和对应的标注列表
    for line in input_data.readlines():
        linedata = list()
        linelabel = list()
        line = line.split()
        numNotO = 0
        for word in line:
            word = word.split('/')
            linedata.append(word[0])
            linelabel.append(word[1])
            all_words.append(word[0])
            tags.add(word[1])
            if word[1] != 'O':
                numNotO += 1
        # 只保留不全为O的子句
        if numNotO != 0:
            datas.append(linedata)
            labels.append(linelabel)
    input_data.close()
    config.logger.info("文本序列的数量:{}".format(len(datas)))
    config.logger.info("文本所有单词数:{}".format(len(all_words)))
    #创建词汇表和标签表
    words_count = collections.Counter(all_words).most_common()
    word2id = {word: i for i, (word, _) in enumerate(words_count, 1)}
    word2id["[PAD]"] = 0
    word2id["[unknown]"] = len(word2id)
    id2word = {i: word for word, i in word2id.items()}
    config.logger.info("词汇表大小:{}".format(len(id2word)))
    tag2id = {tag: i+1 for i, tag in enumerate(tags)}
    tag2id['[PAD]'] = 0
    id2tag = {i: tag for tag, i in tag2id.items()}
    config.logger.info("标签表大小:{}".format(len(id2tag)))
    config.logger.info("标签表:{}".format(id2tag))
    # 数据向量化并处理成相同长度
    data_ids = [torch.tensor([word2id[w] for w in line]) for line in datas]
    label_ids = [torch.tensor([tag2id[t] for t in line]) for line in labels]
    # max_len = 60
    # x = pad_sequences(data_ids, maxlen=max_len, padding='post').astype(np.int64)
    # y = pad_sequences(label_ids, maxlen=max_len, padding='post').astype(np.int64)
    x = pad_sequence(data_ids,batch_first=True,padding_value=0)
    y = pad_sequence(label_ids,batch_first=True,padding_value=0)
    # 向量化后拆分训练集、验证集、测试集
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=43,)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train,y_train,test_size=0.2,random_state=43,)
    # 保存数据
    with open(data_path+'vocab.pkl','wb') as out:
        pickle.dump(word2id, out)
        pickle.dump(id2word, out)    
    with open(data_path+'tags.pkl','wb') as out:
        pickle.dump(tag2id, out)
        pickle.dump(id2tag, out) 
    with open(data_path+'data.pkl','wb') as out:
        pickle.dump(x_train, out)
        pickle.dump(y_train, out)
        pickle.dump(x_test, out)
        pickle.dump(y_test, out)
        pickle.dump(x_valid, out)
        pickle.dump(y_valid, out)
    config.logger.info("词汇表、标签、数据文件pkl保存完成")

if __name__ == '__main__':
    config = Config()
    config.logger.info("START:生成训练数据集")
    data2pkl(config.dataset_dir)
    entity2pkl(config.dataset_dir)
    config.logger.info("FINISH:生成训练数据集")

