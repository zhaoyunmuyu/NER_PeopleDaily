'''

 Filename:  train.py
 Description:  模型训练主函数
 Created:  2022年05月2日 10时20分
 Author:  Li Shao

'''

import sys
sys.path.append('..')
import torch
import pickle
from config.config import Config
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from model.bilstm_crf import BiLSTM_CRF
from process.make_dataset import NERDataset
from torch.utils.data import DataLoader

class BiLSTM_CRF_Model(object):
    def __init__(self,config:Config):
        with open(config.dataset_dir+'vocab.pkl','rb') as inp:
            self.word2id = pickle.load(inp)
        with open(config.dataset_dir+'tags.pkl','rb') as inp:
            self.tag2id = pickle.load(inp)
            self.id2tag = pickle.load(inp)
        config.vocab_size = len(self.word2id)
        config.num_tags = len(self.tag2id)
        self.model = BiLSTM_CRF(config).to(config.device)
    
    def train(self,train_dataloader,valid_dataloader):
        # criterion = nn.CrossEntropyLoss(ignore_index=0)
        max_norm = 10
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=config.lr,
                                     weight_decay=config.weight_decay)
        for epoch in range(config.epochs):
            self.model.train()
            for index, batch in enumerate(train_dataloader):
                x = batch['x'].to(config.device)
                y = batch['y'].to(config.device)
                mask = (x > 0).to(config.device)
                loss = self.model.log_likelihood(x, y, mask)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(),
                                               max_norm=max_norm)
                optimizer.step()
                if index % 200 == 0:
                    config.logger.info('epoch:%d,batch:%d,loss:%f'%(epoch+1,index,loss.item()))
                else:
                    print('epoch:%d,batch:%d'%(epoch+1,index))
            # 验证损失和精度
            aver_loss = 0
            preds, labels = [], []
            for index, batch in enumerate(valid_dataloader):
                # 验证模式
                self.model.eval()
                val_x = batch['x'].to(config.device) 
                val_y = batch['y'].to(config.device)
                val_mask = (val_x > 0).to(config.device)
                predict = self.model(val_x, val_mask)
                loss = self.model.log_likelihood(val_x, val_y, val_mask)
                aver_loss += loss.item()
                # 统计非0的，也就是真实标签的长度
                leng = []
                for i in val_y.cpu():
                    tmp = []
                    for j in i:
                        if j.item() > 0:
                            tmp.append(j.item())
                    leng.append(tmp)
                for index, i in enumerate(predict):
                    preds += i[:len(leng[index])]
                for index, i in enumerate(val_y.tolist()):
                    labels += i[:len(leng[index])]
            # 损失值与评测指标
            # aver_loss /= (len(valid_dataloader) * 64)
            # precision = precision_score(labels, preds, average='macro')
            # recall = recall_score(labels, preds, average='macro')
            # f1 = f1_score(labels, preds, average='macro')
            report = classification_report(labels, preds)
            config.logger.info(report)
            torch.save(self.model.state_dict(),config.cache_save_dir+'params.pkl')
    
    # 在测试集上评判性能
    def test(self, test_dataloader):
        self.model.load_state_dict(torch.load(config.cache_save_dir+'params.pkl'))
        aver_loss = 0
        preds, labels = [], []
        for index, batch in enumerate(test_dataloader):
            # 验证模式
            self.model.eval()
            val_x = batch['x'].to(config.device)
            val_y = batch['y'].to(config.device)
            val_mask = (val_x > 0).to(config.device)
            predict = self.model(val_x, val_mask)
            loss = self.model.log_likelihood(val_x, val_y, val_mask)
            aver_loss += loss.item()
            # 统计非0标签的长度
            leng = []
            for i in val_y.cpu():
                tmp = []
                for j in i:
                    if j.item() > 0:
                        tmp.append(j.item())
                leng.append(tmp)
            for index, i in enumerate(predict):
                preds += i[:len(leng[index])]
            for index, i in enumerate(val_y.tolist()):
                labels += i[:len(leng[index])]
        # 损失值与评测指标
        # aver_loss /= (len(test_dataloader) * 64)
        # precision = precision_score(labels, preds, average='macro')
        # recall = recall_score(labels, preds, average='macro')
        # f1 = f1_score(labels, preds, average='macro')
        report = classification_report(labels, preds)
        config.logger.info(report)
    
    def parse_tags(self, text, path):
        tags = [self.id2tag[idx] for idx in path]
        begin = 0
        end = 0
        res = []
        for idx, tag in enumerate(tags):
            # 将连续的同类型的字连接起来
            if tag.startswith("B"):
                begin = idx
            elif tag.startswith("E"):
                end = idx
                word = text[begin:end + 1]
                label = tag[2:]
                res.append((word, label))
            elif tag=='O':
                res.append((text[idx], tag))
        return res

    def predict(self, str):
        self.model.load_state_dict(torch.load(config.cache_save_dir+'params.pkl'))
        self.model.eval()
        input_vec = []
        for char in str:
            if char not in self.word2id:
                input_vec.append(self.word2id['[unknown]'])
            else:
                input_vec.append(self.word2id[char])
        sentences = torch.tensor(input_vec).view(1, -1).to(config.device)
        mask = sentences > 0
        paths = self.model(sentences, mask)
        res = self.parse_tags(str, paths[0])
        print(res)
        return res
    
if __name__ == '__main__': 
    config = Config()
    with open(config.dataset_dir+'data.pkl','rb') as inp:
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)
        x_valid = pickle.load(inp)
        y_valid = pickle.load(inp)
    train_dataset = NERDataset(x_train, y_train)
    valid_dataset = NERDataset(x_valid, y_valid)
    test_dataset = NERDataset(x_test, y_test)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 num_workers=config.num_workers)
    model = BiLSTM_CRF_Model(config)
    # config.logger.info('START:模型训练')
    # model.train(train_dataloader,valid_dataloader)
    # config.logger.info('FINISH:模型训练')
    
    config.logger.info('START:模型测试')
    # model.test(test_dataloader)
    text = "冯永祥突发奇想，跑到阿尔及利亚旅行，意外结识了印度人民党的领导"
    model.predict(text)
