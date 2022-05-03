import sys
sys.path.append('..')
import torch
import pickle
from config.config import Config
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from model.bilstm_crf import BiLSTM_CRF
from process.make_dataset import NERDataset
from torch.utils.data import DataLoader

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
leng = []
for index, batch in enumerate(valid_dataloader):
    val_x = batch['x'].to(config.device) 
    val_y = batch['y'].to(config.device)
    for i in val_y.cpu():
        tmp = []
        print(i)
        for j in i:
            if j.item() > 0:
                tmp.append(j.item())
        print(tmp)
        leng.append(tmp)
        break
    for index, i in enumerate(val_y.tolist()):
        print(i[:len(leng[index])])
        break
    break