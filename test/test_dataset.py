'''

 Filename:  test_dataset.py
 Description:  测试Dataloader
 Created:  2022年05月2日 10时20分
 Author:  Li Shao

'''

import sys
sys.path.append('..')
import pickle
from process.make_dataset import NERDataset
from config.config import Config
from torch.utils.data import DataLoader

if __name__ == '__main__':
    config = Config()
    with open(config.dataset_dir+'entities.pkl','rb') as inp:
        entities = pickle.load(inp) 
        print(entities['nr'])
        print(entities['ns'])
        print(entities['nt'])
    with open(config.dataset_dir+'data.pkl','rb') as inp:
        x_train = pickle.load(inp)
        y_train = pickle.load(inp)
        x_test = pickle.load(inp)
        y_test = pickle.load(inp)
        x_valid = pickle.load(inp)
        y_valid = pickle.load(inp)
    print("train len:", len(x_train))
    print("test len:", len(x_test))
    print("valid len:", len(x_valid))
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
    test_dataloader  = DataLoader(test_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers)
    