'''

 Filename:  config.py
 Description:  项目的目录位置及参数设置
 Created:  2022年04月28日 17时24分
 Author:  Li Shao

'''

import os
import sys
import torch
import logging
from datetime import datetime

'''
Config类: 把相关配置写在一个配置类中。
'''
class Config():
    def __init__(self):
        # 目录参数
        self.project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = os.path.join(self.project_dir, 'dataset/')
        self.imgs_dir = os.path.join(self.project_dir, 'imgs')
        self.result_dir = os.path.join(self.project_dir, 'result')
        self.cache_save_dir = os.path.join(self.project_dir, 'cache/')
        if not os.path.exists(self.cache_save_dir):
            os.makedirs(self.cache_save_dir)
        
        # 模型参数
        self.embedding_dim = 100
        self.hidden_dim = 200
        self.vocab_size = 0     # 后续赋值
        self.num_tags = 0       # 后续赋值

        # 训练参数
        self.epochs = 50
        self.batch_size = 512
        self.num_workers = 1
        self.dropout = 0.2
        self.lr = 0.001
        self.weight_decay = 1e-5
        self.device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

        # 日志初始化
        logger_init(log_file_name='log', log_level=logging.WARNING, log_dir=self.cache_save_dir)
        self.logger = logging.getLogger('RunningLogger')
        self.logger.setLevel(logging.INFO)

    def show_config(self):
        for name,value in vars(self).items():
            print(name+":",value)

# 日志初始化函数
def logger_init(log_file_name,log_level,log_dir,only_file=False):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_file_name + '_' + str(datetime.now())[:10] + '.txt')
    formatter = '[%(asctime)s] - %(levelname)s: %(message)s'
    # 只打印到日志文件/控制台与日志同时打印
    if only_file:
        logging.basicConfig(filename=log_path,level=log_level,format=formatter,datefmt='%Y-%m-%d %H:%M:%S')
    else:
        logging.basicConfig(level=log_level,format=formatter,datefmt='%Y-%m-%d %H:%M:%S',
                            handlers=[logging.FileHandler(log_path),      # 输入到日志文件中
                                      logging.StreamHandler(sys.stdout)]) # 输出到控制台