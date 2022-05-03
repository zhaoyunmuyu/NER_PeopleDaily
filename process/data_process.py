'''

 Filename:  data_process.py
 Description:  处理原始数据
 Created:  2022年04月28日 16时06分
 Author:  Li Shao

'''

import re
from config.config import Config

# 将连续的标注数据转化为单个
def blank_process(data_path):
    config.logger.info("将连续的标注数据转化为单个")
    with open(data_path+'renmin.txt','r') as inp,open(data_path+'renmin2.txt','w') as out:
        for line in inp.readlines():
            line = line.split('  ')
            i = 1
            while i < len(line) -1: 
                # [北京/ns 石景山/ns 发电/vn 总厂/n]nt --> 北京石景山发电总厂/nt
                if line[i][0] == '[':
                    out.write(line[i].split('/')[0][1:])
                    i += 1
                    while i < len(line) - 1 and line[i].find(']') == -1:
                        if line[i] != '':
                            out.write(line[i].split('/')[0])
                        i += 1
                    out.write(line[i].split('/')[0].strip() + '/' +line[i].split('/')[1][-2:] + ' ')
                # 将连续的两个nr姓名单词连接起来
                elif line[i].split('/')[1] == 'nr':
                    word = line[i].split('/')[0]
                    i += 1
                    if i < len(line) - 1 and line[i].split('/')[1] == 'nr':
                        out.write(word + line[i].split('/')[0] + '/nr ')
                    else:
                        out.write(word + '/nr ')
                        continue
                else:
                    out.write(line[i] + ' ')
                i += 1
            out.write('\n')

# 将单词标注转换成字标注，为字的位置+词的标注
def char_process(data_path):
    config.logger.info("将单词标注转换成字标注，为字的位置+词的标注")
    with open(data_path+'renmin2.txt','r') as inp, open(data_path+'renmin3.txt','w') as out:
        for line in inp.readlines():
            line = line.split(' ')
            i = 0
            while i < len(line) - 1:
                if line[i] == '':
                    i += 1
                    continue
                word = line[i].split('/')[0]
                tag = line[i].split('/')[1]
                # 只保留这nr,ns,nt, 其它的标注都转变成'/O'
                if tag == 'nr' or tag == 'ns' or tag == 'nt':
                    out.write(word[0] + "/B_" + tag + " ")
                    for j in word[1:len(word) - 1]:
                        if j != ' ':
                            out.write(j + "/M_" + tag + " ")
                    out.write(word[-1] + "/E_" + tag + " ")
                else:
                    for temp in word:
                        out.write(temp + '/O ')
                i += 1
            out.write('\n')

# 按标点符号拆分成子句，每个子句为一行
def sen_process(data_path):
    config.logger.info("按标点符号拆分成子句，每个子句为一行")
    with open(data_path+'renmin3.txt','r') as inp, open(data_path+'renmin_result.txt','w') as out:
        texts = inp.read()
        sentences = re.split('[，。！？、‘’“”:]/[O]', texts)
        for sentence in sentences:
            if sentence != " ":
                out.write(sentence.strip() + '\n')

if __name__ == '__main__':
    config = Config()
    config.logger.info("START:处理原始数据")
    blank_process(config.dataset_dir)
    char_process(config.dataset_dir)
    sen_process(config.dataset_dir)
    config.logger.info("FINISH:处理原始数据")