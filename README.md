# NER_PeopleDaily
First attempt of Pytorch NER
## 原始数据来自人民日报，形式为：
迈向/v  充满/v  希望/n  的/u  新/a  世纪/n
处理为三类：
nr | 人名
ns | 地名
nt | 机构名
并用以下标识：
B | 词首
M | 词中
E | 词尾
O | 单字
##