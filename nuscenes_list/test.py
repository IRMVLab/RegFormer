import random
from random import randint

oldf = open('val.txt', 'r')  # 要被抽取的文件dataset.txt，共5000行
newf = open('random_val.txt', 'w')  # 抽取的2000行写入randomtext.txt

resultList = random.sample(range(0, 5753), 575)  # sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素

lines = oldf.readlines()
for i in resultList:
    newf.write(lines[i])
oldf.close()
newf.close()
