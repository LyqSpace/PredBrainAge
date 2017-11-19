import subprocess
import os

# f = open('unused_list.txt', 'w')
#
# data_list = os.listdir('data/IXI-T1-unused')
# for name in data_list:
#     print(name, file=f)
# f.close()

f = open('../data/IXI-T1-unused_list.txt', 'r')

for line in f:
    subprocess.call('rm ../data/IXI-T1/' + line.strip(), shell=True)

f.close()