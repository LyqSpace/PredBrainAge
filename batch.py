import subprocess
import os

# f = open('unused_list.txt', 'w')
#
# data_list = os.listdir('data/IXI-T1-unused')
# for name in data_list:
#     print(name, file=f)
# f.close()

f = open('unused_list.txt', 'r')

for line in f:
    subprocess.call('rm ./data/tmp/' + line.strip(), shell=True)

f.close()