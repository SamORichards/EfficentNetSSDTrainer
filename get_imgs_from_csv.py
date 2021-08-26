from re import split
import shutil
import os

csv = open('output.csv')
for l in csv:
    parts = l.split(',')
    print(parts[1])
    print(os.path.join('./data', parts[0].lower(), parts[1].split('/')[-1]))
    shutil.copy(parts[1], os.path.join('./data', parts[0].lower(), parts[1].split('/')[-1]))