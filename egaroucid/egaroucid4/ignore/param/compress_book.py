'''
with open('book.txt', 'r') as f:
    book = f.read()

with open('book_compress.txt', 'w') as f:
    flag = False
    for i in range(len(book)):
        if i % 300 == 0:
            flag = False
            f.write('"')
        f.write(book[i])
        if i % 300 == 299:
            flag = True
            f.write('"\n')
    if not flag:
        f.write('"')
    f.write(';\n')
'''

from tqdm import trange
from copy import deepcopy

hw = 8
hw2 = 64

all_chars = ['!', '#', '$', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']

print(''.join(all_chars[:hw2]))

char_dict = {}
for i in range(hw2):
    char_dict[all_chars[i]] = i

with open('book.txt', 'r') as f:
    ans = f.read()

print('len raw', len(ans))

with open('book_compress.txt', 'w') as f:
    flag = False
    for i in range(len(ans)):
        if i % 300 == 0:
            flag = False
            f.write('"')
        f.write(ans[i])
        if i % 300 == 299:
            flag = True
            f.write('"\n')
    if not flag:
        f.write('"')
    f.write(';\n')

print('done')