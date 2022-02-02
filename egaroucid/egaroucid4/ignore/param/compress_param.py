import sys
from math import log
from tqdm import trange
from copy import deepcopy

#chars = [chr(i) for i in range(33, 127)]
all_chars = ['!', '#', '$', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', ' ']
ln_char = 2
prev_num = 4
digit_num = 16
print(ln_char, prev_num, digit_num)

chars = all_chars[:ln_char]
chars_add = all_chars[ln_char:]
print('const int ln_char = ', ln_char, ';', sep='')
print('const string chars = "', ''.join(chars), '";', sep='')

with open('param.txt', 'r') as f:
    params = [float(i) for i in f.read().splitlines()]

min_params = -min(params)
max_params = max(params)
print('// max param', max_params)

avg_err = 0.0
mx_err = 0.0
ans = ''
for elem in params:
    elem += min_params
    for _ in range(prev_num):
        elem /= ln_char
    for _ in range(digit_num):
        dig = int(elem / ln_char)
        ans += chars[dig]
        elem -= dig * ln_char
        elem *= ln_char
    for _ in range(digit_num - prev_num):
        elem /= ln_char
    avg_err += elem
    mx_err = max(mx_err, elem)

print('// len', len(ans))
print('// avg err', avg_err / len(params))
print('// max err', mx_err)

with open('param_compress.txt', 'w') as f:
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