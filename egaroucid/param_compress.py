import sys
from math import log, tan, atan
from tqdm import trange
from copy import deepcopy

#chars = [chr(i) for i in range(33, 127)]
#all_chars = ['!', '#', '$', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', ' ']
all_chars = []
#for i in range(ord('亜'), ord('腕') + 1):
for i in range(ord('㐀'), ord('鿕')):
    #for i in range(ord('𠀀'), ord('𮯠') + 1):
    all_chars.append(chr(i))
print(all_chars[:10])
ln_char = len(all_chars)
ln_char_d2 = ln_char // 2
prev_num = 0
digit_num = 1
print(ln_char, prev_num, digit_num, len(all_chars))

chars = all_chars[:ln_char]
chars_add = all_chars[ln_char:]
#print('const int ln_char = ', ln_char, ';', sep='')
#print('const string chars = "', ''.join(chars), '";', sep='')

if len(sys.argv) < 2:
    print('arg err')
    exit(1)

with open(sys.argv[1], 'r') as f:
    params = [float(i) for i in f.read().splitlines()]


min_params = -min(params)
max_params = max(params)
width = max_params + min_params + 0.000001
print('// max param', max_params)
print('// min param', min_params)
print('// width', width)

f_weight = 0.000105

def f(x):
    return tan(f_weight * (x - ln_char_d2))

def rev_f(y):
    return round(atan(y) / f_weight + ln_char_d2)

avg_err = 0.0
mx_err = 0.0
avg_err2 = 0.0
mx_err2 = 0.0
ans = ''
for elem in params:
    first_elem = elem
    '''
    elem += min_params
    elem /= width
    elem *= ln_char
    for _ in range(digit_num):
        dig = min(ln_char - 1, round(elem))
        ans += chars[dig]
        elem -= dig
        elem *= ln_char
    for _ in range(digit_num + 1):
        elem /= ln_char
    elem *= width
    '''
    dig = rev_f(elem)
    ans += chars[dig]
    elem -= f(dig)
    elem = abs(elem)
    avg_err += elem
    mx_err = max(mx_err, elem)
    if first_elem != 0.0:
        avg_err2 += abs(elem / first_elem)
        mx_err2 = max(mx_err2, abs(elem / first_elem))

print('// len', len(ans))
print('// avg err', avg_err / len(params))
print('// max err', mx_err)
print('// avg err2', avg_err2 / len(params))
print('// max err2', mx_err2)
'''
replace_from_str = ''
replace_nums = []
replace_to_str = ''

set_chars = [i for i in chars]
len_chars_add = 450#len(chars_add)
#for i in trange(len(chars_add)):
for i in trange(len_chars_add):
    concat_dict = {}
    #for concat_num in range(2, 7):
    concat_num = 2
    for j in range(len(ans) - concat_num):
        concat_char = ans[j:j + concat_num]
        if concat_char in concat_dict:
            concat_dict[concat_char] += concat_num - 1
        else:
            concat_dict[concat_char] = concat_num - 1
    change = None
    mx = 0
    for key in concat_dict.keys():
        if mx < concat_dict[key]:
            mx = concat_dict[key]
            change = key
    ln_change = len(change)
    replace_from_str = change + replace_from_str
    replace_nums.insert(0, ln_change)
    replace_to_str = chars_add[i] + replace_to_str
    #print(i, change, chars_add[i], ln_change, mx)
    ln_ans = len(ans)
    new_ans = ''
    j = 0
    while True:
        if j >= ln_ans - ln_change:
            if j >= ln_ans:
                break
            new_ans += ans[j]
            j += 1
        elif ans[j:j + ln_change] == change:
            new_ans += chars_add[i]
            j += ln_change
        else:
            new_ans += ans[j]
            j += 1
    ans = deepcopy(new_ans)
    set_chars.append(chars_add[i])

print('// len', len(ans))
'''
if input('sure?: ') != 'yes':
    exit()

with open('param_compress.txt', 'w', encoding='utf-8') as f:
    #f.write('const double compress_bias = ' + str(min_params) + ';\n')
    #f.write('const double compress_width = ' + str(width) + ';\n')
    '''
    f.write('const string chars = \n')
    flag = False
    for i in range(len(chars)):
        if i % 300 == 0:
            flag = False
            f.write('u8"')
        f.write(chars[i])
        if i % 300 == 299:
            flag = True
            f.write('"\n')
    if not flag:
        f.write('"')
    f.write(';\n')
    f.write('const string replace_from_str = \n')
    flag = False
    for i in range(len(replace_from_str)):
        if i % 300 == 0:
            flag = False
            f.write('u8"')
        f.write(replace_from_str[i])
        if i % 300 == 299:
            flag = True
            f.write('"\n')
    if not flag:
        f.write('"')
    f.write(';\n')
    f.write('const string replace_to_str = \n')
    flag = False
    for i in range(len(replace_to_str)):
        if i % 300 == 0:
            flag = False
            f.write('u8"')
        f.write(replace_to_str[i])
        if i % 300 == 299:
            flag = True
            f.write('"\n')
    if not flag:
        f.write('"')
    f.write(';\n')
    f.write('const int repair_num = ' + str(len_chars_add) + ';\n')
    '''
    f.write('const string compressed_data = \n')
    flag = False
    for i in range(len(ans)):
        if i % 300 == 0:
            flag = False
            f.write('u8"')
        f.write(ans[i])
        if i % 300 == 299:
            flag = True
            f.write('"\n')
    if not flag:
        f.write('"')
    f.write(';\n')

print('done')