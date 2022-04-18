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
import sys

hw = 8
hw2 = 64

char_1byte = [
    '!', '#', '$', '&', "'", '(', ')', '*', 
    '+', ',', '-', '.', '/', '0', '1', '2', 
    '3', '4', '5', '6', '7', '8', '9', ':', 
    ';', '<', '=', '>', '?', '@', 'A', 'B', 
    'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
    '[', ']', '^', '_', '`', 'a', 'b', 'c', 
    'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']
all_chars = []
for i in range(ord('亜'), ord('腕') + 1):
    all_chars.append(chr(i))

with open(sys.argv[1], 'r') as f:
    book = f.read()

book = book.replace(' ', char_1byte[37])

ln_raw = len(book)
print('len raw', ln_raw)

replace_dct = {}

for i in range(hw2):
    for j in range(hw2):
        replace_from = char_1byte[i] + char_1byte[j]
        replace_to = all_chars[i * hw2 + j]
        replace_dct[replace_from] = replace_to
ans = ''
for i in range(0, len(book), 2):
    if i + 2 >= len(book):
        book += char_1byte[37]
    ans += replace_dct[book[i] + book[i + 1]]


print('len', len(ans))



chars = all_chars[:hw2 * hw2]
chars_add = all_chars[hw2 * hw2:]

replace_from_str = ''
replace_nums = []
replace_to_str = ''

set_chars = [i for i in chars]
len_chars_add = 500
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


if input('sure?: ') != 'yes':
    exit()

with open('book_compress.txt', 'w', encoding='utf-8') as f:
    #f.write('const int ln_raw = ' + str(ln_raw) + ';\n')
    #f.write('const string chars = u8"' + ''.join(chars) + '";\n')
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