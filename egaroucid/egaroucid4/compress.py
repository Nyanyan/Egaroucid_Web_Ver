import sys

digit_num = 3
#chars = [chr(i) for i in range(33, 127)]
chars = ['!', '#', '$', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~']
ln_char = len(chars)
print(ln_char, ''.join(chars))

if len(sys.argv) < 3:
    print('arg err')
    exit(1)

params = []
for i in range(2, len(sys.argv)):
    with open(sys.argv[i], 'r') as f:
        params.extend([float(i) for i in f.read().splitlines()])

min_params = -min(params)
print(min_params)

ans = ''
for elem in params:
    elem += min_params
    elem *= ln_char
    for _ in range(digit_num):
        dig = int(elem / ln_char)
        ans += chars[dig]
        elem -= dig * ln_char
        elem *= ln_char
print(len(ans))

with open(sys.argv[1], 'w') as f:
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
