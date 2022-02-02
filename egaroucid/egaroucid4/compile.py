import subprocess

with open('ai.cpp', 'r') as f:
    ai = f.read()
with open('param_compress.txt', 'r', encoding='utf-8') as f:
    param = f.read()
ai = ai.replace('REPLACE_PARAM_HERE', param)
with open('book.txt', 'r') as f:
    book = f.read()
book_proc = ''
for i in range(0, len(book), 300):
    book_proc += '"' + book[i:min(len(book), i + 300)] + '"\n'
book_proc += ';'
ai = ai.replace('REPLACE_BOOK_HERE', book_proc)
#ai = ai.replace('    ', '')
with open('ai_compress.cpp', 'w', encoding='utf-8') as f:
    f.write(ai)
print('----------------file created----------------')
#cmd = 'g++ ai_compress.cpp -O3 -march=native -fexcess-precision=fast -funroll-loops -flto -mtune=native -o ai.out'
with open('compile_cmd.txt', 'r') as f:
    cmd = f.read()
o = subprocess.run(cmd, shell=True, encoding='utf-8', stderr=subprocess.STDOUT, timeout=None)
print('------------------compiled------------------')
