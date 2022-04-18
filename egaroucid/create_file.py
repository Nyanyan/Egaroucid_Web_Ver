import subprocess

with open('book_w.hpp', 'r') as f:
    book = f.read()
with open('book_compress.txt', 'r', encoding='utf-8') as f:
    param = f.read()
book = book.replace('REPLACE_BOOK_HERE', param)
with open('book.hpp', 'w', encoding='utf-8') as f:
    f.write(book)

with open('evaluate_w.hpp', 'r') as f:
    evaluate = f.read()
with open('param_compress.txt', 'r', encoding='utf-8') as f:
    param = f.read()
evaluate = evaluate.replace('REPLACE_PARAM_HERE', param)
with open('evaluate.hpp', 'w', encoding='utf-8') as f:
    f.write(evaluate)

print('----------------file created----------------')

