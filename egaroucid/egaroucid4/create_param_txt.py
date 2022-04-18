from tqdm import trange

n_phases = 4
n_patterns = 11
max_evaluate_idx = 59049
max_canput = 30
max_surround = 50
n_add_dense1 = 8
n_all_input = 19
pattern_size = [8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10]


with open('param_pat.txt', 'r') as f:
    pat = [elem for elem in f.read().splitlines()]
print(len(pat))

idx = 0
pat_str = 'const double pattern_arr[n_phases][n_patterns][max_evaluate_idx] = {'
for i in range(n_phases):
    pat_str += '{'
    for j in trange(n_patterns):
        pat_str += '{'
        for k in range(max_evaluate_idx):
            if k < 3 ** pattern_size[j]:
                pat_str += pat[idx]
                idx += 1
                if k < max_evaluate_idx - 1:
                    pat_str += ','
            else:
                pat_str += '-1'
                if k < max_evaluate_idx - 1:
                    pat_str += ','
        pat_str += '}'
        if j < n_patterns - 1:
            pat_str += ','
        pat_str += '\n'
    pat_str += '}'
    if i < n_phases - 1:
        pat_str += ','
    pat_str += '\n'
pat_str += '};\n'



with open('param_add.txt', 'r') as f:
    add = [elem for elem in f.read().splitlines()]
print(len(add))

idx = 0
add_str = 'const double add_arr[n_phases][max_canput * 2 + 1][max_surround + 1][max_surround + 1][n_add_dense1] = {'
for i in range(n_phases):
    add_str += '{'
    for j in trange(max_canput * 2 + 1):
        add_str += '{'
        for k in range(max_surround + 1):
            add_str += '{'
            for l in range(max_surround + 1):
                add_str += '{'
                for m in range(n_add_dense1):
                    add_str += add[idx]
                    idx += 1
                    if m < n_add_dense1 - 1:
                        add_str += ','
                add_str += '}'
                if l < max_surround:
                    add_str += ','
                add_str += '\n'
            add_str += '}'
            if k < max_surround:
                add_str += ','
            add_str += '\n'
        add_str += '}'
        if j < max_canput * 2:
            add_str += ','
        add_str += '\n'
    add_str += '}'
    if i < n_phases - 1:
        add_str += ','
    add_str += '\n'
add_str += '};\n'



with open('param_dense.txt', 'r') as f:
    dense = [elem for elem in f.read().splitlines()]
print(len(dense))

idx = 0
dense_str = 'const double all_dense[n_phases][n_all_input] = {'
bias_str = 'const double all_bias[n_phases] = {'
for i in trange(n_phases):
    dense_str += '{'
    for j in range(n_all_input):
        dense_str += dense[idx]
        idx += 1
        if j < n_all_input - 1:
            dense_str += ','
    dense_str += '}'
    if i < n_phases - 1:
        dense_str += ','
    dense_str += '\n'
    bias_str += dense[idx]
    idx += 1
    if i < n_phases - 1:
        bias_str += ','
dense_str += '};\n'
bias_str += '};\n'

with open('param_compress.txt', 'w') as f:
    f.write(pat_str)
    f.write(add_str)
    f.write(dense_str)
    f.write(bias_str)