from random import randint, randrange
import subprocess
from tqdm import trange, tqdm
from time import sleep, time
from math import exp, tanh
from random import random
import statistics

inf = 10000000.0

hw = 8
min_n_stones = 4 + 10

def digit(n, r):
    n = str(n)
    l = len(n)
    for i in range(r - l):
        n = '0' + n
    return n

def calc_n_stones(board):
    res = 0
    for elem in board:
        res += int(elem != '.')
    return res

evaluate = subprocess.Popen('main.out'.split(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
sleep(1)

min_depth = 3
max_depth = 15

depth_width = max_depth - min_depth + 1

vhs = [[[] for _ in range(max_depth - min_depth + 1)] for _ in range(4)]
vds = [[[] for _ in range(max_depth - min_depth + 1)] for _ in range(4)]
v0s = [[[] for _ in range(max_depth - min_depth + 1)] for _ in range(4)]

vh_vd = []

mpcd = [0, 1, 0, 1, 2, 3, 2, 3, 4, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 8, 9]

def calc_stones(board):
    res = 0
    for i in board:
        if i != '.':
            res += 1
    return res

def collect_data(num):
    global vhs, vds, vh_vd
    try:
        with open('data/records3/' + digit(num, 7) + '.txt', 'r') as f:
            data = list(f.read().splitlines())
    except:
        print('cannot open')
        return
    #for _ in trange(1000):
    depth = min_depth
    max_num = 2000
    for tt, datum in enumerate(tqdm(data[:max_num])):
        #datum = data[randrange(0, len(data))]
        board, player, _ = datum.split()
        n_stones = calc_n_stones(board)
        if n_stones >= 24:
            depth = tt * depth_width // max_num + min_depth #(depth - min_depth + 1) % depth_width + min_depth
            if depth >= 64 - calc_stones(board):
                continue
            board_proc = player + '\n' + str(mpcd[depth]) + '\n'
            for i in range(hw):
                for j in range(hw):
                    board_proc += board[i * hw + j]
                board_proc += '\n'
            #print(board_proc)
            evaluate.stdin.write(board_proc.encode('utf-8'))
            evaluate.stdin.flush()
            vd = float(evaluate.stdout.readline().decode().strip())
            board_proc = player + '\n' + str(depth) + '\n'
            for i in range(hw):
                for j in range(hw):
                    board_proc += board[i * hw + j]
                board_proc += '\n'
            #print(board_proc)
            evaluate.stdin.write(board_proc.encode('utf-8'))
            evaluate.stdin.flush()
            vh = float(evaluate.stdout.readline().decode().strip())
            #print(score)
            vhs[(n_stones - 24) // 10][depth - min_depth].append(vh)
            vds[(n_stones - 24) // 10][depth - min_depth].append(vd)

for i in range(300, 301):
    collect_data(i)
evaluate.kill()

start_temp = 1000.0
end_temp   = 10.0
def temperature_x(x):
    #return pow(start_temp, 1 - x) * pow(end_temp, x)
    return start_temp + (end_temp - start_temp) * x

def prob(p_score, n_score, strt, now, tl):
    dis = p_score - n_score
    if dis >= 0:
        return 1.0
    return exp(dis / temperature_x((now - strt) / tl))

a = 1.0
b = 0.0

def f(x):
    return a * x + b

vh_vd = [[[vhs[i][j][k] - f(vds[i][j][k]) for k in range(len(vhs[i][j]))] for j in range(len(vhs[i]))] for i in range(len(vhs))]
sd = []
for i in range(len(vh_vd)):
    sd.append([])
    for j in range(len(vh_vd[i])):
        if len(vh_vd[i][j]):
            sd[i].append(round(statistics.stdev(vh_vd[i][j]), 3))
        else:
            sd[i].append(0.0)
for each_sd in sd:
    print(str(each_sd).replace('[', '{').replace(']', '}') + ',')