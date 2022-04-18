#pragma once
#pragma once
#include <iostream>
#include <fstream>
#include <math.h>
#include "common.hpp"
#include "board.hpp"

using namespace std;

#define n_patterns 12
#define add_max 100
#define max_evaluate_idx 59049

#define step 256
#define step_2 128

#define p31 3
#define p32 9
#define p33 27
#define p34 81
#define p35 243
#define p36 729
#define p37 2187
#define p38 6561
#define p39 19683
#define p310 59049
#define p31m 2
#define p32m 8
#define p33m 26
#define p34m 80
#define p35m 242
#define p36m 728
#define p37m 2186
#define p38m 6560
#define p39m 19682
#define p310m 59048

#define p41 4
#define p42 16
#define p43 64
#define p44 256
#define p45 1024
#define p46 4096
#define p47 16384
#define p48 65536

#define ln_char 27605
#define f_weight 0.000105
#define n_canput_patterns 4
#define n_dense0 14
#define n_dense1 14
#define n_add_patterns 4
#define n_add_dense0 4
#define n_add_dense1 4

uint_fast16_t pow3[11];
unsigned long long stability_edge_arr[n_8bit][n_8bit][2];
int pattern_arr[n_phases][n_patterns][max_evaluate_idx];
int add_arr[n_phases][n_add_patterns][add_max][add_max];

inline int calc_pop(int a, int b, int s){
    return (a / pow3[s - 1 - b]) % 3;
}


inline int calc_rev_idx(int pattern_idx, int pattern_size, int idx){
    int res = 0;
    if (pattern_idx <= 7){
        for (int i = 0; i < pattern_size; ++i)
            res += pow3[i] * calc_pop(idx, i, pattern_size);
    } else if (pattern_idx == 8){
        res += p39 * calc_pop(idx, 5, pattern_size);
        res += p38 * calc_pop(idx, 4, pattern_size);
        res += p37 * calc_pop(idx, 3, pattern_size);
        res += p36 * calc_pop(idx, 2, pattern_size);
        res += p35 * calc_pop(idx, 1, pattern_size);
        res += p34 * calc_pop(idx, 0, pattern_size);
        res += p33 * calc_pop(idx, 9, pattern_size);
        res += p32 * calc_pop(idx, 8, pattern_size);
        res += p31 * calc_pop(idx, 7, pattern_size);
        res += calc_pop(idx, 6, pattern_size);
    } else if (pattern_idx == 9){
        res += p39 * calc_pop(idx, 0, pattern_size);
        res += p38 * calc_pop(idx, 1, pattern_size);
        res += p37 * calc_pop(idx, 2, pattern_size);
        res += p36 * calc_pop(idx, 3, pattern_size);
        res += p35 * calc_pop(idx, 7, pattern_size);
        res += p34 * calc_pop(idx, 8, pattern_size);
        res += p33 * calc_pop(idx, 9, pattern_size);
        res += p32 * calc_pop(idx, 4, pattern_size);
        res += p31 * calc_pop(idx, 5, pattern_size);
        res += calc_pop(idx, 6, pattern_size);
    } else if (pattern_idx == 10){
        res += p38 * calc_pop(idx, 0, pattern_size);
        res += p37 * calc_pop(idx, 3, pattern_size);
        res += p36 * calc_pop(idx, 6, pattern_size);
        res += p35 * calc_pop(idx, 1, pattern_size);
        res += p34 * calc_pop(idx, 4, pattern_size);
        res += p33 * calc_pop(idx, 7, pattern_size);
        res += p32 * calc_pop(idx, 2, pattern_size);
        res += p31 * calc_pop(idx, 5, pattern_size);
        res += calc_pop(idx, 8, pattern_size);
    } else if (pattern_idx == 11){
        res += p39 * calc_pop(idx, 0, pattern_size);
        res += p38 * calc_pop(idx, 5, pattern_size);
        res += p37 * calc_pop(idx, 7, pattern_size);
        res += p36 * calc_pop(idx, 8, pattern_size);
        res += p35 * calc_pop(idx, 9, pattern_size);
        res += p34 * calc_pop(idx, 1, pattern_size);
        res += p33 * calc_pop(idx, 6, pattern_size);
        res += p32 * calc_pop(idx, 2, pattern_size);
        res += p31 * calc_pop(idx, 3, pattern_size);
        res += calc_pop(idx, 4, pattern_size);
    }
    return res;
}

inline double leaky_relu(double x){
    return max(0.01 * x, x);
}

inline double predict(int pattern_size, double in_arr[], double dense0[n_dense0][20], double bias0[n_dense0], double dense1[n_dense1][n_dense0], double bias1[n_dense1], double dense2[n_dense1], double bias2){
    double hidden0[n_dense0], hidden1;
    int i, j;
    for (i = 0; i < n_dense0; ++i){
        hidden0[i] = bias0[i];
        for (j = 0; j < pattern_size * 2; ++j)
            hidden0[i] += in_arr[j] * dense0[i][j];
        hidden0[i] = leaky_relu(hidden0[i]);
    }
    double res = bias2;
    for (i = 0; i < n_dense1; ++i){
        hidden1 = bias1[i];
        for (j = 0; j < n_dense0; ++j)
            hidden1 += hidden0[j] * dense1[i][j];
        hidden1 = leaky_relu(hidden1);
        res += hidden1 * dense2[i];
    }
    return res;
}

inline void unzip_predict(int phase, int pattern, int size, double dense0[n_dense0][20], double bias0[], double dense1[n_dense1][n_dense0], double bias1[], double dense2[], double bias2){
    int i, j, it;
    double arr[20];
    double pred0[max_evaluate_idx], pred1[max_evaluate_idx];
    for (i = 0; i < pow3[size]; ++i){
        it = i;
        for (j = 0; j < size; ++j){
            arr[size - 1 - j] = 0.0;
            arr[2 * size - 1 - j] = 0.0;
            if (it % 3 == black)
                arr[size - 1 - j] = 1.0;
            else if (it % 3 == white)
                arr[2 * size - 1 - j] = 1.0;
            it /= 3;
        }
        pred0[i] = predict(size, arr, dense0, bias0, dense1, bias1, dense2, bias2);
        pred1[calc_rev_idx(pattern, size, i)] = pred0[i];
    }
    for (i = 0; i < pow3[size]; ++i)
        pattern_arr[phase][pattern][i] = round((pred0[i] + pred1[i]) * hw2 * step);
}

inline double add_predict(int elem0, int elem1, double dense0[n_dense0][20], double bias0[n_dense0], double dense1[n_dense1][n_dense0], double bias1[n_dense1], double dense2[n_dense1], double bias2){
    double hidden0[n_add_dense0], hidden1;
    double in_arr[2];
    in_arr[0] = ((double)elem0 - 30.0) / 60.0;
    in_arr[1] = ((double)elem1 - 30.0) / 60.0;
    int i, j;
    for (i = 0; i < n_add_dense0; ++i){
        hidden0[i] = bias0[i];
        for (j = 0; j < 2; ++j)
            hidden0[i] += in_arr[j] * dense0[i][j];
        hidden0[i] = leaky_relu(hidden0[i]);
    }
    double res = bias2;
    for (i = 0; i < n_add_dense1; ++i){
        hidden1 = bias1[i];
        for (j = 0; j < n_add_dense0; ++j)
            hidden1 += hidden0[j] * dense1[i][j];
        hidden1 = leaky_relu(hidden1);
        res += hidden1 * dense2[i];
    }
    return res;
}

inline void unzip_add_predict(int phase, int pattern, double dense0[n_dense0][20], double bias0[], double dense1[n_dense1][n_dense0], double bias1[], double dense2[], double bias2){
    int i, j;
    for (i = 0; i < add_max; ++i){
        for (j = 0; j < add_max; ++j)
            add_arr[phase][pattern][i][j] = round(add_predict(i, j, dense0, bias0, dense1, bias1, dense2, bias2) * hw2 * step);
    }
}

inline double compress_f(int x){
    return tan(f_weight * (double)(x - ln_char / 2));
}

inline double unzip_element_predict(int *idx, int zip_int[]){
    return compress_f(zip_int[(*idx)++]);
}

inline void zip_to_int_predict(int zip_int[], string compressed_data){
    int i, siz, num;
    siz = compressed_data.size() / 3;
    cout << "unzipping " << siz << " elements" << endl;
    for (i = 0; i < siz; ++i){
        num = -406528;
        num += ((int)compressed_data[i * 3] + 128) * 4096;
        num += ((int)compressed_data[i * 3 + 1] + 128) * 64;
        num += (int)compressed_data[i * 3 + 2] + 128;
        zip_int[i] = num;
    }
}

inline void evaluate_func_init(){
    int phase, pattern, i, j;
    constexpr int pattern_sizes[n_patterns] = {8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 9, 10};
    double dense0[n_dense0][20], bias0[n_dense0], dense1[n_dense1][n_dense0], bias1[n_dense1], dense2[n_dense1], bias2;
    int zip_int[100000];
REPLACE_PARAM_HERE
    zip_to_int_predict(zip_int, compressed_data);
    int zip_idx = 0;
    for (phase = 0; phase < n_phases; ++phase){
        for (pattern = 0; pattern < n_patterns; ++pattern){
            for (i = 0; i < n_dense0; ++i){
                for (j = 0; j < pattern_sizes[pattern] * 2; ++j)
                    dense0[i][j] = unzip_element_predict(&zip_idx, zip_int);
            }
            for (i = 0; i < n_dense0; ++i)
                bias0[i] = unzip_element_predict(&zip_idx, zip_int);
            for (i = 0; i < n_dense1; ++i){
                for (j = 0; j < n_dense0; ++j)
                    dense1[i][j] = unzip_element_predict(&zip_idx, zip_int);
            }
            for (i = 0; i < n_dense1; ++i)
                bias1[i] = unzip_element_predict(&zip_idx, zip_int);
            for (i = 0; i < n_dense1; ++i)
                dense2[i] = unzip_element_predict(&zip_idx, zip_int);
            bias2 = unzip_element_predict(&zip_idx, zip_int);
            unzip_predict(phase, pattern, pattern_sizes[pattern], dense0, bias0, dense1, bias1, dense2, bias2);
        }
        for (pattern = 0; pattern < n_canput_patterns; ++pattern){
            for (i = 0; i < n_add_dense0; ++i){
                for (j = 0; j < 2; ++j)
                    dense0[i][j] = unzip_element_predict(&zip_idx, zip_int);
            }
            for (i = 0; i < n_add_dense0; ++i)
                bias0[i] = unzip_element_predict(&zip_idx, zip_int);
            for (i = 0; i < n_add_dense1; ++i){
                for (j = 0; j < n_add_dense0; ++j)
                    dense1[i][j] = unzip_element_predict(&zip_idx, zip_int);
            }
            for (i = 0; i < n_add_dense1; ++i)
                bias1[i] = unzip_element_predict(&zip_idx, zip_int);
            for (i = 0; i < n_add_dense1; ++i)
                dense2[i] = unzip_element_predict(&zip_idx, zip_int);
            bias2 = unzip_element_predict(&zip_idx, zip_int);
            unzip_add_predict(phase, pattern, dense0, bias0, dense1, bias1, dense2, bias2);
        }
    }
    cout << "evaluation function initialized n_param " << zip_idx << endl;
}

string create_line(int b, int w){
    string res = "";
    for (int i = 0; i < hw; ++i){
        if ((b >> i) & 1)
            res += "X";
        else if ((w >> i) & 1)
            res += "O";
        else
            res += ".";
    }
    return res;
}

inline void probably_move_line(int p, int o, int place, int *np, int *no){
    int i, j;
    *np = p | (1 << place);
    for (i = place - 1; i >= 0 && (1 & (o >> i)); --i);
    if (1 & (p >> i)){
        for (j = place - 1; j > i; --j)
            *np ^= 1 << j;
    }
    for (i = place + 1; i < hw && (1 & (o >> i)); ++i);
    if (1 & (p >> i)){
        for (j = place + 1; j < i; ++j)
            *np ^= 1 << j;
    }
    *no = o & ~(*np);
}

int calc_stability_line(int b, int w, int ob, int ow){
    int i, nb, nw, res = 0b11111111;
    res &= b & ob;
    res &= w & ow;
    for (i = 0; i < hw; ++i){
        if ((1 & (b >> i)) == 0 && (1 & (w >> i)) == 0){
            probably_move_line(b, w, i, &nb, &nw);
            res &= calc_stability_line(nb, nw, ob, ow);
            probably_move_line(w, b, i, &nw, &nb);
            res &= calc_stability_line(nb, nw, ob, ow);
        }
    }
    return res;
}

inline void init_evaluation_base() {
    int idx, place, b, w, stab;
    pow3[0] = 1;
    for (idx = 1; idx < 11; ++idx)
        pow3[idx] = pow3[idx- 1] * 3;
    pow3[0] = 1;
    for (idx = 1; idx < 11; ++idx)
        pow3[idx] = pow3[idx- 1] * 3;
    for (b = 0; b < n_8bit; ++b) {
        for (w = b; w < n_8bit; ++w){
            stab = calc_stability_line(b, w, b, w);
            stability_edge_arr[b][w][0] = 0;
            stability_edge_arr[b][w][1] = 0;
            for (place = 0; place < hw; ++place){
                if (1 & (stab >> place)){
                    stability_edge_arr[b][w][0] |= 1ULL << place;
                    stability_edge_arr[b][w][1] |= 1ULL << (place * hw);
                }
            }
            stability_edge_arr[w][b][0] = stability_edge_arr[b][w][0];
            stability_edge_arr[w][b][1] = stability_edge_arr[b][w][1];
        }
    }
}

bool evaluate_init(){
    init_evaluation_base();
    evaluate_func_init();
    return true;
}

inline unsigned long long calc_surround_part(const unsigned long long player, const int dr){
    return (player << dr | player >> dr);
}

inline int calc_surround(const unsigned long long player, const unsigned long long empties){
    return pop_count_ull(empties & (
        calc_surround_part(player & 0b0111111001111110011111100111111001111110011111100111111001111110ULL, 1) | 
        calc_surround_part(player & 0b0000000011111111111111111111111111111111111111111111111100000000ULL, hw) | 
        calc_surround_part(player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL, hw_m1) | 
        calc_surround_part(player & 0b0000000001111110011111100111111001111110011111100111111000000000ULL, hw_p1)
    ));
}

inline void calc_stability(board *b, int *stab0, int *stab1){
    unsigned long long full_h, full_v, full_d7, full_d9;
    unsigned long long edge_stability = 0, black_stability = 0, white_stability = 0, n_stability;
    unsigned long long h, v, d7, d9;
    const unsigned long long black_mask = b->b & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
    const unsigned long long white_mask = b->w & 0b0000000001111110011111100111111001111110011111100111111000000000ULL;
    int bk, wt;
    bk = b->b & 0b11111111;
    wt = b->w & 0b11111111;
    edge_stability |= stability_edge_arr[bk][wt][0] << 56;
    bk = (b->b >> 56) & 0b11111111;
    wt = (b->w >> 56) & 0b11111111;
    edge_stability |= stability_edge_arr[bk][wt][0];
    bk = join_v_line(b->b, 0);
    wt = join_v_line(b->w, 0);
    edge_stability |= stability_edge_arr[bk][wt][1] << 7;
    bk = join_v_line(b->b, 7);
    wt = join_v_line(b->w, 7);
    edge_stability |= stability_edge_arr[bk][wt][1];
    b->full_stability(&full_h, &full_v, &full_d7, &full_d9);
    n_stability = (edge_stability & b->b) | (full_h & full_v & full_d7 & full_d9 & black_mask);
    while (n_stability & ~black_stability){
        black_stability |= n_stability;
        h = (black_stability >> 1) | (black_stability << 1) | full_h;
        v = (black_stability >> hw) | (black_stability << hw) | full_v;
        d7 = (black_stability >> hw_m1) | (black_stability << hw_m1) | full_d7;
        d9 = (black_stability >> hw_p1) | (black_stability << hw_p1) | full_d9;
        n_stability = h & v & d7 & d9 & black_mask;
    }
    n_stability = (edge_stability & b->w) | (full_h & full_v & full_d7 & full_d9 & white_mask);
    while (n_stability & ~white_stability){
        white_stability |= n_stability;
        h = (white_stability >> 1) | (white_stability << 1) | full_h;
        v = (white_stability >> hw) | (white_stability << hw) | full_v;
        d7 = (white_stability >> hw_m1) | (white_stability << hw_m1) | full_d7;
        d9 = (white_stability >> hw_p1) | (white_stability << hw_p1) | full_d9;
        n_stability = h & v & d7 & d9 & white_mask;
    }
    *stab0 = pop_count_ull(black_stability);
    *stab1 = pop_count_ull(white_stability);
}

inline void calc_stability_fast(board *b, int *stab0, int *stab1){
    unsigned long long edge_stability = 0;
    int bk, wt;
    bk = b->b & 0b11111111;
    wt = b->w & 0b11111111;
    edge_stability |= stability_edge_arr[bk][wt][0] << 56;
    bk = (b->b >> 56) & 0b11111111;
    wt = (b->w >> 56) & 0b11111111;
    edge_stability |= stability_edge_arr[bk][wt][0];
    bk = join_v_line(b->b, 0);
    wt = join_v_line(b->w, 0);
    edge_stability |= stability_edge_arr[bk][wt][1] << 7;
    bk = join_v_line(b->b, 7);
    wt = join_v_line(b->w, 7);
    edge_stability |= stability_edge_arr[bk][wt][1];
    *stab0 = pop_count_ull(edge_stability & b->b);
    *stab1 = pop_count_ull(edge_stability & b->w);
}

inline int pop_digit(unsigned long long x, int place){
    return 1 & (x >> place);
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4){
    return pattern_arr[phase_idx][pattern_idx][b_arr[p0] * p34 + b_arr[p1] * p33 + b_arr[p2] * p32 + b_arr[p3] * p31 + b_arr[p4]];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5){
    return pattern_arr[phase_idx][pattern_idx][b_arr[p0] * p35 + b_arr[p1] * p34 + b_arr[p2] * p33 + b_arr[p3] * p32 + b_arr[p4] * p31 + b_arr[p5]];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6){
    return pattern_arr[phase_idx][pattern_idx][b_arr[p0] * p36 + b_arr[p1] * p35 + b_arr[p2] * p34 + b_arr[p3] * p33 + b_arr[p4] * p32 + b_arr[p5] * p31 + b_arr[p6]];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7){
    return pattern_arr[phase_idx][pattern_idx][b_arr[p0] * p37 + b_arr[p1] * p36 + b_arr[p2] * p35 + b_arr[p3] * p34 + b_arr[p4] * p33 + b_arr[p5] * p32 + b_arr[p6] * p31 + b_arr[p7]];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7, const int p8){
    return pattern_arr[phase_idx][pattern_idx][b_arr[p0] * p38 + b_arr[p1] * p37 + b_arr[p2] * p36 + b_arr[p3] * p35 + b_arr[p4] * p34 + b_arr[p5] * p33 + b_arr[p6] * p32 + b_arr[p7] * p31 + b_arr[p8]];
}

inline int pick_pattern(const int phase_idx, const int p, const int pattern_idx, const int b_arr[], const int p0, const int p1, const int p2, const int p3, const int p4, const int p5, const int p6, const int p7, const int p8, const int p9){
    return pattern_arr[phase_idx][pattern_idx][b_arr[p0] * p39 + b_arr[p1] * p38 + b_arr[p2] * p37 + b_arr[p3] * p36 + b_arr[p4] * p35 + b_arr[p5] * p34 + b_arr[p6] * p33 + b_arr[p7] * p32 + b_arr[p8] * p31 + b_arr[p9]];
}

inline int calc_pattern(const int phase_idx, board *b, const int b_arr[]){
    return 
        pick_pattern(phase_idx, b->p, 0, b_arr, 8, 9, 10, 11, 12, 13, 14, 15) + pick_pattern(phase_idx, b->p, 0, b_arr, 1, 9, 17, 25, 33, 41, 49, 57) + pick_pattern(phase_idx, b->p, 0, b_arr, 48, 49, 50, 51, 52, 53, 54, 55) + pick_pattern(phase_idx, b->p, 0, b_arr, 6, 14, 22, 30, 38, 46, 54, 62) + 
        pick_pattern(phase_idx, b->p, 1, b_arr, 16, 17, 18, 19, 20, 21, 22, 23) + pick_pattern(phase_idx, b->p, 1, b_arr, 2, 10, 18, 26, 34, 42, 50, 58) + pick_pattern(phase_idx, b->p, 1, b_arr, 40, 41, 42, 43, 44, 45, 46, 47) + pick_pattern(phase_idx, b->p, 1, b_arr, 5, 13, 21, 29, 37, 45, 53, 61) + 
        pick_pattern(phase_idx, b->p, 2, b_arr, 24, 25, 26, 27, 28, 29, 30, 31) + pick_pattern(phase_idx, b->p, 2, b_arr, 3, 11, 19, 27, 35, 43, 51, 59) + pick_pattern(phase_idx, b->p, 2, b_arr, 32, 33, 34, 35, 36, 37, 38, 39) + pick_pattern(phase_idx, b->p, 2, b_arr, 4, 12, 20, 28, 36, 44, 52, 60) + 
        pick_pattern(phase_idx, b->p, 3, b_arr, 3, 12, 21, 30, 39) + pick_pattern(phase_idx, b->p, 3, b_arr, 4, 11, 18, 25, 32) + pick_pattern(phase_idx, b->p, 3, b_arr, 24, 33, 42, 51, 60) + pick_pattern(phase_idx, b->p, 3, b_arr, 59, 52, 45, 38, 31) + 
        pick_pattern(phase_idx, b->p, 4, b_arr, 2, 11, 20, 29, 38, 47) + pick_pattern(phase_idx, b->p, 4, b_arr, 5, 12, 19, 26, 33, 40) + pick_pattern(phase_idx, b->p, 4, b_arr, 16, 25, 34, 43, 52, 61) + pick_pattern(phase_idx, b->p, 4, b_arr, 58, 51, 44, 37, 30, 23) + 
        pick_pattern(phase_idx, b->p, 5, b_arr, 1, 10, 19, 28, 37, 46, 55) + pick_pattern(phase_idx, b->p, 5, b_arr, 6, 13, 20, 27, 34, 41, 48) + pick_pattern(phase_idx, b->p, 5, b_arr, 8, 17, 26, 35, 44, 53, 62) + pick_pattern(phase_idx, b->p, 5, b_arr, 57, 50, 43, 36, 29, 22, 15) + 
        pick_pattern(phase_idx, b->p, 6, b_arr, 0, 9, 18, 27, 36, 45, 54, 63) + pick_pattern(phase_idx, b->p, 6, b_arr, 7, 14, 21, 28, 35, 42, 49, 56) + 
        pick_pattern(phase_idx, b->p, 7, b_arr, 9, 0, 1, 2, 3, 4, 5, 6, 7, 14) + pick_pattern(phase_idx, b->p, 7, b_arr, 9, 0, 8, 16, 24, 32, 40, 48, 56, 49) + pick_pattern(phase_idx, b->p, 7, b_arr, 49, 56, 57, 58, 59, 60, 61, 62, 63, 54) + pick_pattern(phase_idx, b->p, 7, b_arr, 54, 63, 55, 47, 39, 31, 23, 15, 7) + 
        pick_pattern(phase_idx, b->p, 8, b_arr, 0, 2, 3, 4, 5, 7, 10, 11, 12, 13) + pick_pattern(phase_idx, b->p, 8, b_arr, 0, 16, 24, 32, 40, 56, 17, 25, 33, 41) + pick_pattern(phase_idx, b->p, 8, b_arr, 56, 58, 59, 60, 61, 63, 50, 51, 52, 53) + pick_pattern(phase_idx, b->p, 8, b_arr, 7, 23, 31, 39, 47, 63, 22, 30, 38, 46) + 
        pick_pattern(phase_idx, b->p, 9, b_arr, 0, 9, 18, 27, 1, 10, 19, 8, 17, 26) + pick_pattern(phase_idx, b->p, 9, b_arr, 7, 14, 21, 28, 6, 13, 20, 15, 22, 29) + pick_pattern(phase_idx, b->p, 9, b_arr, 56, 49, 42, 35, 57, 50, 43, 48, 41, 34) + pick_pattern(phase_idx, b->p, 9, b_arr, 63, 54, 45, 36, 62, 53, 44, 55, 46, 37) + 
        pick_pattern(phase_idx, b->p, 10, b_arr, 0, 1, 2, 8, 9, 10, 16, 17, 18) + pick_pattern(phase_idx, b->p, 10, b_arr, 7, 6, 5, 15, 14, 13, 23, 22, 21) + pick_pattern(phase_idx, b->p, 10, b_arr, 56, 57, 58, 48, 49, 50, 40, 41, 42) + pick_pattern(phase_idx, b->p, 10, b_arr, 63, 62, 61, 55, 54, 53, 47, 46, 45) + 
        pick_pattern(phase_idx, b->p, 11, b_arr, 0, 1, 2, 3, 4, 8, 9, 16, 24, 32) + pick_pattern(phase_idx, b->p, 11, b_arr, 7, 6, 5, 4, 3, 15, 14, 23, 31, 39) + pick_pattern(phase_idx, b->p, 11, b_arr, 63, 62, 61, 60, 59, 55, 54, 47, 39, 31) + pick_pattern(phase_idx, b->p, 11, b_arr, 56, 57, 58, 59, 60, 48, 49, 40, 32, 24);
}

inline int end_evaluate(board *b){
    return b->count();
}

inline int mid_evaluate(board *b){
    int phase_idx, sur0, sur1, canput0, canput1, stab0, stab1, num0, num1;
    unsigned long long black_mobility, white_mobility, empties;
    int b_arr[hw2];
    b->translate_to_arr(b_arr);
    black_mobility = get_mobility(b->b, b->w);
    white_mobility = get_mobility(b->w, b->b);
    empties = ~(b->b | b->w);
    canput0 = min(add_max - 1, pop_count_ull(black_mobility));
    canput1 = min(add_max - 1, pop_count_ull(white_mobility));
    if (canput0 == 0 && canput1 == 0)
        return end_evaluate(b);
    phase_idx = b->phase();
    sur0 = min(add_max - 1, calc_surround(b->b, empties));
    sur1 = min(add_max - 1, calc_surround(b->w, empties));
    calc_stability(b, &stab0, &stab1);
    num0 = pop_count_ull(b->b);
    num1 = pop_count_ull(b->w);
    //cout << canput0 << " " << canput1 << " " << sur0 << " " << sur1 << " " << stab0 << " " << stab1 << " " << num0 << " " << num1 << endl;
    int res = (b->p ? -1 : 1) * (
        calc_pattern(phase_idx, b, b_arr) + 
        add_arr[phase_idx][0][canput0][canput1] + 
        add_arr[phase_idx][1][sur0][sur1] + 
        add_arr[phase_idx][2][stab0][stab1] + 
        add_arr[phase_idx][3][num0][num1]
        );
    if (res > 0)
        res += step_2;
    else if (res < 0)
        res -= step_2;
    res /= step;
    return max(-hw2, min(hw2, res));
}