#pragma once
#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "transpose_table.hpp"

using namespace std;

#define search_epsilon 1
#define cache_hit 10000
#define cache_now 10000
#define parity_vacant_bonus 5
#define canput_bonus 10
#define w_former_search 20
#define w_stability 5
#define w_evaluate 10
#define w_surround 5
#define w_mobility 30
#define w_parity 10

#define mpc_min_depth 3
#define mpc_max_depth 15
#define mpc_min_depth_final 5
#define mpc_max_depth_final 30

#define simple_mid_threshold 3
#define simple_end_threshold 7
#define simple_end_threshold2 13

#define po_max_depth 15

#define mid_first_threshold_div 5
#define end_first_threshold_div 6

const int cell_weight[hw2] = {
    18,  4,  16, 12, 12, 16,  4, 18,
     4,  2,   6,  8,  8,  6,  2,  4,
    16,  6,  14, 10, 10, 14,  6, 16,
    12,  8,  10,  0,  0, 10,  8, 12,
    12,  8,  10,  0,  0, 10,  8, 12,
    16,  6,  14, 10, 10, 14,  6, 16,
     4,  2,   6,  8,  8,  6,  2,  4,
    18,  4,  16, 12, 12, 16,  4, 18
};

const int mpcd[41] = {
    0, 1, 0, 1, 2, 3, 2, 3, 4, 3, 
    4, 3, 4, 5, 4, 5, 6, 5, 6, 7, 
    6, 7, 6, 7, 8, 7, 8, 9, 8, 9, 
    8, 9, 10, 9, 10, 11, 10, 11, 10, 11,
    12
};

double mpcsd[n_phases][mpc_max_depth - mpc_min_depth + 1]={
    {4.03, 3.868, 5.32, 5.835, 4.183, 11.24, 5.07, 3.636, 6.546, 6.504, 4.731, 4.71, 6.713},
    {5.14, 5.947, 4.954, 5.95, 7.297, 4.839, 6.218, 4.48, 4.901, 5.128, 5.708, 4.917, 5.511},
    {5.095, 4.254, 4.93, 6.168, 6.944, 5.912, 8.311, 7.827, 6.558, 7.074, 4.332, 8.546, 10.793},
    {3.742, 3.941, 3.453, 8.207, 4.035, 8.374, 8.544, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
};

double mpcsd_final[mpc_max_depth_final - mpc_min_depth_final + 1] = {
    5.385, 7.366, 6.492, 6.701, 6.453, 5.528, 6.557, 6.213, 6.208, 7.868, 8.039, 6.752, 9.902, 9.416, 8.04, 8.512, 7.576, 7.887, 7.796, 7.73, 8.09, 8.427, 8.115, 8.605, 8.298, 10.272
};

unsigned long long can_be_flipped[hw2];
vector<int> vacant_lst;

struct search_result{
    int policy;
    int value;
    int depth;
    int nps;
};

struct enhanced_mtd{
    int policy;
    int error;
    int l;
    int u;
    board b;
};

inline int enhanced_mtd_cost(const enhanced_mtd &elem){
    int res = 0;
    res += max(elem.error, elem.error * 4);
    res += elem.b.v;
    return res;
}

bool operator< (const enhanced_mtd &elem1, const enhanced_mtd &elem2){
    return enhanced_mtd_cost(elem1) < enhanced_mtd_cost(elem2);
};

int cmp_vacant(int p, int q){
    return cell_weight[p] > cell_weight[q];
}

//int nega_alpha(board *b, bool skipped, int depth, int alpha, int beta, int *n_nodes);

inline int move_ordering(board *b, board *nb, const int hash, const int policy){
    int v = transpose_table.child_get_now(b, hash, policy) * w_former_search;
    if (v == -child_inf * w_former_search){
        v = transpose_table.child_get_prev(b, hash, policy) * w_former_search;
        if (v == -child_inf * w_former_search)
            v = 0;
        else
            v += cache_hit;
    } else
        v += cache_hit + cache_now;
    v += cell_weight[policy];
    v += -mid_evaluate(nb) * w_evaluate;
    //int n_nodes = 0;
    //v += -nega_alpha(nb, false, 2, -hw2, hw2, &n_nodes);
    int stab0, stab1;
    calc_stability_fast(nb, &stab0, &stab1);
    unsigned long long n_empties = ~(nb->b | nb->w);
    if (b->p == black){
        v += (stab0 - stab1) * w_stability;
        v += (calc_surround(nb->w, n_empties) - calc_surround(nb->b, n_empties)) * w_surround;
    } else{
        v += (stab1 - stab0) * w_stability;
        v += (calc_surround(nb->b, n_empties) - calc_surround(nb->w, n_empties)) * w_surround;
    }
    v -= pop_count_ull(nb->mobility_ull()) * w_mobility;
    if (b->parity & cell_div4[policy])
        v += w_parity;
    return v;
}

inline void move_ordering_eval(board *b){
    b->v = -mid_evaluate(b);
}

inline bool stability_cut(board *b, int *alpha, int *beta){
    int stab[2];
    calc_stability(b, &stab[0], &stab[1]);
    *alpha = max(*alpha, 2 * stab[b->p] - hw2);
    *beta = min(*beta, hw2 - 2 * stab[1 - b->p]);
    return *alpha >= *beta;
}

inline int calc_canput_exact(board *b){
    return pop_count_ull(b->mobility_ull());
}

bool move_ordering_sort(pair<int, board> &a, pair<int, board> &b){
    return a.second.v > b.second.v;
}

bool move_ordering_sort_int_int(pair<int, int> &a, pair<int, int> &b){
    return a.second > b.second;
}
