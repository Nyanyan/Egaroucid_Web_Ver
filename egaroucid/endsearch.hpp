#pragma once
#include <iostream>
#include <functional>
#include <queue>
#include "common.hpp"
#include "board.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "transpose_table.hpp"
#include "midsearch.hpp"
#include "level.hpp"
#if USE_MULTI_THREAD
    #include "thread_pool.hpp"
#endif

using namespace std;

int nega_alpha_ordering_final_nomemo(board *b, bool skipped, int depth, int alpha, int beta, bool use_mpc, double use_mpct, unsigned long long *n_nodes){
    ++(*n_nodes);
    if (!global_searching)
        return -inf;
    if (depth <= simple_mid_threshold)
        return nega_alpha(b, skipped, depth, alpha, beta, n_nodes);
    if (stability_cut(b, &alpha, &beta))
        return alpha;
    if (mpc_min_depth <= depth && depth <= mpc_max_depth && use_mpc){
        if (mpc_higher(b, skipped, depth, beta, use_mpct, n_nodes))
            return beta;
        if (mpc_lower(b, skipped, depth, alpha, use_mpct, n_nodes))
            return alpha;
    }
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_alpha_ordering_final_nomemo(b, true, depth, -beta, -alpha, use_mpc, use_mpct, n_nodes);
        b->p = 1 - b->p;
        return res;
    }
    vector<board> nb;
    mobility mob;
    int canput = 0;
    int hash = b->hash() & search_hash_mask;
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, b, cell);
            nb.emplace_back(b->move_copy(&mob));
            nb[canput].v = move_ordering(b, &nb[canput], hash, cell);
            //nb[canput].v -= canput_bonus * calc_canput_exact(&nb[canput]);
            #if USE_END_PO && false
                if (depth <= po_max_depth && b->parity & cell_div4[cell])
                    nb[canput].v += parity_vacant_bonus;
            #endif
            ++canput;
        }
    }
    if (canput >= 2)
        sort(nb.begin(), nb.end());
    int g, v = -inf;
    for (board &nnb: nb){
        g = -nega_alpha_ordering_final_nomemo(&nnb, false, depth - 1, -beta, -alpha, use_mpc, use_mpct, n_nodes);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    return v;
}

inline bool mpc_higher_final(board *b, bool skipped, int depth, int beta, double t, unsigned long long *n_nodes){
    if (b->n + mpcd[depth] >= hw2 - 5)
        return false;
    int bound = beta + ceil(t * mpcsd_final[depth - mpc_min_depth_final]);
    if (bound > hw2)
        bound = hw2; //return false;
    return nega_alpha_ordering_final_nomemo(b, skipped, mpcd[depth], bound - search_epsilon, bound, true, t, n_nodes) >= bound;
}

inline bool mpc_lower_final(board *b, bool skipped, int depth, int alpha, double t, unsigned long long *n_nodes){
    if (b->n + mpcd[depth] >= hw2 - 5)
        return false;
    int bound = alpha - ceil(t * mpcsd_final[depth - mpc_min_depth_final]);
    if (bound < -hw2)
        bound = -hw2; //return false;
    return nega_alpha_ordering_final_nomemo(b, skipped, mpcd[depth], bound, bound + search_epsilon, true, t, n_nodes) <= bound;
}

inline int last1(board *b, int alpha, int beta, int p0, unsigned long long *n_nodes){
    ++(*n_nodes);
    int score = hw2 - 2 * b->count_opponent();
    int n_flip;
    if (b->p == black)
        n_flip = count_last_flip(b->b, b->w, p0);
    else
        n_flip = count_last_flip(b->w, b->b, p0);
    if (n_flip == 0){
        ++(*n_nodes);
        if (score <= 0){
            score -= 2;
            if (score >= alpha){
                if (b->p == white)
                    n_flip = count_last_flip(b->b, b->w, p0);
                else
                    n_flip = count_last_flip(b->w, b->b, p0);
                score -= 2 * n_flip;
            }
        } else{
            if (score >= alpha){
                if (b->p == white)
                    n_flip = count_last_flip(b->b, b->w, p0);
                else
                    n_flip = count_last_flip(b->w, b->b, p0);
                if (n_flip)
                    score -= 2 * n_flip + 2;
            }
        }
        
    } else
        score += 2 * n_flip;
    return score;
}

inline int last2(board *b, bool skipped, int alpha, int beta, int p0, int p1, unsigned long long *n_nodes){
    ++(*n_nodes);
    int v = -inf, g;
    mobility mob;
    calc_flip(&mob, b, p0);
    if (mob.flip){
        b->move(&mob);
        g = -last1(b, -beta, -alpha, p1, n_nodes);
        b->undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    calc_flip(&mob, b, p1);
    if (mob.flip){
        b->move(&mob);
        g = -last1(b, -beta, -alpha, p0, n_nodes);
        b->undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (v == -inf){
        if (skipped)
            v = end_evaluate(b);
        else{
            b->p = 1 - b->p;
            v = -last2(b, true, -beta, -alpha, p0, p1, n_nodes);
            b->p = 1 - b->p;
        }
        return v;
    }
    return v;
}

inline int last3(board *b, bool skipped, int alpha, int beta, int p0, int p1, int p2, unsigned long long *n_nodes){
    ++(*n_nodes);
    int p0_parity = (b->parity & cell_div4[p0]);
    int p1_parity = (b->parity & cell_div4[p1]);
    int p2_parity = (b->parity & cell_div4[p2]);
    if (p0_parity == 0 && p1_parity && p2_parity){
        int tmp = p0;
        p0 = p1;
        p1 = p2;
        p2 = tmp;
    } else if (p0_parity && p1_parity == 0 && p2_parity){
        swap(p1, p2);
    } else if (p0_parity == 0 && p1_parity == 0 && p2_parity){
        int tmp = p0;
        p0 = p2;
        p2 = p1;
        p1 = tmp;
    } else if (p0_parity == 0 && p1_parity && p2_parity == 0){
        swap(p0, p1);
    }
    unsigned long long legal = b->mobility_ull();
    int v = -inf, g;
    if (legal == 0){
        if (skipped)
            v = end_evaluate(b);
        else{
            b->p = 1 - b->p;
            v = -last3(b, true, -beta, -alpha, p0, p1, p2, n_nodes);
            b->p = 1 - b->p;
        }
        return v;
    }
    mobility mob;
    if (1 & (legal >> p0)){
        calc_flip(&mob, b, p0);
        b->move(&mob);
        g = -last2(b, false, -beta, -alpha, p1, p2, n_nodes);
        b->undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p1)){
        calc_flip(&mob, b, p1);
        b->move(&mob);
        g = -last2(b, false, -beta, -alpha, p0, p2, n_nodes);
        b->undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p2)){
        calc_flip(&mob, b, p2);
        b->move(&mob);
        g = -last2(b, false, -beta, -alpha, p0, p1, n_nodes);
        b->undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    return v;
}

inline int last4(board *b, bool skipped, int alpha, int beta, int p0, int p1, int p2, int p3, unsigned long long *n_nodes){
    ++(*n_nodes);
    if (!skipped){
        int p0_parity = (b->parity & cell_div4[p0]);
        int p1_parity = (b->parity & cell_div4[p1]);
        int p2_parity = (b->parity & cell_div4[p2]);
        int p3_parity = (b->parity & cell_div4[p3]);
        if (p0_parity == 0 && p1_parity && p2_parity && p3_parity){
            int tmp = p0;
            p0 = p1;
            p1 = p2;
            p2 = p3;
            p3 = tmp;
        } else if (p0_parity && p1_parity == 0 && p2_parity && p3_parity){
            int tmp = p1;
            p1 = p2;
            p2 = p3;
            p3 = tmp;
        } else if (p0_parity && p1_parity && p2_parity == 0 && p3_parity){
            swap(p2, p3);
        } else if (p0_parity == 0 && p1_parity == 0 && p2_parity && p3_parity){
            swap(p0, p2);
            swap(p1, p3);
        } else if (p0_parity == 0 && p1_parity && p2_parity == 0 && p3_parity){
            int tmp = p0;
            p0 = p1;
            p1 = p3;
            p3 = p2;
            p2 = tmp;
        } else if (p0_parity == 0 && p1_parity && p2_parity && p3_parity == 0){
            int tmp = p0;
            p0 = p1;
            p1 = p2;
            p2 = tmp;
        } else if (p0_parity && p1_parity == 0 && p2_parity == 0 && p3_parity){
            int tmp = p1;
            p1 = p3;
            p3 = p2;
            p2 = tmp;
        } else if (p0_parity && p1_parity == 0 && p2_parity && p3_parity == 0){
            swap(p1, p2);
        } else if (p0_parity == 0 && p1_parity == 0 && p2_parity == 0 && p3_parity){
            int tmp = p0;
            p0 = p3;
            p3 = p2;
            p2 = p1;
            p1 = tmp;
        } else if (p0_parity == 0 && p1_parity == 0 && p2_parity && p3_parity == 0){
            int tmp = p0;
            p0 = p2;
            p2 = p1;
            p1 = tmp;
        } else if (p0_parity == 0 && p1_parity && p2_parity == 0 && p3_parity == 0){
            swap(p0, p1);
        }
    }
    unsigned long long legal = b->mobility_ull();
    int v = -inf, g;
    if (legal == 0){
        if (skipped)
            v = end_evaluate(b);
        else{
            b->p = 1 - b->p;
            v = -last4(b, true, -beta, -alpha, p0, p1, p2, p3, n_nodes);
            b->p = 1 - b->p;
        }
        return v;
    }
    mobility mob;
    if (1 & (legal >> p0)){
        calc_flip(&mob, b, p0);
        b->move(&mob);
        g = -last3(b, false, -beta, -alpha, p1, p2, p3, n_nodes);
        b->undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p1)){
        calc_flip(&mob, b, p1);
        b->move(&mob);
        g = -last3(b, false, -beta, -alpha, p0, p2, p3, n_nodes);
        b->undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p2)){
        calc_flip(&mob, b, p2);
        b->move(&mob);
        g = -last3(b, false, -beta, -alpha, p0, p1, p3, n_nodes);
        b->undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    if (1 & (legal >> p3)){
        calc_flip(&mob, b, p3);
        b->move(&mob);
        g = -last3(b, false, -beta, -alpha, p0, p1, p2, n_nodes);
        b->undo(&mob);
        alpha = max(alpha, g);
        if (beta <= alpha)
            return alpha;
        v = max(v, g);
    }
    return v;
}

inline int last5(board *b, bool skipped, int alpha, int beta, int p0, int p1, int p2, int p3, int p4, unsigned long long *n_nodes){
    ++(*n_nodes);
    unsigned long long legal = b->mobility_ull();
    int v = -inf, g;
    if (legal == 0){
        if (skipped)
            v = end_evaluate(b);
        else{
            b->p = 1 - b->p;
            v = -last5(b, true, -beta, -alpha, p0, p1, p2, p3, p4, n_nodes);
            b->p = 1 - b->p;
        }
        return v;
    }
    mobility mob;
    if (!skipped){
        int p0_parity = (b->parity & cell_div4[p0]);
        int p1_parity = (b->parity & cell_div4[p1]);
        int p2_parity = (b->parity & cell_div4[p2]);
        int p3_parity = (b->parity & cell_div4[p3]);
        int p4_parity = (b->parity & cell_div4[p4]);
        if (p0_parity && (1 & (legal >> p0))){
            calc_flip(&mob, b, p0);
            b->move(&mob);
            g = -last4(b, false, -beta, -alpha, p1, p2, p3, p4, n_nodes);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (p1_parity && (1 & (legal >> p1))){
            calc_flip(&mob, b, p1);
            b->move(&mob);
            g = -last4(b, false, -beta, -alpha, p0, p2, p3, p4, n_nodes);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (p2_parity && (1 & (legal >> p2))){
            calc_flip(&mob, b, p2);
            b->move(&mob);
            g = -last4(b, false, -beta, -alpha, p0, p1, p3, p4, n_nodes);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (p3_parity && (1 & (legal >> p3))){
            calc_flip(&mob, b, p3);
            b->move(&mob);
            g = -last4(b, false, -beta, -alpha, p0, p1, p2, p4, n_nodes);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (p4_parity && (1 & (legal >> p4))){
            calc_flip(&mob, b, p4);
            b->move(&mob);
            g = -last4(b, false, -beta, -alpha, p0, p1, p2, p3, n_nodes);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (p0_parity == 0 && (1 & (legal >> p0))){
            calc_flip(&mob, b, p0);
            b->move(&mob);
            g = -last4(b, false, -beta, -alpha, p1, p2, p3, p4, n_nodes);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (p1_parity == 0 && (1 & (legal >> p1))){
            calc_flip(&mob, b, p1);
            b->move(&mob);
            g = -last4(b, false, -beta, -alpha, p0, p2, p3, p4, n_nodes);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (p2_parity == 0 && (1 & (legal >> p2))){
            calc_flip(&mob, b, p2);
            b->move(&mob);
            g = -last4(b, false, -beta, -alpha, p0, p1, p3, p4, n_nodes);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (p3_parity == 0 && (1 & (legal >> p3))){
            calc_flip(&mob, b, p3);
            b->move(&mob);
            g = -last4(b, false, -beta, -alpha, p0, p1, p2, p4, n_nodes);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (p4_parity == 0 && (1 & (legal >> p4))){
            calc_flip(&mob, b, p4);
            b->move(&mob);
            g = -last4(b, false, -beta, -alpha, p0, p1, p2, p3, n_nodes);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
    } else{
        if (1 & (legal >> p0)){
            calc_flip(&mob, b, p0);
            b->move(&mob);
            g = -last4(b, false, -beta, -alpha, p1, p2, p3, p4, n_nodes);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (1 & (legal >> p1)){
            calc_flip(&mob, b, p1);
            b->move(&mob);
            g = -last4(b, false, -beta, -alpha, p0, p2, p3, p4, n_nodes);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (1 & (legal >> p2)){
            calc_flip(&mob, b, p2);
            b->move(&mob);
            g = -last4(b, false, -beta, -alpha, p0, p1, p3, p4, n_nodes);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (1 & (legal >> p3)){
            calc_flip(&mob, b, p3);
            b->move(&mob);
            g = -last4(b, false, -beta, -alpha, p0, p1, p2, p4, n_nodes);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
        if (1 & (legal >> p4)){
            calc_flip(&mob, b, p4);
            b->move(&mob);
            g = -last4(b, false, -beta, -alpha, p0, p1, p2, p3, n_nodes);
            b->undo(&mob);
            alpha = max(alpha, g);
            if (beta <= alpha)
                return alpha;
            v = max(v, g);
        }
    }
    return v;
}

inline void pick_vacant(board *b, int cells[]){
    int idx = 0;
    unsigned long long empties = ~(b->b | b->w);
    for (const int &cell: vacant_lst){
        if (1 & (empties >> cell))
            cells[idx++] = cell;
    }
}


int nega_alpha_final(board *b, bool skipped, const int depth, int alpha, int beta, unsigned long long *n_nodes){
    if (!global_searching)
        return -inf;
    if (depth == 5){
        int cells[5];
        pick_vacant(b, cells);
        return last5(b, skipped, alpha, beta, cells[0], cells[1], cells[2], cells[3], cells[4], n_nodes);
    }
    ++(*n_nodes);
    if (stability_cut(b, &alpha, &beta))
        return alpha;
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_alpha_final(b, true, depth, -beta, -alpha, n_nodes);
        b->p = 1 - b->p;
        return res;
    }
    int g, v = -inf;
    mobility mob;
    if (0 < b->parity && b->parity < 15){
        for (const int &cell: vacant_lst){
            if ((b->parity & cell_div4[cell]) && (1 & (legal >> cell))){
                calc_flip(&mob, b, cell);
                b->move(&mob);
                g = -nega_alpha_final(b, false, depth - 1, -beta, -alpha, n_nodes);
                b->undo(&mob);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
        }
        for (const int &cell: vacant_lst){
            if ((b->parity & cell_div4[cell]) == 0 && (1 & (legal >> cell))){
                calc_flip(&mob, b, cell);
                b->move(&mob);
                g = -nega_alpha_final(b, false, depth - 1, -beta, -alpha, n_nodes);
                b->undo(&mob);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
        }
    } else{
        for (const int &cell: vacant_lst){
            if (1 & (legal >> cell)){
                calc_flip(&mob, b, cell);
                b->move(&mob);
                g = -nega_alpha_final(b, false, depth - 1, -beta, -alpha, n_nodes);
                b->undo(&mob);
                alpha = max(alpha, g);
                if (beta <= alpha)
                    return alpha;
                v = max(v, g);
            }
        }
    }
    return v;
}

int nega_alpha_ordering_simple_final(board *b, bool skipped, const int depth, int alpha, int beta, bool use_mpc, double mpct_in, unsigned long long *n_nodes){
    if (!global_searching)
        return -inf;
    if (depth <= simple_end_threshold)
        return nega_alpha_final(b, skipped, depth, alpha, beta, n_nodes);
    ++(*n_nodes);
    int hash = b->hash() & search_hash_mask;
    int l, u;
    transpose_table.get_now(b, hash, &l, &u);
    if (u == l)
        return u;
    if (l >= beta)
        return l;
    if (alpha >= u)
        return u;
    alpha = max(alpha, l);
    beta = min(beta, u);
    if (mpc_min_depth_final <= depth && depth <= mpc_max_depth_final && use_mpc){
        if (mpc_higher_final(b, skipped, depth, beta, mpct_in, n_nodes))
            return beta;
        if (mpc_lower_final(b, skipped, depth, alpha, mpct_in, n_nodes))
            return alpha;
    }
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_alpha_ordering_simple_final(b, true, depth, -beta, -alpha, use_mpc, mpct_in, n_nodes);
        b->p = 1 - b->p;
        return res;
    }
    const int canput = pop_count_ull(legal);
    board *nb = new board[canput];
    mobility mob;
    int idx = 0;
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, b, cell);
            b->move_copy(&mob, &nb[idx]);
            nb[idx].v = -canput_bonus * calc_canput_exact(&nb[idx]);
            if (depth <= po_max_depth && (b->parity & cell_div4[cell]))
                nb[idx].v += parity_vacant_bonus;
            ++idx;
        }
    }
    if (canput >= 2)
        sort(nb, nb + canput);
    int g, v = -inf;
    for (idx = 0; idx < canput; ++idx){
        g = -nega_alpha_ordering_simple_final(&nb[idx], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, n_nodes);
        alpha = max(alpha, g);
        if (beta <= alpha){
            if (l < alpha)
                transpose_table.reg(b, hash, alpha, u);
            delete[] nb;
            return alpha;
        }
        v = max(v, g);
    }
    delete[] nb;
    if (v <= alpha)
        transpose_table.reg(b, hash, l, v);
    else
        transpose_table.reg(b, hash, v, v);
    return v;
}

int nega_alpha_ordering_final(board *b, bool skipped, const int depth, int alpha, int beta, bool use_mpc, double mpct_in, unsigned long long *n_nodes){
    if (!global_searching)
        return -inf;
    if (depth <= simple_end_threshold2)
        return nega_alpha_ordering_simple_final(b, skipped, depth, alpha, beta, use_mpc, mpct_in, n_nodes);
    ++(*n_nodes);
    int hash = b->hash() & search_hash_mask;
    int l, u;
    transpose_table.get_now(b, hash, &l, &u);
    if (u == l)
        return u;
    if (l >= beta)
        return l;
    if (alpha >= u)
        return u;
    alpha = max(alpha, l);
    beta = min(beta, u);
    if (mpc_min_depth_final <= depth && depth <= mpc_max_depth_final && use_mpc){
        if (mpc_higher_final(b, skipped, depth, beta, mpct_in, n_nodes))
            return beta;
        if (mpc_lower_final(b, skipped, depth, alpha, mpct_in, n_nodes))
            return alpha;
    }
    unsigned long long legal = b->mobility_ull();
    if (legal == 0){
        if (skipped)
            return end_evaluate(b);
        b->p = 1 - b->p;
        int res = -nega_alpha_ordering_final(b, true, depth, -beta, -alpha, use_mpc, mpct_in, n_nodes);
        b->p = 1 - b->p;
        return res;
    }
    const int canput = pop_count_ull(legal);
    board *nb = new board[canput];
    mobility mob;
    int idx = 0;
    //int first_threshold = 0;
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, b, cell);
            b->move_copy(&mob, &nb[idx]);
            nb[idx].v = move_ordering(b, &nb[idx], hash, cell);
            //if (-mid_evaluate(&nb[idx]) > beta - 3)
            //    ++first_threshold;
            ++idx;
        }
    }
    //first_threshold = max(1, first_threshold);
    if (canput >= 2)
        sort(nb, nb + canput);
    int g, v = -inf;
    for (idx = 0; idx < canput; ++idx){
        g = -nega_alpha_ordering_final(&nb[idx], false, depth - 1, -beta, -alpha, use_mpc, mpct_in, n_nodes);
        //transpose_table.child_reg(b, hash, nb[idx].policy, g);
        alpha = max(alpha, g);
        if (beta <= alpha){
            #if USE_END_TC
                if (l < alpha)
                    transpose_table.reg(b, hash, alpha, u);
            #endif
            delete[] nb;
            return alpha;
        }
        v = max(v, g);
    }
    delete[] nb;
    if (v <= alpha)
        transpose_table.reg(b, hash, l, v);
    else
        transpose_table.reg(b, hash, v, v);
    return v;
}

int mtd_final(board *b, bool skipped, int depth, int l, int u, bool use_mpc, double use_mpct, int g, unsigned long long *n_nodes){
    int beta;
    l /= 2;
    u /= 2;
    g = max(l, min(u, g / 2));
    //cout << l * 2 << " " << g * 2 << " " << u * 2 << endl;
    while (u - l > 0){
        beta = max(l + search_epsilon, g);
        g = nega_alpha_ordering_final(b, skipped, depth, beta * 2 - search_epsilon, beta * 2, use_mpc, use_mpct, n_nodes) / 2;
        if (g < beta)
            u = g;
        else
            l = g;
        //cout << l * 2 << " " << g * 2 << " " << u * 2 << endl;
    }
    //cout << g * 2 << endl;
    return g * 2;
}

inline search_result endsearch(board b, long long strt, bool use_mpc, double use_mpct){
    unsigned long long legal = b.mobility_ull();
    vector<pair<int, board>> nb;
    mobility mob;
    vector<int> prev_vals;
    int i;
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            calc_flip(&mob, &b, cell);
            nb.emplace_back(make_pair(cell, b.move_copy(&mob)));
            //cout << cell << " ";
        }
    }
    //cout << endl;
    int canput = nb.size();
    //cout << "canput: " << canput << endl;
    int policy = -1;
    int tmp_policy = -1;
    int alpha, beta, g, value;
    unsigned long long searched_nodes = 0;
    transpose_table.hash_get = 0;
    transpose_table.hash_reg = 0;
    int max_depth = hw2 - b.n;
    alpha = -hw2;
    beta = hw2;
    int pre_search_depth = max(1, min(15, max_depth - simple_end_threshold + simple_mid_threshold + 3));
    cout << "pre search depth " << pre_search_depth << endl;
    double pre_search_mpcd = 0.6;
    transpose_table.init_now();
    for (i = 0; i < canput; ++i)
        nb[i].second.v = -mtd(&nb[i].second, false, pre_search_depth - 1, -hw2, hw2, true, pre_search_mpcd, &searched_nodes);
    swap(transpose_table.now, transpose_table.prev);
    transpose_table.init_now();
    for (i = 0; i < canput; ++i){
        nb[i].second.v += -mtd(&nb[i].second, false, pre_search_depth, -hw2, hw2, true, pre_search_mpcd, &searched_nodes);
        nb[i].second.v /= 2;
    }
    swap(transpose_table.now, transpose_table.prev);
    if (canput >= 2)
        sort(nb.begin(), nb.end(), move_ordering_sort);
    cout << "pre search depth " << pre_search_depth << " policy " << nb[0].first << " value " << nb[0].second.v << " nodes " << searched_nodes << endl;
    transpose_table.init_now();
    searched_nodes = 0;
    if (nb[0].second.n < hw2 - 5){
        for (i = 0; i < canput; ++i){
            g = -mtd_final(&nb[i].second, false, max_depth - 1, -beta, -alpha, use_mpc, use_mpct, -nb[i].second.v, &searched_nodes);
            //g = -nega_scout_final(&nb[i].second, false, max_depth - 1, -beta, -alpha, use_mpc, use_mpct, true, &searched_nodes);
            cout << "policy " << nb[i].first << " value " << g << " expected " << nb[i].second.v << endl;
            if (alpha < g || i == 0){
                alpha = g;
                tmp_policy = nb[i].first;
            }
        }
        
    } else{
        int cells[5];
        for (i = 0; i < canput; ++i){
            pick_vacant(&nb[i].second, cells);
            if (nb[i].second.n == hw2 - 5)
                g = -last5(&nb[i].second, false, -beta, -alpha, cells[0], cells[1], cells[2], cells[3], cells[4], &searched_nodes);
            else if (nb[i].second.n == hw2 - 4)
                g = -last4(&nb[i].second, false, -beta, -alpha, cells[0], cells[1], cells[2], cells[3], &searched_nodes);
            else if (nb[i].second.n == hw2 - 3)
                g = -last3(&nb[i].second, false, -beta, -alpha, cells[0], cells[1], cells[2], &searched_nodes);
            else if (nb[i].second.n == hw2 - 2)
                g = -last2(&nb[i].second, false, -beta, -alpha, cells[0], cells[1], &searched_nodes);
            else if (nb[i].second.n == hw2 - 1)
                g = -last1(&nb[i].second, -beta, -alpha, cells[0], &searched_nodes);
            else
                g = -end_evaluate(&nb[i].second);
            if (alpha < g || i == 0){
                alpha = g;
                tmp_policy = nb[i].first;
            }
        }
    }
    swap(transpose_table.now, transpose_table.prev);
    if (global_searching){
        policy = tmp_policy;
        value = alpha;
        cout << "final depth: " << max_depth << " policy: " << policy << " value: " << alpha << " nodes: " << searched_nodes << endl;
    } else {
        value = -inf;
        for (int i = 0; i < (int)nb.size(); ++i){
            if (nb[i].second.v > value){
                value = nb[i].second.v;
                policy = nb[i].first;
            }
        }
    }
    search_result res;
    res.policy = policy;
    res.value = value;
    res.depth = max_depth;
    res.nps = 0;
    return res;
}
