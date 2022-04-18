#pragma once
#include <iostream>
//#include <emmintrin.h>
#include "mobility.hpp"

using namespace std;

#define p171 17ULL
#define p172 289ULL
#define p173 4913ULL
#define p174 83521ULL
#define p175 1419857ULL
#define p176 24137569ULL
#define p177 410338673ULL
#define p191 19ULL
#define p192 361ULL
#define p193 6859ULL
#define p194 130321ULL
#define p195 2476099ULL
#define p196 47045881ULL
#define p197 893871739ULL

const int cell_div4[hw2] = {
    1, 1, 1, 1, 2, 2, 2, 2, 
    1, 1, 1, 1, 2, 2, 2, 2, 
    1, 1, 1, 1, 2, 2, 2, 2, 
    1, 1, 1, 1, 2, 2, 2, 2, 
    4, 4, 4, 4, 8, 8, 8, 8, 
    4, 4, 4, 4, 8, 8, 8, 8, 
    4, 4, 4, 4, 8, 8, 8, 8, 
    4, 4, 4, 4, 8, 8, 8, 8
};

inline int create_one_color(int idx, const int k) {
    int res = 0;
    for (int i = 0; i < hw; ++i) {
        if (idx % 3 == k) {
            res |= 1 << i;
        }
        idx /= 3;
    }
    return res;
}

inline int trans(const int pt, const int k) {
    if (k == 0)
        return pt << 1;
    else
        return pt >> 1;
}

inline int move_line_half(const int p, const int o, const int place, const int k) {
    int mask;
    int res = 0;
    int pt = 1 << (hw_m1 - place);
    if (pt & p || pt & o)
        return 0;
    mask = trans(pt, k);
    while (mask && (mask & o)) {
        ++res;
        mask = trans(mask, k);
        if (mask & p)
            return res;
    }
    return 0;
}

unsigned long long get_mobility(const unsigned long long P, const unsigned long long O){
    unsigned long long moves, mO;
	unsigned long long flip1, flip7, flip9, flip8, pre1, pre7, pre9, pre8;
	mO = O & 0x7e7e7e7e7e7e7e7eULL;
	flip1  = mO & (P << 1);		flip7  = mO & (P << 7);		flip9  = mO & (P << 9);		flip8  = O & (P << 8);
	flip1 |= mO & (flip1 << 1);	flip7 |= mO & (flip7 << 7);	flip9 |= mO & (flip9 << 9);	flip8 |= O & (flip8 << 8);
	pre1 = mO & (mO << 1);		pre7 = mO & (mO << 7);		pre9 = mO & (mO << 9);		pre8 = O & (O << 8);
	flip1 |= pre1 & (flip1 << 2);	flip7 |= pre7 & (flip7 << 14);	flip9 |= pre9 & (flip9 << 18);	flip8 |= pre8 & (flip8 << 16);
	flip1 |= pre1 & (flip1 << 2);	flip7 |= pre7 & (flip7 << 14);	flip9 |= pre9 & (flip9 << 18);	flip8 |= pre8 & (flip8 << 16);
	moves = flip1 << 1;		moves |= flip7 << 7;		moves |= flip9 << 9;		moves |= flip8 << 8;
	flip1  = mO & (P >> 1);		flip7  = mO & (P >> 7);		flip9  = mO & (P >> 9);		flip8  = O & (P >> 8);
	flip1 |= mO & (flip1 >> 1);	flip7 |= mO & (flip7 >> 7);	flip9 |= mO & (flip9 >> 9);	flip8 |= O & (flip8 >> 8);
	pre1 >>= 1;			pre7 >>= 7;			pre9 >>= 9;			pre8 >>= 8;
	flip1 |= pre1 & (flip1 >> 2);	flip7 |= pre7 & (flip7 >> 14);	flip9 |= pre9 & (flip9 >> 18);	flip8 |= pre8 & (flip8 >> 16);
	flip1 |= pre1 & (flip1 >> 2);	flip7 |= pre7 & (flip7 >> 14);	flip9 |= pre9 & (flip9 >> 18);	flip8 |= pre8 & (flip8 >> 16);
	moves |= flip1 >> 1;		moves |= flip7 >> 7;		moves |= flip9 >> 9;		moves |= flip8 >> 8;
	return moves & ~(P|O);
}

class board {
    public:
        unsigned long long b;
        unsigned long long w;
        int p;
        int policy;
        int v;
        int n;
        int parity;

    public:
        bool operator<(const board& another) const {
            return v > another.v;
        }

        inline board copy(){
            board res;
            res.b = b;
            res.w = w;
            res.p = p;
            res.policy = policy;
            res.v = v;
            res.n = n;
            res.parity = parity;
            return res;
        }

        void copy(board *res){
            res->b = b;
            res->w = w;
            res->p = p;
            res->policy = policy;
            res->v = v;
            res->n = n;
            res->parity = parity;
        }

        inline unsigned long long hash(){
            /*
            return
                (b * 3) ^ 
                ((b >> 16) * p171) ^ 
                ((b >> 32) * p173) ^ 
                ((b >> 48) * p175) ^ 
                (w * 5) ^ 
                ((w >> 16) * p191) ^ 
                ((w >> 32) * p193) ^ 
                ((w >> 48) * p195);
            */
            return
                (b * 3) ^ ((b >> 8) * 7) ^ 
                ((b >> 16) * p171) ^ 
                ((b >> 24) * p172) ^ 
                ((b >> 32) * p173) ^ 
                ((b >> 40) * p174) ^ 
                ((b >> 48) * p175) ^ 
                ((b >> 56) * p176) ^ 
                (w * 5) ^ ((w >> 8) * 11) ^ 
                ((w >> 16) * p191) ^ 
                ((w >> 24) * p192) ^ 
                ((w >> 32) * p193) ^ 
                ((w >> 40) * p194) ^ 
                ((w >> 48) * p195) ^ 
                ((w >> 56) * p196);
            
            /*
            unsigned long long res = 0;
            for (int i = 0; i < hw2; ++i){
                if (1 & (b >> i))
                    res ^= hash_rand[0][i];
                if (1 & (w >> i))
                    res ^= hash_rand[1][i];
            }
            return res;
            */
        }

        inline unsigned long long hash_player(){
            if (p == black)
                return hash();
            /*
            return
                (w * 3) ^ 
                ((w >> 16) * p171) ^ 
                ((w >> 32) * p173) ^ 
                ((w >> 48) * p175) ^ 
                (b * 5) ^ 
                ((b >> 16) * p191) ^ 
                ((b >> 32) * p193) ^ 
                ((b >> 48) * p195);
            */
            return
                (w * 3) ^ ((w >> 8) * 7) ^ 
                ((w >> 16) * p171) ^ 
                ((w >> 24) * p172) ^ 
                ((w >> 32) * p173) ^ 
                ((w >> 40) * p174) ^ 
                ((w >> 48) * p175) ^ 
                ((w >> 56) * p176) ^ 
                (b * 5) ^ ((b >> 8) * 11) ^ 
                ((b >> 16) * p191) ^ 
                ((b >> 24) * p192) ^ 
                ((b >> 32) * p193) ^ 
                ((b >> 40) * p194) ^ 
                ((b >> 48) * p195) ^ 
                ((b >> 56) * p196);
        }

        inline void white_mirror(){
            b = white_line(b);
            w = white_line(w);
        }

        inline void black_mirror(){
            b = black_line(b);
            w = black_line(w);
        }

        inline void vertical_mirror(){
            b = mirror_v(b);
            w = mirror_v(w);
        }

        inline void print() {
            for (int i = hw2_m1; i >= 0; --i){
                if (1 & (b >> i))
                    cout << "X ";
                else if (1 & (w >> i))
                    cout << "O ";
                else
                    cout << ". ";
                if (i % hw == 0)
                    cout << endl;
            }
        }

        inline unsigned long long mobility_ull(){
            unsigned long long res;
            if (p == black)
                res = get_mobility(b, w);
            else
                res = get_mobility(w, b);
            return res;
        }

        inline unsigned long long mobility_ull_black(){
            return get_mobility(b, w);
        }

        inline unsigned long long mobility_ull_white(){
            return get_mobility(w, b);
        }

        inline void full_stability(unsigned long long *h, unsigned long long *v, unsigned long long *d7, unsigned long long *d9){
            const unsigned long long stones = (b | w);
            *h = full_stability_h(stones);
            *v = full_stability_v(stones);
            full_stability_d(stones, d7, d9);
        }

        inline void move(const mobility *mob) {
            if (p == black){
                b ^= mob->flip;
                w &= ~b;
                b |= 1ULL << mob->pos;
            } else{
                w ^= mob->flip;
                b &= ~w;
                w |= 1ULL << mob->pos;
            }
            p = 1 - p;
            ++n;
            policy = mob->pos;
            parity ^= cell_div4[mob->pos];
        }

        inline void move_copy(const mobility *mob, board *res) {
            if (p == black){
                res->b = b ^ mob->flip;
                res->w = w & (~res->b);
                res->b |= 1ULL << mob->pos;
            } else{
                res->w = w ^ mob->flip;
                res->b = b & (~res->w);
                res->w |= 1ULL << mob->pos;
            }
            res->p = 1 - p;
            res->n = n + 1;
            res->policy = mob->pos;
            res->parity = parity ^ cell_div4[mob->pos];
        }

        inline board move_copy(const mobility *mob) {
            board res;
            move_copy(mob, &res);
            return res;
        }

        inline void undo(const mobility *mob){
            p = 1 - p;
            --n;
            policy = -1;
            parity ^= cell_div4[mob->pos];
            if (p == black){
                b &= ~(1ULL << mob->pos);
                b ^= mob->flip;
                w |= mob->flip;
            } else{
                w &= ~(1ULL << mob->pos);
                w ^= mob->flip;
                b |= mob->flip;
            }
        }

        inline void translate_to_arr(int res[]) {
            for (int i = 0; i < hw2; ++i){
                if (1 & (b >> i))
                    res[hw2_m1 - i] = black;
                else if (1 & (w >> i))
                    res[hw2_m1 - i] = white;
                else
                    res[hw2_m1 - i] = vacant;
            }
        }

        inline void translate_from_arr(const int arr[], int player) {
            int i;
            b = 0;
            w = 0;
            n = hw2;
            parity = 0;
            for (i = 0; i < hw2; ++i) {
                if (arr[hw2_m1 - i] == black)
                    b |= 1ULL << i;
                else if (arr[hw2_m1 - i] == white)
                    w |= 1ULL << i;
                else{
                    --n;
                    parity ^= cell_div4[i];
                }
            }
            p = player;
            policy = -1;
        }

        inline void translate_from_ull(const unsigned long long bk, const unsigned long long wt, int player) {
            if (bk & wt)
                cout << "both on same square" << endl;
            b = bk;
            w = wt;
            n = hw2;
            parity = 0;
            for (int i = 0; i < hw2; ++i) {
                if ((1 & (bk >> i)) == 0 && (1 & (wt >> i)) == 0){
                    --n;
                    parity ^= cell_div4[i];
                }
            }
            p = player;
            policy = -1;
        }

        inline int count(int player){
            int b_score = pop_count_ull(b), w_score = pop_count_ull(w);
            int black_score = b_score - w_score, vacant_score = hw2 - b_score - w_score;
            if (black_score > 0)
                black_score += vacant_score;
            else if (black_score < 0)
                black_score -= vacant_score;
            return (player ? -1 : 1) * black_score;
        }

        inline int count(){
            return count(p);
        }

        inline int count_player(){
            if (p == black)
                return pop_count_ull(b);
            return pop_count_ull(w);
        }

        inline int count_opponent(){
            if (p == white)
                return pop_count_ull(b);
            return pop_count_ull(w);
        }

        inline int raw_count(){
            if (p == black)
                return pop_count_ull(b);
            return pop_count_ull(w);
        }

        inline void board_canput(int canput_arr[], const unsigned long long mobility_black, const unsigned long long mobility_white){
            for (int i = 0; i < hw2; ++i){
                canput_arr[i] = 0;
                if (1 & (mobility_black >> i))
                    ++canput_arr[i];
                if (1 & (mobility_white >> i))
                    canput_arr[i] += 2;
            }
        }

        inline void board_canput(int canput_arr[]){
            const unsigned long long mobility_black = get_mobility(b, w);
            const unsigned long long mobility_white = get_mobility(w, b);
            board_canput(canput_arr, mobility_black, mobility_white);
        }

        inline void check_player(){
            bool passed = (mobility_ull() == 0);
            if (passed){
                p = 1 - p;
                passed = (mobility_ull() == 0);
                if (passed)
                    p = vacant;
            }
        }

        inline void reset(){
            constexpr int first_board[hw2] = {
                vacant,vacant,vacant,vacant,vacant,vacant,vacant,vacant,
                vacant,vacant,vacant,vacant,vacant,vacant,vacant,vacant,
                vacant,vacant,vacant,vacant,vacant,vacant,vacant,vacant,
                vacant,vacant,vacant,white,black,vacant,vacant,vacant,
                vacant,vacant,vacant,black,white,vacant,vacant,vacant,
                vacant,vacant,vacant,vacant,vacant,vacant,vacant,vacant,
                vacant,vacant,vacant,vacant,vacant,vacant,vacant,vacant,
                vacant,vacant,vacant,vacant,vacant,vacant,vacant,vacant
            };
            translate_from_arr(first_board, black);
        }

        inline int phase(){
            return min(n_phases - 1, max(0, (n - 24) / 10));
        }
    
    private:
        inline unsigned long long full_stability_h(unsigned long long full){
            full &= full >> 1;
            full &= full >> 2;
            full &= full >> 4;
            return (full & 0x0101010101010101) * 0xff;
        }

        inline unsigned long long full_stability_v(unsigned long long full){
            full &= (full >> 8) | (full << 56);
            full &= (full >> 16) | (full << 48);
            full &= (full >> 32) | (full << 32);
            return full;
        }

        inline void full_stability_d(unsigned long long full, unsigned long long *full_d7, unsigned long long *full_d9){
            static const unsigned long long edge = 0xff818181818181ff;
            static const unsigned long long e7[] = {
                0xffff030303030303, 0xc0c0c0c0c0c0ffff, 0xffffffff0f0f0f0f, 0xf0f0f0f0ffffffff };
            static const unsigned long long e9[] = {
                0xffffc0c0c0c0c0c0, 0x030303030303ffff, 0x0f0f0f0ff0f0f0f0 };
            unsigned long long l7, r7, l9, r9;
            l7 = r7 = full;
            l7 &= edge | (l7 >> 7);		r7 &= edge | (r7 << 7);
            l7 &= e7[0] | (l7 >> 14);	r7 &= e7[1] | (r7 << 14);
            l7 &= e7[2] | (l7 >> 28);	r7 &= e7[3] | (r7 << 28);
            *full_d7 = l7 & r7;

            l9 = r9 = full;
            l9 &= edge | (l9 >> 9);		r9 &= edge | (r9 << 9);
            l9 &= e9[0] | (l9 >> 18);	r9 &= e9[1] | (r9 << 18);
            *full_d9 = l9 & r9 & (e9[2] | (l9 >> 36) | (r9 << 36));
        }
};

inline void calc_flip(mobility *mob, board *b, const int policy){
    if (b->p == black)
        mob->calc_flip(b->b, b->w, policy);
    else
        mob->calc_flip(b->w, b->b, policy);
}