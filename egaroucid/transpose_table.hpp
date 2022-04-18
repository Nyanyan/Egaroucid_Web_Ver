#pragma once
#include "common.hpp"
#include "board.hpp"

#define search_hash_table_size 262144
constexpr int search_hash_mask = search_hash_table_size - 1;
#define child_inf 100

class search_node{
    public:
        bool reg;
        unsigned long long b;
        unsigned long long w;
        int p;
        int l;
        int u;
        char child[hw2];
    public:
        inline void register_value(board *bd, const int ll, const int uu){
            reg = true;
            b = bd->b;
            w = bd->w;
            p = bd->p;
            l = ll;
            u = uu;
            for (int i = 0; i < hw2; ++i)
                child[i] = -child_inf;
        }

        inline void register_value(const int ll, const int uu){
            l = ll;
            u = uu;
        }

        inline void get(int *ll, int *uu){
            *ll = l;
            *uu = u;
        }

        inline void register_child_value(board *bd, const int policy, const int v){
            reg = true;
            b = bd->b;
            w = bd->w;
            p = bd->p;
            for (int i = 0; i < hw2; ++i)
                child[i] = -child_inf;
            child[policy] = max(-child_inf, v);
        }

        inline void register_child_value(const int policy, const int v){
            child[policy] = max(-child_inf, v);
        }

        inline int child_get(const int policy){
            return child[policy];
        }
};

class transpose_table{
    public:
        int prev;
        int now;
        int hash_reg;
        int hash_get;
    
    private:
        search_node table[2][search_hash_table_size];
        #if USE_MULTI_THREAD
            mutex mtx;
        #endif

    public:
        inline void init_prev(){
            for(int i = 0; i < search_hash_table_size; ++i)
                this->table[this->prev][i].reg = false;
        }

        inline void init_now(){
            for(int i = 0; i < search_hash_table_size; ++i)
                this->table[this->now][i].reg = false;
        }

        inline void reg(board *key, int hash, int l, int u){
            //++this->hash_reg;
            if (!this->table[this->now][hash].reg)
                this->table[this->now][hash].register_value(key, l, u);
            else if (!compare_key(key, &this->table[this->now][hash]))
                this->table[this->now][hash].register_value(key, l, u);
            else
                this->table[this->now][hash].register_value(l, u);
        }

        inline void child_reg(board *key, const int hash, const int policy, int v){
            //++this->hash_reg;
            if (!this->table[this->now][hash].reg)
                this->table[this->now][hash].register_child_value(key, policy, v);
            else if (!compare_key(key, &this->table[this->now][hash]))
                this->table[this->now][hash].register_child_value(key, policy, v);
            else
                this->table[this->now][hash].register_child_value(policy, v);
        }

        inline void get_now(board *key, const int hash, int *l, int *u){
            if (this->table[this->now][hash].reg){
                if (compare_key(key, &this->table[this->now][hash])){
					this->table[this->now][hash].get(l, u);
                    //++this->hash_get;
                } else{
                    *l = -inf;
                    *u = inf;
                }
            } else{
                *l = -inf;
                *u = inf;
            }
        }

        inline void get_prev(board *key, const int hash, int *l, int *u){
            if (this->table[this->prev][hash].reg){
                if (compare_key(key, &this->table[this->prev][hash])){
                    this->table[this->prev][hash].get(l, u);
                    //++this->hash_get;
                } else{
                    *l = -inf;
                    *u = inf;
                }
            } else{
                *l = -inf;
                *u = inf;
            }
        }

        inline int child_get_prev(board *key, const int hash, const int policy){
            if (this->table[this->prev][hash].reg){
                if (compare_key(key, &this->table[this->prev][hash])){
                    return this->table[this->prev][hash].child_get(policy);
                    //++this->hash_get;
                }
            }
            return -child_inf;
        }

        inline int child_get_now(board *key, const int hash, const int policy){
            if (this->table[this->now][hash].reg){
                if (compare_key(key, &this->table[this->now][hash])){
                    return this->table[this->now][hash].child_get(policy);
                    //++this->hash_get;
                }
            }
            return -child_inf;
        }
    
    private:
        inline bool compare_key(board *a, search_node *b){
            return a->p == b->p && a->b == b->b && a->w == b->w;
        }
};

transpose_table transpose_table;

inline void transpose_table_init(){
    transpose_table.now = 0;
    transpose_table.prev = 1;
    transpose_table.init_now();
    transpose_table.init_prev();
    cout << "transpose table initialized" << endl;
}