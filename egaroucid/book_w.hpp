#pragma once
#include <iostream>
#include "common.hpp"

using namespace std;

#define book_hash_table_size 8192
constexpr int book_hash_mask = book_hash_table_size - 1;
#define book_stones 64
#define ln_repair_book 27

struct book_node{
    unsigned long long b;
    unsigned long long w;
    int policy;
    book_node* p_n_node;
};

class book{
    private:
        book_node *book[book_hash_table_size];

    public:
        inline int get(board *key){
            book_node *p_node = book[key->hash() & book_hash_mask];
            while(p_node != NULL){
                if(compare_key(key, p_node)){
                    return p_node->policy;
                }
                p_node = p_node->p_n_node;
            }
            return -1;
        }

        inline void init(){
            int i;
REPLACE_BOOK_HERE
            string param_compressed1 = unzip_str_book(compressed_data, repair_num, replace_from_str, replace_to_str);
            int zip_int[500000];
            int ln = zip_to_int_book(param_compressed1, zip_int);
            int coord;
            board fb;
            for(int i = 0; i < book_hash_table_size; ++i)
                book[i] = NULL;
            int n_book = 0;
            int data_idx = 0;
            int tmp[16];
            mobility mob;
            while (data_idx < ln - 1){
                fb.reset();
                calc_flip(&mob, &fb, 37);
                fb.move(&mob);
                while (true){
                    coord = zip_int[data_idx++];
                    if (coord == 37)
                        break;
                    calc_flip(&mob, &fb, coord);
                    fb.move(&mob);
                }
                coord = zip_int[data_idx++];
                n_book += register_symmetric_book(fb, coord);
            }
            cout << n_book << " boards in book" << endl;
        }
    private:
        inline book_node* node_init(board *key, int policy){
            book_node* p_node = NULL;
            p_node = (book_node*)malloc(sizeof(book_node));
            p_node->b = key->b;
            p_node->w = key->w;
            p_node->policy = policy;
            p_node->p_n_node = NULL;
            return p_node;
        }

        inline void register_book(board *key, int hash, int policy){
            if(book[hash] == NULL){
                book[hash] = node_init(key, policy);
            } else {
                book_node *p_node = book[hash];
                book_node *p_pre_node = NULL;
                p_pre_node = p_node;
                while(p_node != NULL){
                    if(compare_key(key, p_node)){
                        p_node->policy = policy;
                        return;
                    }
                    p_pre_node = p_node;
                    p_node = p_node->p_n_node;
                }
                p_pre_node->p_n_node = node_init(key, policy);
            }
        }

        inline int register_symmetric_book(board b, int policy){
			register_book(&b, b.hash() & book_hash_mask, policy);
            b.white_mirror();
            policy = policy % hw * hw + policy / hw;
            register_book(&b, b.hash() & book_hash_mask, policy);
            b.vertical_mirror();
            policy = hw2_m1 - policy;
            register_book(&b, b.hash() & book_hash_mask, policy);
            b.white_mirror();
            policy = policy % hw * hw + policy / hw;
            register_book(&b, b.hash() & book_hash_mask, policy);
			return 1;
        }

        inline string unzip_str_book(string compressed_data, int repair_num, string replace_from_str, string replace_to_str){
            int i, j, k, repair_idx, str_size;
            string repair_str;
            string replace_str;
            cout << "before unzipping " << compressed_data.size() / 3 << endl;
            string param_compressed1 = compressed_data, param_compressed2;
            repair_idx = 0;
            for (i = 0; i < repair_num * 3; i += 3){
                repair_str = replace_from_str.substr(repair_idx, 6);
                repair_idx += 6;
                replace_str = replace_to_str.substr(i, 3);
                param_compressed2 = u8"";
                str_size = param_compressed1.size();
                for (j = 0; j < str_size; j += 3){
                    if (param_compressed1[j] == replace_str[0] && param_compressed1[j + 1] == replace_str[1] && param_compressed1[j + 2] == replace_str[2]){
                        param_compressed2 += repair_str;
                    } else {
                        for (k = 0; k < 3; ++k)
                            param_compressed2 += param_compressed1[j + k];
                    }
                }
                param_compressed1.swap(param_compressed2);
            }
            cout << "unzipped paired " << param_compressed1.size() / 3 << endl;
            return param_compressed1;
        }

        inline int zip_to_int_book(string param_compressed1, int zip_int[]){
            int i, siz, num;
            siz = param_compressed1.size() / 3;
            for (i = 0; i < siz; ++i){
                num = -413340;
                num += ((int)param_compressed1[i * 3] + 128) * 4096;
                num += ((int)param_compressed1[i * 3 + 1] + 128) * 64;
                num += (int)param_compressed1[i * 3 + 2] + 128;
                zip_int[i * 2] = num / 64;
                zip_int[i * 2 + 1] = num % 64;
            }
            return siz * 2;
        }

        inline bool compare_key(board *aa, book_node *bb){
            return aa->b == bb->b && aa->w == bb->w;
        }
};

book book;