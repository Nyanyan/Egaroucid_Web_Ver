// Egaroucid5 Light
#include <iostream>
#include "common.hpp"
#include "board.hpp"
#include "book.hpp"
#include "evaluate.hpp"
#include "search.hpp"
#include "transpose_table.hpp"
#include "book.hpp"
#include "ai.hpp"

inline void init(){
    mobility_init();
    transpose_table_init();
    evaluate_init();
    book.init();
}
/*
inline void input_board(board *b, int ai_player){
    int i;
    char elem;
    int arr[hw2];
    vacant_lst.clear();
    for (i = 0; i < hw2; ++i){
        cin >> elem;
        if (elem == '.'){
            arr[i] = vacant;
            vacant_lst.emplace_back(hw2_m1 - i);
        } else
            arr[i] = (int)elem - (int)'0';
    }
    b->translate_from_arr(arr, ai_player);
    if (vacant_lst.size() >= 2)
        sort(vacant_lst.begin(), vacant_lst.end(), cmp_vacant);
}

inline double calc_result_value(int v){
    return v;
    //return (double)round((double)v * hw2 / sc_w * 100) / 100.0;
}

inline void print_result(int policy, int value){
    cout << (hw_m1 - policy / hw) << " " << (hw_m1 - policy % hw) << " " << calc_result_value(value) << endl;
}

inline void print_result(search_result result){
    cout << (hw_m1 - result.policy / hw) << " " << (hw_m1 - result.policy % hw) << " " << calc_result_value(result.value) << endl;
}

int main(){
    init();
    board b;
    int ai_player;
    while (true){
        cin >> ai_player;
        int max_depth;
        cin >> max_depth;
        input_board(&b, ai_player);
        cout << mid_evaluate(&b) << endl;
        transpose_table.init_now();
        transpose_table.init_prev();
        //bool use_mpc = false;
        bool use_mpc = max_depth >= 11 ? true : false;
        double use_mpct = 1.5;
        unsigned long long searched_nodes = 0;
        cout << mtd(&b, false, max_depth, -hw2, hw2, use_mpc, use_mpct, &searched_nodes) << endl;
    }
    return 0;
}
*/
inline int input_board(board *bd, const int *arr){
    int i, j;
    unsigned long long b = 0, w = 0;
    int elem;
    int n_stones = 0;
    vacant_lst.clear();
    for (i = 0; i < hw; ++i){
        for (j = 0; j < hw; ++j){
            elem = arr[i * hw + j];
            if (elem != -1){
                b |= (unsigned long long)(elem == 0) << (i * hw + j);
                w |= (unsigned long long)(elem == 1) << (i * hw + j);
                ++n_stones;
            } else{
                vacant_lst.push_back(i * hw + j);
            }
        }
    }
    if (n_stones < hw2_m1)
        sort(vacant_lst.begin(), vacant_lst.end(), cmp_vacant);
    bd->b = b;
    bd->w = w;
    return n_stones;
}

inline double calc_result_value(int v){
    return (double)v;
}

inline void print_result(int policy, int value){
    cout << policy / hw << " " << policy % hw << " " << calc_result_value(value) << endl;
}

inline void print_result(search_result result){
    cout << result.policy / hw << " " << result.policy % hw << " " << calc_result_value(result.value) << endl;
}

inline int output_coord(int policy, int raw_val){
    return 1000 * policy + 100 + raw_val;
}

extern "C" void init_ai(){
    cout << "initializing AI" << endl;
    init();
    cout << "AI iniitialized" << endl;
}

extern "C" double ai(int *arr_board, int level, int ai_player){
    cout << "start AI" << endl;
    int i, n_stones, policy;
    board b;
    search_result result;
    cout << endl;
    n_stones = input_board(&b, arr_board);
    b.n = n_stones;
    b.p = ai_player;
    b.print();
    cout << n_stones - 4 << "moves" << endl;
    result = ai_search(b, level);
    cout << "searched policy " << result.policy << " value " << result.value << " nps " << result.nps << endl;
    double res = output_coord(result.policy, result.value);
    cout << "res " << res << endl;
    return res;
}

extern "C" void calc_value(int *arr_board, int *res, int level, int ai_player){
    ai_player = 1 - ai_player;
    int i, n_stones, policy;
    board b;
    search_result result;
    n_stones = input_board(&b, arr_board);
    b.print();
    b.n = n_stones;
    b.p = ai_player;
    cout << n_stones - 4 << "moves" << endl;
    int tmp_res[hw2];
    vector<int> moves;
    unsigned long long legal = b.mobility_ull();
    for (const int &cell: vacant_lst){
        if (1 & (legal >> cell)){
            moves.emplace_back(cell);
        }
    }
    for (i = 0; i < hw2; ++i)
        tmp_res[i] = -1;
    board nb;
    mobility mob;
    int depth1, depth2;
    bool use_mpc;
    double mpct;
    get_level(level, b.n - 3, &depth1, &depth2, &use_mpc, &mpct);
    unsigned long long searched_nodes = 0;
    if (b.n >= hw2 - depth2 - 1){
        transpose_table.init_now();
        int g;
        for (const int &policy: moves){
            calc_flip(&mob, &b, policy);
            b.move_copy(&mob, &nb);
            g = -mtd(&nb, false, depth2 / 2, -hw2, hw2, use_mpc, mpct, &searched_nodes);
            tmp_res[policy] = -mtd_final(&nb, false, depth2, -hw2, hw2, use_mpc, mpct, g, &searched_nodes);
        }
    } else{
        transpose_table.init_now();
        for (int depth = min(3, max(0, depth1 - 4)); depth <= depth1; ++depth){
            for (const int &policy: moves){
                calc_flip(&mob, &b, policy);
                b.move_copy(&mob, &nb);
                tmp_res[policy] += -mtd(&nb, false, depth2, -hw2, hw2, use_mpc, mpct, &searched_nodes);
                tmp_res[policy] /= 2;
            }
        }
        swap(transpose_table.now, transpose_table.prev);
    }
    for (i = 0; i < hw2; ++i)
        res[10 + i] = max(-64, min(64, tmp_res[i]));
    for (int y = 0; y < hw; ++y){
        for (int x = 0; x < hw; ++x)
            cout << tmp_res[y * hw + x] << " ";
        cout << endl;
    }
    ai_player = 1 - ai_player;
}
