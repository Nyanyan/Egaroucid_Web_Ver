#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#pragma GCC target("sse,sse2,sse3,ssse3,sse4,popcnt,abm,mmx")

// Egaroucid3

#include <iostream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <chrono>
#include <string>
#include <unordered_map>
#include <random>

using namespace std;

#define hw 8
#define hw_m1 7
#define hw_p1 9
#define hw2 64
#define hw22 128
#define hw2_m1 63
#define hw2_mhw 56
#define hw2_p1 65
#define n_line 6561
#define max_evaluate_idx 59049
#define inf 1000000000.0
#define window 1e-10
#define b_idx_num 38

#define book_hash_table_size 16384
#define book_hash_mask (book_hash_table_size - 1)
#define ln_repair_book 27

#define search_hash_table_size 1048576
#define search_hash_mask (search_hash_table_size - 1)

#define n_patterns 11
#define n_phases 4
#define n_dense0 16
#define n_dense1 16

#define mpca 1.089838739292347
#define mpcsd 0.23107466045337674
#define mpct 2.05
#define mpcwindow 1e-10

#define n_all_input 14
#define n_all_dense0 16

#define digit_num 3

int ai_player;
int read_depth;
int win_read_depth;
int book_depth;

struct board{
    int b[b_idx_num];
    int p;
    int policy;
    double v;
    int n;
    int op;
};

struct book_node{
    int k[hw];
    int policy;
    book_node* p_n_node;
};

struct search_node{
    int k[b_idx_num];
    pair<double, double> v;
    search_node* p_n_node;
};

struct search_result{
    int policy;
    double value;
    int depth;
};

const int idx_n_cell[b_idx_num] = {8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3};
const int move_offset[b_idx_num] = {1, 1, 1, 1, 1, 1, 1, 1, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7};
const int global_place[b_idx_num][hw] = {
    {0, 1, 2, 3, 4, 5, 6, 7},{8, 9, 10, 11, 12, 13, 14, 15},{16, 17, 18, 19, 20, 21, 22, 23},{24, 25, 26, 27, 28, 29, 30, 31},{32, 33, 34, 35, 36, 37, 38, 39},{40, 41, 42, 43, 44, 45, 46, 47},{48, 49, 50, 51, 52, 53, 54, 55},{56, 57, 58, 59, 60, 61, 62, 63},
    {0, 8, 16, 24, 32, 40, 48, 56},{1, 9, 17, 25, 33, 41, 49, 57},{2, 10, 18, 26, 34, 42, 50, 58},{3, 11, 19, 27, 35, 43, 51, 59},{4, 12, 20, 28, 36, 44, 52, 60},{5, 13, 21, 29, 37, 45, 53, 61},{6, 14, 22, 30, 38, 46, 54, 62},{7, 15, 23, 31, 39, 47, 55, 63},
    {5, 14, 23, -1, -1, -1, -1, -1},{4, 13, 22, 31, -1, -1, -1, -1},{3, 12, 21, 30, 39, -1, -1, -1},{2, 11, 20, 29, 38, 47, -1, -1},{1, 10, 19, 28, 37, 46, 55, -1},{0, 9, 18, 27, 36, 45, 54, 63},{8, 17, 26, 35, 44, 53, 62, -1},{16, 25, 34, 43, 52, 61, -1, -1},{24, 33, 42, 51, 60, -1, -1, -1},{32, 41, 50, 59, -1, -1, -1, -1},{40, 49, 58, -1, -1, -1, -1, -1},
    {2, 9, 16, -1, -1, -1, -1, -1},{3, 10, 17, 24, -1, -1, -1, -1},{4, 11, 18, 25, 32, -1, -1, -1},{5, 12, 19, 26, 33, 40, -1, -1},{6, 13, 20, 27, 34, 41, 48, -1},{7, 14, 21, 28, 35, 42, 49, 56},{15, 22, 29, 36, 43, 50, 57, -1},{23, 30, 37, 44, 51, 58, -1, -1},{31, 38, 45, 52, 59, -1, -1, -1},{39, 46, 53, 60, -1, -1, -1, -1},{47, 54, 61, -1, -1, -1, -1, -1}
};
vector<vector<int>> place_included;
int pow3[11], pow17[hw];
int mod3[n_line][hw];
int move_arr[2][n_line][hw][2];
bool legal_arr[2][n_line][hw];
int flip_arr[2][n_line][hw];
int put_arr[2][n_line][hw];
int local_place[b_idx_num][hw2];
const double cell_weight[hw2] = {
    0.2880, -0.1150, 0.0000, -0.0096, -0.0096, 0.0000, -0.1150, 0.2880,
    -0.1150, -0.1542, -0.0288, -0.0288, -0.0288, -0.0288, -0.1542, -0.1150,
    0.0000, -0.0288, 0.0000, -0.0096, -0.0096, 0.0000, -0.0288, 0.0000,
    -0.0096, -0.0288, -0.0096, -0.0096, -0.0096, -0.0096, -0.0288, -0.0096,
    -0.0096, -0.0288, -0.0096, -0.0096, -0.0096, -0.0096, -0.0288, -0.0096,
    0.0000, -0.0288, 0.0000, -0.0096, -0.0096, 0.0000, -0.0288, 0.0000,
    -0.1150, -0.1542, -0.0288, -0.0288, -0.0288, -0.0288, -0.1542, -0.1150,
    0.2880, -0.1150, 0.0000, -0.0096, -0.0096, 0.0000, -0.1150, 0.2880
};
int count_arr[n_line];
int count_all_arr[n_line];
int pop_digit[n_line][hw];
int pop_mid[n_line][hw][hw];
int reverse_board[n_line];
int canput_arr[2][n_line];
int surround_arr[2][n_line];
int open_arr[n_line][hw];

vector<int> vacant_lst;
book_node *book[book_hash_table_size];
search_node *search_replace_table[2][search_hash_table_size];
long long searched_nodes;
int f_search_table_idx;
double ev_arr[n_phases][n_patterns][max_evaluate_idx];
double all_dense0[n_phases][n_all_dense0][n_all_input];
double all_bias0[n_phases][n_all_dense0];
double all_dense1[n_phases][n_all_dense0];
double all_bias1[n_phases];

inline unsigned long long calc_hash(const int *p){
    unsigned long long seed = 0;
    for (int i = 0; i < hw; ++i)
        seed += p[i] * pow17[i];
    return seed;
}

inline bool compare_key(const int *a, const int *b){
    for (int i = 0; i < hw; ++i){
        if (a[i] != b[i])
            return false;
    }
    return true;
}

inline void print_board_line(int tmp){
    int j;
    string res = "";
    for (j = 0; j < hw; ++j){
        if (tmp % 3 == 0)
            res = "X " + res;
        else if (tmp % 3 == 1)
            res = "O " + res;
        else
            res = ". " + res;
        tmp /= 3;
    }
    cout << res;
}

inline void print_board(const int* board){
    int i, j, tmp;
    string res;
    for (i = 0; i < hw; ++i){
        tmp = board[i];
        res = "";
        for (j = 0; j < hw; ++j){
            if (tmp % 3 == 0)
                res = "X " + res;
            else if (tmp % 3 == 1)
                res = "O " + res;
            else
                res = ". " + res;
            tmp /= 3;
        }
        cout << res << endl;
    }
    cout << endl;
}

inline int create_one_color(int idx, const int k){
    int res = 0;
    for (int i = 0; i < hw; ++i){
        if (idx % 3 == k){
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
        return res;
    mask = trans(pt, k);
    while (mask && (mask & o)) {
        ++res;
        mask = trans(mask, k);
        if (mask & p)
            return res;
    }
    return 0;
}

inline void init_pow(){
    int idx;
    pow3[0] = 1;
    for (idx = 1; idx < 11; ++idx)
        pow3[idx] = pow3[idx- 1] * 3;
    pow17[0] = 1;
    for (idx = 1; idx < hw; ++idx)
        pow17[idx] = pow17[idx - 1] * 17;
}

inline void init_move(){
    int idx, b, w, place;
    bool surround_flag;
    for (idx = 0; idx < n_line; ++idx){
        b = create_one_color(idx, 0);
        w = create_one_color(idx, 1);
        count_arr[idx] = 0;
        count_all_arr[idx] = 0;
        reverse_board[idx] = 0;
        canput_arr[0][idx] = 0;
        canput_arr[1][idx] = 0;
        surround_arr[0][idx] = 0;
        surround_arr[1][idx] = 0;
        for (place = 0; place < hw; ++place){
            count_arr[idx] += 1 & (b >> place);
            count_arr[idx] -= 1 & (w >> place);
            count_all_arr[idx] += 1 & (b >> place);
            count_all_arr[idx] += 1 & (w >> place);
            reverse_board[idx] *= 3;
            if (1 & (b >> place))
                reverse_board[idx] += 0;
            else if (1 & (w >> place)) 
                reverse_board[idx] += 1;
            else
                reverse_board[idx] += 2;
            surround_flag = false;
            if (place > 0){
                if ((1 & (b >> (place - 1))) == 0 && (1 & (w >> (place - 1))) == 0)
                    surround_flag = true;
            }
            if (place < hw_m1){
                if ((1 & (b >> (place + 1))) == 0 && (1 & (w >> (place + 1))) == 0)
                    surround_flag = true;
            }
            if (1 & (b >> place) && surround_flag)
                ++surround_arr[0][idx];
            else if (1 & (w >> place) && surround_flag)
                ++surround_arr[1][idx];
        }
        for (place = 0; place < hw; ++place){
            move_arr[0][idx][place][0] = move_line_half(b, w, place, 0);
            move_arr[0][idx][place][1] = move_line_half(b, w, place, 1);
            if (move_arr[0][idx][place][0] || move_arr[0][idx][place][1])
                legal_arr[0][idx][place] = true;
            else
                legal_arr[0][idx][place] = false;
            move_arr[1][idx][place][0] = move_line_half(w, b, place, 0);
            move_arr[1][idx][place][1] = move_line_half(w, b, place, 1);
            if (move_arr[1][idx][place][0] || move_arr[1][idx][place][1])
                legal_arr[1][idx][place] = true;
            else
                legal_arr[1][idx][place] = false;
            if (legal_arr[0][idx][place])
                ++canput_arr[0][idx];
            if (legal_arr[1][idx][place])
                ++canput_arr[1][idx];
        }
        for (place = 0; place < hw; ++place){
            flip_arr[0][idx][place] = idx;
            flip_arr[1][idx][place] = idx;
            put_arr[0][idx][place] = idx;
            put_arr[1][idx][place] = idx;
            if (b & (1 << (hw_m1 - place)))
                flip_arr[1][idx][place] += pow3[hw_m1 - place];
            else if (w & (1 << (hw_m1 - place)))
                flip_arr[0][idx][place] -= pow3[hw_m1 - place];
            else{
                put_arr[0][idx][place] -= pow3[hw_m1 - place] * 2;
                put_arr[1][idx][place] -= pow3[hw_m1 - place];
            }
        }
        for (place = 0; place < hw; ++place){
            open_arr[idx][hw_m1 - place] = 0;
            if (place - 1 >= 0){
                if ((1 & (b >> (place - 1))) == 0 && (1 & (w >> (place - 1))) == 0)
                    ++open_arr[idx][hw_m1 - place];
            }
            if (place + 1 < hw){
                if ((1 & (b >> (place + 1))) == 0 && (1 & (w >> (place + 1))) == 0)
                    ++open_arr[idx][hw_m1 - place];
            }
        }
    }
}

inline void init_local_place(){
    int idx, place, l_place;
    for (idx = 0; idx < b_idx_num; ++idx){
        for (place = 0; place < hw2; ++place){
            local_place[idx][place] = -1;
            for (l_place = 0; l_place < hw; ++l_place){
                if (global_place[idx][l_place] == place)
                    local_place[idx][place] = l_place;
            }
        }
    }
}

inline void init_included(){
    int idx, place, l_place;
    for (place = 0; place < hw2; ++place){
        vector<int> included;
        for (idx = 0; idx < b_idx_num; ++idx){
            for (l_place = 0; l_place < hw; ++l_place){
                if (global_place[idx][l_place] == place)
                    included.push_back(idx);
            }
        }
        place_included.push_back(included);
    }
}

inline void init_pop_digit(){
    int i, j;
    for (i = 0; i < n_line; ++i){
        for (j = 0; j < hw; ++j)
            pop_digit[i][j] = (i / pow3[hw_m1 - j]) % 3;
    }
}

inline void init_mod3(){
    int i, j;
    for (i = 0; i < n_line; ++i){
        for (j = 0; j < hw; ++j)
            mod3[i][j] = i % pow3[j];
    }
}

inline void init_pop_mid(){
    int i, j, k;
    for (i = 0; i < n_line; ++i){
        for (j = 0; j < hw; ++j){
            for (k = 0; k < hw; ++k)
                pop_mid[i][j][k] = (i - i / pow3[j] * pow3[j]) / pow3[k];
        }
    }
}

inline board move(const board *b, const int global_place){
    board res;
    res.op = 0;
    int j, place, g_place;
    for (int i = 0; i < b_idx_num; ++i)
        res.b[i] = b->b[i];
    for (const int &i: place_included[global_place]){
        place = local_place[i][global_place];
        for (j = 1; j <= move_arr[b->p][b->b[i]][place][0]; ++j){
            g_place = global_place - move_offset[i] * j;
            for (const int &idx: place_included[g_place]){
                res.b[idx] = flip_arr[b->p][res.b[idx]][local_place[idx][g_place]];
                res.op += open_arr[res.b[idx]][local_place[idx][g_place]];
            }
        }
        for (j = 1; j <= move_arr[b->p][b->b[i]][place][1]; ++j){
            g_place = global_place + move_offset[i] * j;
            for (const int &idx: place_included[g_place]){
                res.b[idx] = flip_arr[b->p][res.b[idx]][local_place[idx][g_place]];
                res.op += open_arr[res.b[idx]][local_place[idx][g_place]];
            }
        }
    }
    for (const int &idx: place_included[global_place]){
        res.b[idx] = put_arr[b->p][res.b[idx]][local_place[idx][global_place]];
        res.op += open_arr[res.b[idx]][local_place[idx][global_place]];
    }
    res.p = 1 - b->p;
    res.n = b->n + 1;
    res.policy = global_place;
    return res;
}

inline void book_hash_table_init(book_node** hash_table){
    for(int i = 0; i < book_hash_table_size; ++i)
        hash_table[i] = NULL;
}

inline book_node* book_node_init(const int *key, int policy){
    book_node* p_node = NULL;
    p_node = (book_node*)malloc(sizeof(book_node));
    for (int i = 0; i < hw; ++i)
        p_node->k[i] = key[i];
    p_node->policy = policy;
    p_node->p_n_node = NULL;
    return p_node;
}

inline void register_book(book_node** hash_table, const int *key, int hash, int policy){
    if(hash_table[hash] == NULL){
        hash_table[hash] = book_node_init(key, policy);
    } else {
        book_node *p_node = hash_table[hash];
        book_node *p_pre_node = NULL;
        p_pre_node = p_node;
        while(p_node != NULL){
            if(compare_key(key, p_node->k)){
                p_node->policy = policy;
                return;
            }
            p_pre_node = p_node;
            p_node = p_node->p_n_node;
        }
        p_pre_node->p_n_node = book_node_init(key, policy);
    }
}

inline int get_book(const int *key){
    book_node *p_node = book[calc_hash(key) & book_hash_mask];
    while(p_node != NULL){
        if(compare_key(key, p_node->k)){
            return p_node->policy;
        }
        p_node = p_node->p_n_node;
    }
    return -1;
}

inline void init_book(){
    int i;
    const string chars = "!#$&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abc";
    unordered_map<char, int> char_keys;
    string param_compressed1 = 
" NN5 6N56- @N56-@8 AN56-@8AP ON56-@E MN56-@EM8 ON56-@EM8O7 DN56-@EM8O7DL =N56-@EM8O7DL=K CN56-@EM8O7DL=KC; <N56-@EM8O7DL=KC;<4 QN56-@EM8O7DL=KC;<4QZ .N56-@EM8O7DL=KC;<4QZ.' AN56-@EM8O7DL=KC;<4QZ.'A/ 3N56-@EM8O7DL=KC;<4QZ.'A/3+ 0N56-@EM8O7DL=KC;<4QZ.'A/3+0U $N56-@EM8O7DL=KC;<4QZ.'A/3+0U$V _N56-@EM8O7"
"DL=KC;<4QZ.'A/3+0U$V_B JN56-@EMU ON56-@EMUOP =N56-@EMUOP=4 VN56-@EMUOP=4V_ WN56-@EMUOP=4V_WX `N567 8N56789 ON56789OE =N56789OM @N56789OP =N56789OP=W @N56789OW @N5678E @N5678U ON5678V =N5678V=E ON5678V=EO9 MN56= @N56=@- 7N56=@. PN56=@7 8N56=@789 <N56=@789<O /N56=@789<O/A .N56=@78A PN56=@78AP/ BN56=@7"
"8AP/BM EN56=@78AP/BMEV WN56=@78AP/BMEVWO 0N56=@78AP/BMEVWO09 &N56=@78AP/BMEVWO0I JN56=@78AP/BMEVWO0IJQ 'N56=@78AP9 BN56=@78AP9BJ RN56=@78AP9BJRO <N56=@78AP9BM /N56=@78AP9BM/V :N56=@78AP9BM/V:0 'N56=@78AP9BM/V:0'O .N56=@78AP9BO /N56=@78API :N56=@78API:B 9N56=@78API:B92 QN56=@78API:B92Q0 (N56=@78API:B"
"92Q0(U <N56=@78API:B9Q ON56=@78API:B9QO2 ZN56=@78API:M 9N56=@78API:M9B /N56=@78APM EN56=@78APMEV /N56=@78APO MN56=@78APOME /N56=@78APOMI XN56=@78APOMIXE QN56=@78E MN56=@78I AN56=@78IAP ON56=@78O PN56=@78OPA 9N56=@78OPA9I BN56=@78OPA9IB/ :N56=@78OPA9IB/:Q 0N56=@78OPA9IB0 :N56=@78OPA9IB0:Q EN56=@78OPA"
"9IB0:QEJ RN56=@78OPA9IB0:QEJRL /N56=@78OPA9IB0:QEJRL/( &N56=@78OPA9IB: /N56=@78OPA9IB:/R 2N56=@78OPA9Q /N56=@78OPA9Q/0 EN56=@78OPU /N56=@78OPV ^N56=@78OPW VN56=@78P ON56=@78POE MN56=@78POEMA IN56=@78POEMAIR QN56=@78POEMX WN56=@78POEMXWI QN56=@78POI AN56=@78POIAQ JN56=@78POIAQJW VN56=@8 7N56=@87/ EN5"
"6=@87/E. 0N56=@870 /N56=@87E ON56=@87M EN56=@87MEP ON56=@87O PN56=@87OP- EN56=@87OP/ EN56=@87OP/E. IN56=@87OP/E.IL DN56=@87OP0 /N56=@87P ON56=@87POE MN56=@A 7N56=@A7/ EN56=@E 4N56=@E4+ MN56=@E4+MD <N56=@E4+MD<L 3N56=@E4+MD<L3; ON56=@E4+MD<L3;O. -N56=@E4- <N56=@E4-<3 MN56=@E4-<3MO .N56=@E4-<7 ON56=@E"
"4-<7O3 8N56=@E4-<7O38I .N56=@E4-<7O38I.$ /N56=@E4-<7O38I.$/; DN56=@E4-<7O8 PN56=@E4-<7O8P3 .N56=@E4-<7O8P3.$ QN56=@E4-<7O8P3.; DN56=@E4-<7O8PD .N56=@E4-<7O8PD./ MN56=@E4-<7O8PD./ML KN56=@E4-<7O8PD./MLKW $N56=@E4-<7O8PD./MLKW$X (N56=@E4-<7O8PL MN56=@E4-<7O8PLMW XN56=@E4-<7O8PX IN56=@E4-<7O8PXID QN56="
"@E4-<7O8PXIQ AN56=@E4-<7OC 3N56=@E4-<7OC3D AN56=@E4-<7OC3DA9 PN56=@E4-<7OC3DA9PB .N56=@E4-<7OC3DA9PB.$ MN56=@E4-<7OD AN56=@E4-<7ODA+ PN56=@E4-<7ODA+PC ;N56=@E4-<7ODA+PC;3 MN56=@E4-<7ODA+PC;3ML 8N56=@E4-<7ODA9 PN56=@E4-<7ODA9PB .N56=@E4-<7ODA9PB.I MN56=@E4-<7ODAQ PN56=@E4-<7ODAQPI .N56=@E4-<7ODAV MN5"
"6=@E4-<7OL MN56=@E4-<7OLM3 8N56=@E4-<7OLM38A $N56=@E4-<7OLM8 .N56=@E4-<7OLMD ;N56=@E4-<7OLMD;8 KN56=@E4-<7OLMD;8K3 .N56=@E4-<7OLMD;8K3.$ PN56=@E4-<7OLMD;8K3.$P/ (N56=@E4-<7OLMD;8K3.$P/(& QN56=@E4-<7OLMD;8K3.$P/(&QC +N56=@E4-<7OLMD;8K3.$P/(&QC+I JN56=@E4-<7OLMD;8K3.$P/(&QC+IJV WN56=@E4-<7OLMD;8K3.$P/"
"(&QI JN56=@E4-<7OLMD;8K3.$P/(&QIJC +N56=@E4-<7OLMD;8K3.$P/(&QIJC+V WN56=@E4-<7OLMD;8K3.$P/(&QIJV WN56=@E4-<7OLMD;8K3.$P/(' QN56=@E4-<7OLMD;8K3.$P/('QC +N56=@E4-<7OLMD;8K3.$P/('QC+I &N56=@E4-<7OLMD;8K3.$P/('QC+I&) ,N56=@E4-<7OLMD;8K3.$P/('QC+I&),X WN56=@E4-<7OLMD;8K3.$P/('QI JN56=@E4-<7OLMD;8K3.$P/('"
"QIJC +N56=@E4-<7OLMD;8K3.$PC /N56=@E4-<7OLMD;8K3.$PC/( QN56=@E4-<7OLMD;8K3.$PC/(QS 0N56=@E4-<7OLMD;8K3.$PC/(QS0U &N56=@E4-<7OLMD;8K3.$PC/(QS0U&' #N56=@E4-<7OLMD;8K3.$PC/(QS0U&'#! +N56=@E4-<7OLMD;8KX PN56=@E4-<7OLMD;8KXP3 IN56=@E4-<7OLMD;8KXP3IC 9N56=@E4-<7OLMD;8KXP3IC9: +N56=@E4-<7OLMD;8KXP3IC9:+A B"
"N56=@E4-<7OLMD;8KXP3IC9:+ABJ /N56=@E4-<7OLMD;8KXP3IC9A :N56=@E4-<7OLMD;K $N56=@E4-<7OLMD;K$+ PN56=@E4-<7OLMD;K$+PA 9N56=@E4-<7OLMD;K$+PA9I .N56=@E4-<7OLMD;K$+PA9I./ CN56=@E4-<7OLMD;K$+PA9I./C3 (N56=@E4-<7OLMD;K$+PA9I./C3(X 8N56=@E4-<7OLMD;X PN56=@E4-<7OLMD;XP9 AN56=@E4-<7OLMD;XP9AV UN56=@E4-<7OLMD;X"
"PA 9N56=@E4-<7OLMD;XPV `N56=@E4-<7OLMD;XPV`U WN56=@E4-<7OLMD;XPV`W UN56=@E4-<7OLMD;XPV`WUK CN56=@E4-<7OLMD;XPV`WUKC3 8N56=@E4-<7OLMD;XPV`WUKC38S aN56=@E4-<7OLMD;XPV`WUKC38Sa_ ^N56=@E4-<7OLMD;XPV`WUKC38Sa_^A .N56=@E4-<7OLMD;XPV`WUKC38Sa_^A.& /N56=@E4-<7OLMW XN56=@E4-<7OM PN56=@E4-<7OMPA .N56=@E4-<7OM"
"PC ;N56=@E4-<7OMPC;3 LN56=@E4-<7OMPC;3LA .N56=@E4-<7OMPC;3LD 8N56=@E4-<7OMPC;3LD8U KN56=@E4-<7OMPC;3LD8UKS ^N56=@E4-<7OMPC;3LD8UKS^9 /N56=@E4-<7OMPC;D KN56=@E4-<7OMPC;DK3 +N56=@E4-<7OMPC;DK3+V ^N56=@E4-<7OMPC;DK3+V^_ `N56=@E4-<7OMPC;DK3+V^_`U WN56=@E4-<7OMPC;DKV UN56=@E4-<7OMPC;DKVUW `N56=@E4-<7OMPC"
";DKVUW`3 +N56=@E4-<7OMPC;DKVUW`9 LN56=@E4-<7OMPC;DKVUW`L _N56=@E4-<7OMPC;DKVUW`L_+ SN56=@E4-<7OMPC;DKVUW`L_+S9 8N56=@E4-<7OMPC;DKVUW`L_+S98A IN56=@E4-<7OMPC;DKVUW`L_+S98AIR XN56=@E4-<7OMPC;DKVUW`L_+S98AIX JN56=@E4-<7OMPC;DKVUW`L_+S98AIXJ0 2N56=@E4-<7OMPC;DKVUW`L_+S98AIXJ02a TN56=@E4-<7OMPC;DKVUW`L_+"
"S98AIXJ02aT[ 3N56=@E4-<7OMPC;DKVUW`L_+S98I /N56=@E4-<7OMPC;DKVUW`L_+S98I/0 QN56=@E4-<7OMPC;DKVUW`L_+S98I/0QX .N56=@E4-<7OMPC;DKVUW`L_+S98I/0QX.' &N56=@E4-<7OMPC;DKVUW`L_+S98I/Q TN56=@E4-<7OMPC;DKVUW`L_+S98I/X TN56=@E4-<7OMPC;DKVUW`L_+S98I/XT0 .N56=@E4-<7OMPC;DKVUW`L_+S98] &N56=@E4-<7OMPC;DKVUW`L_+SI"
" 8N56=@E4-<7OMPC;DKVUW`L_+SI8A 9N56=@E4-<7OMPC;DKVUW`L_+SI8A9/ JN56=@E4-<7OMPC;DKVUW`L_+SI8A9/JQ RN56=@E4-<7OMPC;DKVUW`L_+SI8A9X aN56=@E4-<7OMPC;DKVUW`L_+SI8A9XaQ :N56=@E4-<7OMPC;DKVUW`L_+Sa bN56=@E4-<7OMPC;DKVUW`L_+SabQ IN56=@E4-<7OMPC;DKVUW`L_+SabQI9 XN56=@E4-<7OMPC;DKVUW`L_9 XN56=@E4-<7OMPC;DKVUW"
"`L_9XI .N56=@E4-<7OMPC;DKVUW`L_9XI.$ SN56=@E4-<7OMPC;DKVUW`L_9XI.$SQ RN56=@E4-<7OMPC;DKVUW`L_9XI.$SQR^ 0N56=@E4-<7OMPC;DKVUW`L_9XI.$SQRa bN56=@E4-<7OMPC;DKVUW`L_9XI.& JN56=@E4-<7OMPC;DKVUW`L_9XI.&JQ SN56=@E4-<7OMPC;DKVUW`L_9XI.Q $N56=@E4-<7OMPC;DKVUW`L_a bN56=@E4-<7OMPC;DKVUW`L_ab] XN56=@E4-<7OMPC;D"
"KVUW`_ ^N56=@E4-<7OMPC;DKVUW`_^L aN56=@E4-<7OMPC;DKVUW`_^La9 $N56=@E4-<7OMPC;DKVUW`_^La9$I .N56=@E4-<7OMPC;DKVUW`_^La9$I.+ 3N56=@E4-<7OMPC;DKVUW`_^La9$I.+3S 0N56=@E4-<7OMPC;DKVUW`_^X LN56=@E4-<7OMPC;DKVUW`_^XL9 $N56=@E4-<7OMPC;DKVUW`_^XL9$A .N56=@E4-<7OMPC;DKVUX LN56=@E4-<7OMPC;DKVUXL^ WN56=@E4-<7OM"
"PC;DKVUXL^Wa 8N56=@E4-<7OMPC;DKVU^ `N56=@E4-<7OMPC;DKVU^`_ ]N56=@E4-<7OMPC;V DN56=@E4-<7OMPC;VDL WN56=@E4-<7OMPC;VDLWK SN56=@E4-<7OMPV DN56=@E4-<7OMPVD9 LN56=@E4-<7OMPVD9LI $N56=@E4-<7OMPVD9LI$X ^N56=@E4-<7OMPVD9LI$X^3 .N56=@E4-<7OMPVD9LI$X^3.; CN56=@E4-<7OMPVD9LI$X^3.;C/ (N56=@E4-<7OMPVD9LI$X^3.;C/"
"(W 8N56=@E4-<7OMPVD9LI$X^3.;C/(W80 `N56=@E4-<7OMPVD9LI$X^3.;C/(W80`a UN56=@E4-<7OMPVDC WN56=@E4-<7OMPVDCWK ;N56=@E4-<7OMPVDCWK;L _N56=@E4-<7OMPVDL WN56=@E4-<7OMPVDLW3 8N56=@E4-<7OMPVDLW389 _N56=@E4-<7OMPVDLW389_A UN56=@E4-<7OMPVDLW389_AU; :N56=@E4-<7OMPVDLW389_AU;:J .N56=@E4-<7OMPVDLW389_AU;:J.$ /N5"
"6=@E4-<7OMPVDLW389_AU;:J.$/( CN56=@E4-<7OMPVDLW389_AU;:J.$/(CK IN56=@E4-<7OMPVDLWC 3N56=@E4-<7OMPVDLWC3K ^N56=@E4-<7OMPVDLWC3K^; SN56=@E4-<7OMPVDLWI 9N56=@E4-<7OMPVDLWI93 8N56=@E4-<7OMPVDW UN56=@E4-<7OMPVDWUC ;N56=@E4-<7OMPVDWUL aN56=@E4-<7OMPVDWULaK XN56=@E4-<7OMPVDWULaKX^ CN56=@E4-<7OMPVDWULaKX^C`"
" .N56=@E4-<7OMPVDWULaKX_ CN56=@E4-<7OMPVDWULaKX_C` ^N56=@E4-<7OMPVDWULaKX_C`^9 .N56=@E4-<7OMPVDWULaKX_C`^9.$ 0N56=@E4-<7OMPVDWULaKX_C`^9.$0' AN56=@E4-<7OMPVDWULaKX_C`^A .N56=@E4-<7OMPVDWULaKX_C`^A.$ 'N56=@E4-<7OMPVDWULaKX_C`^Q AN56=@E4-<7OMPVDWULaKX_C`^QA9 $N56=@E4-<7OMPVDWULaKX_C`^QA9$; IN56=@E4-<7"
"OMPVDWULaKX_C`^QA9$;I. 3N56=@E4-<7OMPVDWULaKX` ^N56=@E4-<7OMPVDWULaKX`^_ CN56=@E4-<7OMPVDWULa^ XN56=@E4-<7OMPVDWULa^XK CN56=@E4-<7OMPVDWULa_ `N56=@E4-<7OMPVDWULa_`^ ]N56=@E4-<7OMPVDWULa_`^]I 9N56=@E4-<7OMPVDWULa_`^]I9X bN56=@E4-<7OMPVDWULa_`^]Y IN56=@E4-<7OMPVDWULa` _N56=@E4-<7OMPVDX UN56=@E4-<7OMPV"
"DXUL CN56=@E4-<7OMPVDXULCK WN56=@E4-<7OW /N56=@E4-<7OW/' .N56=@E4-<7OW/8 .N56=@E4-<7OW/8.$ 'N56=@E4-<7OW/D .N56=@E4-<7OW/L .N56=@E4-<; MN56=@E4-<D .N56=@E4-<D.3 CN56=@E4-<D.3C; +N56=@E4-<D.7 MN56=@E4-<D.7M$ ;N56=@E4-<D.7M$;8 ON56=@E4-<D.7M$;8O3 'N56=@E4-<D.7M$;8O3'& #N56=@E4-<D.7M8 LN56=@E4-<D.7M; K"
"N56=@E4-<D.7M;KW +N56=@E4-<D.7ML ON56=@E4-<D.7MLO$ ;N56=@E4-<D.7MLO$;C 3N56=@E4-<D.7MLO$;C3/ SN56=@E4-<D.7MLO& CN56=@E4-<M .N56=@E4-<M.& 7N56=@E4-<M.&7/ 8N56=@E4-<M.&7C ;N56=@E4-<M.&7C;O KN56=@E4-<M.3 ON56=@E4-<M.3O; DN56=@E4-<M.3O;D' $N56=@E4-<M.3O;D'$7 &N56=@E4-<M.3O;D7 CN56=@E4-<M.3O;DC LN56=@E4-"
"<M.3O;DCL' $N56=@E4-<M.3O;DCL'$/ &N56=@E4-<M.3O;DCLX WN56=@E4-<M.3O;DL $N56=@E4-<M.3O;DL$7 UN56=@E4-<M.3O;DL$7U8 0N56=@E4-<M.3O;DL$7U80^ QN56=@E4-<M.3O;DL$7U80^QV AN56=@E4-<M.3O;DL$7U80^QVAX WN56=@E4-<M.3O;DL$7U80^QVAXWa TN56=@E4-<M.3O;DL$7U80^QVAXWaT[ 9N56=@E4-<M.3O;DL$7U80^QVAXWaT[9P ]N56=@E4-<M.3"
"O;DL$7U80^QW AN56=@E4-<M.3O;DL$7U80^QWAX VN56=@E4-<M.3O;DL$7U80^QWAXV_ TN56=@E4-<M.3O;DL$7U80^QWAXV_T[ ]N56=@E4-<M.3O;DL$7U80^QWAXV_T[]/ 'N56=@E4-<M.3O;DV UN56=@E4-<M.3O;DVUW LN56=@E4-<M.3O;DVUWLC KN56=@E4-<M.3O;DW VN56=@E4-<M.3O;DWVC LN56=@E4-<M.3O;DWVL PN56=@E4-<M.3O;DWVLP' ,N56=@E4-<M.3O;DWVLP',X"
" &N56=@E4-<M.3O;DWVLP',X&^ _N56=@E4-<M.3O;DWVLP',X&^_7 8N56=@E4-<M.3O;DWVLP',X&^_78/ ]N56=@E4-<M.3O;DWVLP',X&^_78/]` $N56=@E4-<M.3O;DWVLPX _N56=@E4-<M.7 ON56=@E4-<M.7O$ LN56=@E4-<M.7O$LV 8N56=@E4-<M.7O& DN56=@E4-<M.7O&DL AN56=@E4-<M.7O/ DN56=@E4-<M.7O/DL UN56=@E4-<M.7O/DLU3 8N56=@E4-<M.7O8 PN56=@E4-"
"<M.7OU VN56=@E4-<M.O DN56=@E4-<M.OD/ LN56=@E4-<M.OD7 8N56=@E4-<M.OD78/ QN56=@E4-<M.OD78A JN56=@E4-<M.OD78C KN56=@E4-<M.OD78CK3 LN56=@E4-<M.OD78CK3L' QN56=@E4-<M.OD78CK3L'Q+ TN56=@E4-<M.OD78CK3L'Q+TA VN56=@E4-<M.OD78CK3L'QU ^N56=@E4-<M.OD78CK3L; +N56=@E4-<M.OD78CK3L;+9 :N56=@E4-<M.OD78CK3L;+A $N56=@E"
"4-<M.OD78CK3L;+A$0 9N56=@E4-<M.OD78CK3L;+A$09B :N56=@E4-<M.OD78CK3L;+A$09B:2 (N56=@E4-<M.OD78CK3L;+A$09B:2(/ UN56=@E4-<M.OD78CK3L;+A$09B:2(/UV `N56=@E4-<M.OD78CK3L;+A$09B:2(1 IN56=@E4-<M.OD78CK3LA $N56=@E4-<M.OD78CK3LU $N56=@E4-<M.OD78CKS LN56=@E4-<M.OD78CKSL+ $N56=@E4-<M.OD78CKSL+$U QN56=@E4-<M.OD7"
"8CKSLU QN56=@E4-<M.OD78CKSLUQI RN56=@E4-<M.OD8 LN56=@E4-<M.ODC KN56=@E4-<M.ODCK3 LN56=@E4-<M.ODCK3L' $N56=@E4-<M.ODCK3L'$; +N56=@E4-<M.ODCK3L+ $N56=@E4-<M.ODCK3L9 UN56=@E4-<M.ODCK3L9U8 $N56=@E4-<M.ODCK3L9U8$+ WN56=@E4-<M.ODCK3L; +N56=@E4-<M.ODCK3L;+' $N56=@E4-<M.ODCK3L;+'$7 &N56=@E4-<M.ODCK3L;+'$7&#"
" /N56=@E4-<M.ODCK3L;+'$7&#/( 8N56=@E4-<M.ODCK3L;+7 8N56=@E4-<M.ODCK3L;+78' QN56=@E4-<M.ODCK3L;+78'QA UN56=@E4-<M.ODCK3L;+78'QAUV `N56=@E4-<M.ODCK3L;+78'QAUV`_ ^N56=@E4-<M.ODCK3L;+78'QAUV`_^] XN56=@E4-<M.ODCK3L;+789 AN56=@E4-<M.ODCK3L;+789A: VN56=@E4-<M.ODCK3L;+78A $N56=@E4-<M.ODCK3L;+78A$0 9N56=@E4-"
"<M.ODCK3L;+78A$09& /N56=@E4-<M.ODCK3L;+78A$09&/' JN56=@E4-<M.ODCK3L;+78A$09&/( BN56=@E4-<M.ODCK3L;+78A$09/ (N56=@E4-<M.ODCK3L;+78A$09/(& 'N56=@E4-<M.ODCK3L;+78A$09B :N56=@E4-<M.ODCK3L;+78A$09B:& 'N56=@E4-<M.ODCK3L;+78A$09B:&'/ (N56=@E4-<M.ODCK3L;+78A$09B:&'/(2 IN56=@E4-<M.ODCK3L;+78A$09B:2 (N56=@E4-"
"<M.ODCK3L;+78A$09B:2(& /N56=@E4-<M.ODCK3L;+78A$09B:2(/ UN56=@E4-<M.ODCK3L;+78A$09B:2(/UV `N56=@E4-<M.ODCK3L;+78A$09B:2(/UV`T XN56=@E4-<M.ODCK3L;+78A$09B:2(/UV`TX] IN56=@E4-<M.ODCK3L;+78A$09B:2(/UV`TX]IW PN56=@E4-<M.ODCK3L;+78A$09B:2(/UV`_ ^N56=@E4-<M.ODCK3L;+78A$09B:2(/UV`_^] XN56=@E4-<M.ODCK3L;+78A"
"$09B:2(/UV`_^]Xa WN56=@E4-<M.ODCK3L;+78A$09B:2(/UV`_^]XaWT QN56=@E4-<M.ODCK3L;+78A$09B:2(/UV`_^]XaWTQI RN56=@E4-<M.ODCK3L;+78A$09B:2(1 IN56=@E4-<M.ODCK3L;+78A$09B:2(1I/ PN56=@E4-<M.ODCK3L;+78A$09B:2(1IJ QN56=@E4-<M.ODCK3L;+78A$09B:2(1IP VN56=@E4-<M.ODCK3L;+78A$09B:2(1IPV^ `N56=@E4-<M.ODCK3L;+78A$09B"
":2(1IPV^`U WN56=@E4-<M.ODCK3L;+78A$09B:2(1IPV^`UWa _N56=@E4-<M.ODCK3L;+9 AN56=@E4-<M.ODCK3L;+9A7 8N56=@E4-<M.ODCK3L;+9A78' :N56=@E4-<M.ODCK3L;+9A78':0 /N56=@E4-<M.ODCK3L;+9A78':0/( &N56=@E4-<M.ODCK3L;+9A78':0/(&$ ,N56=@E4-<M.ODCK3L;+9A78':0/(&$,J BN56=@E4-<M.ODCK3L;+9A78: JN56=@E4-<M.ODCK3L;+9A78:J'"
" QN56=@E4-<M.ODCK3L;+9A8 IN56=@E4-<M.ODCK3L;+9A8I' 7N56=@E4-<M.ODCK3L;+9A8I'7B 0N56=@E4-<M.ODCK3L;+9A8I'7B0/ (N56=@E4-<M.ODCK3L;+9A8IP 7N56=@E4-<M.ODCK3L;+9A: IN56=@E4-<M.ODCK3LU $N56=@E4-<M.ODCK3LU$; +N56=@E4-<M.ODCK3LU$;+7 8N56=@E4-<M.ODCKS LN56=@E4-<M.ODCKSL+ $N56=@E4-<M.ODCKSL; VN56=@E4-<M.ODCKS"
"L;VU +N56=@E4-<M.ODCKSLU IN56=@E4-<M.ODCKSLUI+ $N56=@E4-<M.ODCKSLUI+$' /N56=@E4-<M.ODCKSLUI+$'/( ^N56=@E4-<M.ODCKSLUI+$'/(^8 7N56=@E4-<M.ODCKSLUI+$'/(^870 WN56=@E4-<M.ODCKSLUI+$'/(^870WX VN56=@E4-<M.ODCKSLUI+$/ 7N56=@E4-<M.ODCKSLUI+$/7' ;N56=@E4-<M.ODCKSLUI+$/7';3 XN56=@E4-<M.ODCKSLUI+$7 /N56=@E4-<M"
".ODCKSLUI+$7/( VN56=@E4-<M.ODCKSLUI7 PN56=@E4-<M.ODCKSLUI8 7N56=@E4-<M.ODCKSLUI87A PN56=@E4-<M.ODCKSLUI; +N56=@E4-<M.ODCKSLUI;+' $N56=@E4-<M.ODCKSLUI;+'$/ 7N56=@E4-<M.ODCKSLUI;+'$/78 (N56=@E4-<M.ODCKSLUI;+'$8 /N56=@E4-<M.ODCKSLUI;+'$8/( 7N56=@E4-<M.ODCKSLUI;+'$8/(70 9N56=@E4-<M.ODCKSLUI;+'$8/(709& )"
"N56=@E4-<M.ODCKSLUI;+'$8/(709&)P #N56=@E4-<M.ODCKSLUI;+7 PN56=@E4-<M.ODCKSLUI;+8 7N56=@E4-<M.ODCKSLUI;+87' ,N56=@E4-<M.ODCKSLUI;+87',0 $N56=@E4-<M.ODCKSLUI;+87',0$9 AN56=@E4-<M.ODCKSLUIA PN56=@E4-<M.ODCKSLUIAP7 8N56=@E4-<M.ODCKSLUIAP789 BN56=@E4-<M.ODCKSLUIAP789B: QN56=@E4-<M.U DN56=@E4-<M.V DN56=@E"
"4-<M.VDC ON56=@E4. MN56=@E4.M3 -N56=@E4.M3-< 7N56=@E4.M8 /N56=@E4.M8/< 7N56=@E4.M< ON56=@E4.M<OD -N56=@E4.M<OD-L CN56=@E4.M<OD-LC; 3N56=@E4.MA ON56=@E4.MAOW /N56=@E4.MD ;N56=@E4.ML <N56=@E4.ML<D ON56=@E4.ML<DO- ;N56=@E4.ML<DO-;3 7N56=@E4.ML<DO-;370 PN56=@E4.ML<DO-;370PV UN56=@E4.ML<DO-;8 'N56=@E4.ML"
"<DO-;8'C &N56=@E4.ML<DO-;K CN56=@E4.ML<DO-;KC+ SN56=@E4.ML<DO-;X 7N56=@E4.ML<DO-;X78 PN56=@E4.ML<DO8 CN56=@E4.ML<V KN56=@E4.ML<VKD WN56=@E4.MU WN56=@E4.MV /N56=@E4.MV/< -N56=@E4/ 7N56=@E4/7- LN56=@E4/7-LD ON56=@E4/7-LDO0 MN56=@E4/7-LDO0M< PN56=@E4/7-LM .N56=@E4/7-LM.& <N56=@E4/7-LM.&<O (N56=@E4/7-LM"
".&<O(' VN56=@E4/7-LM.&<O('VD 0N56=@E4/7-LM.&<O('VD0K $N56=@E4/7-LM.&<O('VD0K$W 8N56=@E4/7-LM.&<O('VD0K$W8C PN56=@E4/7-LM.&<O('VD0K$W8CP; `N56=@E4/7-LM.&<O('VD0K$W8CP;`I AN56=@E4/7-LM.&<O('VD0K$W8CP;`IAX JN56=@E4/7-LM.&<O(D KN56=@E4/7-LM.&<O(DK' 8N56=@E4/7-LM.&<O(DK'8) #N56=@E4/7. <N56=@E4/7.<3 -N56="
"@E4/7.<3-& MN56=@E4/7.<8 0N56=@E4/7.<80( AN56=@E4/7.<80(AL MN56=@E4/7.<80(ALM9 PN56=@E4/7.<80(ALM9PU -N56=@E4/7.<80(ALM9PU-; 3N56=@E4/7.<80(ALM9PU-;3C DN56=@E4/7.<80(ALM9PU-;3CDK VN56=@E4/7.<80(ALM9PU-;3CDKV+ QN56=@E4/7.<80(ALM9PU-;3CDKV+QO :N56=@E4/7.<80(ALM9PU-;3CDKV+QO:I JN56=@E4/7.<803 &N56=@E4/"
"7.<803&$ 'N56=@E4/7.<803&$'( ON56=@E4/7.<803&$'(OM PN56=@E4/7.<9 -N56=@E4/7.<D -N56=@E4/78 DN56=@E4/78D0 MN56=@E4/78D0M. ON56=@E4/78D0M.OW PN56=@E4/78D0MV WN56=@E4/78D0MVW. ON56=@E4/78D0MVW.OU PN56=@E4/78D0MVW.OUPa `N56=@E4/78D0MVW.OUPa`_ IN56=@E4/78D< MN56=@E48 DN56=@E48D< MN56=@E48D<M- 3N56=@E48D<"
"M. 3N56=@E48D<MC ON56=@E48D<MCO/ ;N56=@E48D<MV 7N56=@E48D<MW ON56=@E48D<MWOV ;N56=@E48D<MWOV;U PN56=@E48DV MN56=@E49 DN56=@E49D< MN56=@E4A DN56=@E4AD< MN56=@E4ADV MN56=@E4ADVML UN56=@E4V DN56=@E4VD< MN56=@E4VDC MN56=@E4VDCM< ON56=@E4W DN56=@E4WDC MN56=@E4WDCM< ON56=@M 7N56=@M7. EN56=@M7.EV <N56=@M7."
"EV<4 -N56=@M7A EN56=@M7AE- 8N56=@M7P ON56=@M7PO. AN56=@M7PO.AX WN56=@M7PO.AXW0 IN56=@M7POA EN56=@M7V EN56=@M7VEO PN56=@M7VEOP. WN56=@O 4N56=@O4- 7N56=@O4-7. <N56=@O4-7.</ 8N56=@O4-7.</89 (N56=@O4-7.</8E DN56=@O4-7.</8M (N56=@O4-7.</8M(' 0N56=@O4-7.<9 MN56=@O4-7.<9ME XN56=@O4-7.<9MEX0 DN56=@O4-7.<9ME"
"X0DW PN56=@O4-7.<9MEX0DWPI aN56=@O4-7.<9MEX0DWPIaV AN56=@O4-7.<9MEX0DWPIaVA8 `N56=@O4-7.<9MEX0DWPIaVA8`U /N56=@O4-7.<E MN56=@O4-7.<EM/ XN56=@O4-7.<EM/XV WN56=@O4-7.<EM0 8N56=@O4-7.<EMC ;N56=@O4-7.<EMC;D KN56=@O4-7.<EMC;DK0 LN56=@O4-7.<EMC;DK0LV &N56=@O4-7.<EMV LN56=@O4-7.<EMVL/ 8N56=@O4-7.<EMVL9 XN5"
"6=@O4-7.<EMVL9XP /N56=@O4-7.<EMVL9XP/8 AN56=@O4-7.<EMVL9XP/8A( 0N56=@O4-7.<EMVL9XP/8A(0' :N56=@O4-7.<EMVLD $N56=@O4-7.<EMVLK PN56=@O4-7.<EMVLU $N56=@O4-7.<EMVLU$& 'N56=@O4-7.<EMVLU$&'A XN56=@O4-7.<M /N56=@O4-78 PN56=@O4-7A EN56=@O4. EN56=@O4.E8 /N56=@O4.E< -N56=@O4.E<-3 7N56=@O4.E<-37/ &N56=@O4.E<-3"
"7/&$ (N56=@O4.E<-37/&$(0 #N56=@O4.E<-37/&$(0#L MN56=@O4.EA /N56=@O4.EA/< -N56=@O4.ED MN56=@O4.EDM< -N56=@O4.EDM<-W CN56=@O4.EDM<-WC; 3N56=@O4.EDM<-WC;3$ 7N56=@O4.EL MN56=@O4.ELMD 7N56=@O4.EM 7N56=@O4.EV /N56=@O4/ EN56=@O4/E8 7N56=@O4/E87< 3N56=@O4/E< .N56=@O4/E<.D MN56=@O4/E<.DMW CN56=@O4/E<.DMWC7 V"
"N56=@O4/E<.DMWC7V8 KN56=@O4/E<.DMWC7V; 3N56=@O4/E<.M CN56=@O4/E<.MC7 DN56=@O4/E<.MC7D8 AN56=@O4/E<.MC7D8A$ LN56=@O4/E<.MC8 LN56=@O4/E<.MC8LD KN56=@O4/E<.MC8LDK7 (N56=@O4/E<.MC8LDK7($ 'N56=@O4/E<.MC8LDK7($'- 0N56=@O4/E<.MC8LDK7($'-0; 3N56=@O4/E<.MC8LDK7($'-0;3& #N56=@O4/E<.MC8LDK7($'-0;3&#, UN56=@O4/"
"E<.MC8LDK7($'-0;3&#,UV `N56=@O4/E<.MC8LDK7($'-0;3&#,UV`T QN56=@O4/E<.MC8LDK7($'-0;3&#,UV`TQ] [N56=@O4/E<.MC8LDK7($'-0;3&#,UV`TQ][S PN56=@O4/E<.MC8LDK7($'-0;3&#,UV`TQ][SPI JN56=@O4/E<.MC8LDK7($'-0;3&#,UV`TQ][SPIJX WN56=@O4/E<.MC8LDK7($'-0;3&#,UV`TQ][SPIJXW^ _N56=@O4/E<.MC8LDK7($'-0;3&#,UV`TQ][SPIJXW^"
"_a bN56=@O4/E<.MC8LDK7($'-0;3&#,UV`TQ][SPIJXW^_abB RN56=@O4/E<.W 8N56=@O4/EL 7N56=@O4/EM 7N56=@O4/EM7. PN56=@O4/EM7.P8 -N56=@O4/EM7.P8-9 UN56=@O4/EM7.P8-< &N56=@O4/EM7.P8-<&D ;N56=@O4/EM7.P8-A VN56=@O4/EM7.P8-AVI WN56=@O4/EM7.P8-AVIW< LN56=@O4/EM7.P8-AVIW<LD BN56=@O4/EM7.PD <N56=@O4/EM7.PD<- $N56=@O"
"4/EM7.PD<-$& 'N56=@O4/EM7.PD<-$8 'N56=@O4/EM7.PD<-$8'I 0N56=@O4/EM78 PN56=@O4/EU 7N56=@O4/EU7. PN56=@O4/EU7.P8 -N56=@O4/EU7.P8-A IN56=@O4/EV 7N56=@O4/EV7. PN56=@O4/EV7.P8 -N56=@O48 EN56=@O48E. /N56=@O49 DN56=@O4A EN56=@O4M DN56=@O4MD- /N56=@O4MD. 7N56=@O4MD.7/ 8N56=@O4MD; KN56=@O4MD;K< PN56=@O4MD;K<"
"PI EN56=@O4MD< EN56=@O4MDA PN56=@O4U DN56=@O4V EN56=@O4VE. /N56=@O4VED MN56=@P 8N56=@P8- IN56=@P8-IA 7N56=@P8-IQ 7N56=@P8-IQ7J EN56=@P8-IQ7JEA .N56=@P8-IQ7JEA.$ &N56=@P8. IN56=@P8.IO VN56=@P8.IOVJ AN56=@P8/ ON56=@P8/OE MN56=@P87 <N56=@P87<D ON56=@P87<DOA IN56=@P87<DOAI9 QN56=@P87<DOAI; QN56=@P87<DOA"
"I;Q9 0N56=@P87<DOE MN56=@P87<DOEMA IN56=@P87<DOEMI QN56=@P87<DOI /N56=@P87<DOM /N56=@P87<DOM/E -N56=@P87<DOM/E-4 3N56=@P87<DOM/E-43A KN56=@P87<DOM/E-C 4N56=@P87<DOM/E-C4+ LN56=@P87<DOW IN56=@P87<DOWIA /N56=@P87<DOWIA/. 0N56=@P87<DOWIE UN56=@P87<DOWIEUA /N56=@P87<DOWIEUA/. &N56=@P87<DOWIEUM /N56=@P87"
"<DOWIEUM/C LN56=@P87<DOWIEUM/Q .N56=@P87<DOWIEUQ /N56=@P87<DOWIEUQ/M .N56=@P87<DOWIEUV `N56=@P87<DOWIEUV`_ aN56=@P87<DOWIEUV`_ab MN56=@P87<DOWIEUV`_abMX QN56=@P87<DOWIQ /N56=@P87<DOWIQ/M XN56=@P87<DOWIQ/MX' _N56=@P87<DOWIQ/MXB `N56=@P87<DOWIQ/MXB`E AN56=@P87<E ON56=@P87<EO; MN56=@P87<I QN56=@P87<IQ9"
" ON56=@P87<IQ9OD 4N56=@P87<IQ9OD4X 0N56=@P87<IQD ON56=@P87<IQDO9 4N56=@P87<IQDO94X JN56=@P87<IQDO94XJB KN56=@P87<IQDO94XJBKR AN56=@P87<V ON56=@P89 ON56=@P89OE MN56=@P8A IN56=@P8AI7 :N56=@P8AI7:0 9N56=@P8AI7:09J /N56=@P8AI7:09J/M EN56=@P8AI7:9 0N56=@P8AI7:90' /N56=@P8AI7:90'/( EN56=@P8AI7:90'/(EJ .N5"
"6=@P8AI7:90( /N56=@P8AI7:90O BN56=@P8AI7:90OBQ WN56=@P8AI7:90R JN56=@P8AI7:90RJB ZN56=@P8AI7:M 9N56=@P8AI7:M9R JN56=@P8AI7:M9RJV XN56=@P8AI7:O QN56=@P8AI7:R JN56=@P8AI7:RJU VN56=@P8AI7:U VN56=@P8AI7:UV9 0N56=@P8AI7:UVE 9N56=@P8AI7:UVE9O MN56=@P8AI7:UVE9OMQ WN56=@P8AI7:UVM XN56=@P8AIO EN56=@P8E ON56="
"@P8EO. MN56=@P8EO7 IN56=@P8EO7IX VN56=@P8EO7IXVA MN56=@P8EOA 7N56=@P8EOA7- MN56=@P8EOA7-M. /N56=@P8EOW AN56=@P8EOX WN56=@P8EOXW7 MN56=@P8O WN56=@P8OW. EN56=@P8OW.EA 7N56=@P8OW.ED IN56=@P8OW.EDIU AN56=@P8OW.EDIUA7 /N56=@P8OW.EDIUA7/( &N56=@P8OW.EL AN56=@P8OW.EX 7N56=@P8OW.EX7D MN56=@P8OW7 AN56=@P8OW7"
"AI 9N56=@P8OW7AI9B :N56=@P8OW7AQ /N56=@P8OW7AV XN56=@P8OW7AVXI EN56=@P8OW7AVXIEM 9N56=@P8OW7AX VN56=@P8OWA 7N56=@P8OWM QN56=@P8OWMQ. VN56=@P8OWMQ.VX UN56=@P8OWMQ.VXU7 IN56=@P8OWMQ.Va `N56=@P8OWMQ/ IN56=@P8OWMQ/I7 VN56=@P8OWMQ/I7VX UN56=@P8OWMQI XN56=@P8OWMQIX/ VN56=@P8OWMQIX/V7 JN56=@P8OWMQIX9 AN56="
"@P8OWMQIXA JN56=@P8OWMQIXAJ_ EN56=@P8OWMQIXAJ_E: 7N56=@P8OWMQIXAJ_E:7B RN56=@P8OWMQIXR UN56=@P8OWMQIXRUV ^N56=@P8OWMQIXRUV^. DN56=@P8OWMQIXRUV^.D9 EN56=@P8OWMQIXRUV^.D9E7 4N56=@P8OWMQIXRUV^.D9E74a BN56=@P8OWMQIXRU_ EN56=@P8OWMQIXRUa VN56=@P8OWMQIX_ EN56=@P8OWMQIX` RN56=@P8OWMQIX`R. EN56=@P8OWMQIX`R."
"E7 aN56=@P8OWMQIX`R.E< UN56=@P8OWMQIX`R.E<U7 aN56=@P8OWMQIX`R.E<U7aV _N56=@P8OWMQIX`R.E<U7aV_^ ]N56=@P8OWMQIX`R.E<U7aV_^]9 0N56=@P8OWMQIX`R.E<UV LN56=@P8OWMQIX`R.E<UVL7 DN56=@P8OWMQIX`R.E<UVL7D9 aN56=@P8OWMQIX`R.E<UVL7D; CN56=@P8OWMQIX`R.E<UVL7D;C^ aN56=@P8OWMQIX`R.E<UVL7DA aN56=@P8OWMQIX`R.E<UVL7DA"
"aC /N56=@P8OWMQIX`R.E<UVL7DAaC/9 :N56=@P8OWMQIX`R.E<UVL7DAaC/9:B JN56=@P8OWMQIX`R.E<UVL7DAaC/9:BJ^ 2N56=@P8OWMQIX`R.E<U^ aN56=@P8OWMQIX`R.EL DN56=@P8OWMQIX`R.ELD7 aN56=@P8OWMQIX`R.ELD7ab UN56=@P8OWMQIX`R.ELD7abUA 0N56=@P8OWMQIX`R.ELD7abUA0/ <N56=@P8OWMQIX`R/ EN56=@P8OWMQIXa AN56=@P8OWMQIXaA. _N56=@P"
"8OWMQIXaA._7 VN56=@P8OWMQIXaA._7V9 0N56=@P8OWMQIXaA/ _N56=@P8OWMQIXaA/_7 VN56=@P8OWMQIXaA/_7V9 0N56=@P8OWMQIXaA/_7V90B JN56=@P8OWMQIXaA/_7V90BJR :N56=@P8OWMQR UN56=@P8OWMQ_ EN56=@P8OWMQ_E< DN56=@P8OWMQ_E<D7 /N56=@P8OWMQ_E<D7/- 0N56=@P8OWMQ_EL XN56=@P8OWMQ_ELXI RN56=@P8OWMQ` EN56=@P8OWMQ`E< DN56=@P8O"
"WMQ`EL <N56=@P8OWMQ`EL<4 DN56=@P8OWMQ`EL<4D; 3N56=@P8OWMQa XN56=@P8OWMQaX. EN56=@P8OWMQaX.E7 <N56=@P8OWMQaX.E7<4 /N56=@P8OWMQaX.E7<4/( -N56=@P8OWMQaX.E7<4/(-I UN56=@P8OWMQaX.E7<4/(-IUA &N56=@P8OWMQaX.E7<4/C $N56=@P8OWMQaX.E7<4/C$D -N56=@P8OWMQaX.E7<4/C$I AN56=@P8OWMQaX.E7<4/C$IA0 DN56=@P8OWMQaX.E7<4"
"/C$IA0D; UN56=@P8OWMQaX.E7<4/C$IA0D;UV LN56=@P8OWMQaX.E7<4/C$IA0D;UVL^ KN56=@P8OWMQaX.E7<4/C$IA0D;UVL^K: -N56=@P8OWMQaX.E7<4/C$IA0D;UVL^K:-J BN56=@P8OWMQaX.E7<C ;N56=@P8OWMQaX.E7<I UN56=@P8OWMQaX.E7<R `N56=@P8OWMQaX.E< _N56=@P8OWMQaX.E<_7 /N56=@P8OWMQaX.E<_7/9 AN56=@P8OWMQaX.E<_7/9AI LN56=@P8OWMQaX."
"E<_7/9AILU VN56=@P8OWMQaX.E<_7/I BN56=@P8OWMQaX.E<_7/IB9 AN56=@P8OWMQaX.E<_7/IB9AJ 0N56=@P8OWMQaX.E<_7/IB9AJ0' RN56=@P8OWMQaX.E<_7/IBV AN56=@P8OWMQaX/ _N56=@P8OWMQaX/_I AN56=@P8OWMQaX/_IA7 VN56=@P8OWMQaXI AN56=@P8OWMQaXIA. _N56=@P8OWMQaXIA._7 VN56=@P8OWMQaXIA._7V9 0N56=@P8OWMQaXIA/ _N56=@P8OWMQaXIA/"
"_7 VN56=@P8OWMQaXIA/_7VB JN56=@P8OWMQaXIA/_7VBJR 'N56=@P8OWMQaXIA/_7VBJR'0 UN56=@P8OWMQaXIA/_7VBJR'0U^ EN56=@P8OWMQaXIA7 EN56=@P8OWMQaXIA7E9 4N56=@P8OWMQaXIA7E94: BN56=@P8OWMQaXIA7E94:B< ;N56=@P8OWMQaXIA7E94:B<;D _N56=@P8OWMQaXIA7E94:B<;D_J RN56=@P8OWMQaXIAB JN56=@P8OWMQaXIABJR VN56=@P8OWMQaXIABJRV0"
" 7N56=@P8OWMQaXIABJRV07. LN56=@P8OWMQaXIABJRV07.LE `N56=@P8OWMQaXIABJRV9 7N56=@P8OWMQaXIABJRV970 <N56=@P8OWMQaXIABJRV970<^ _N56=@P8OWMQaXR EN56=@P8OWMQaXRE7 <N56=@P8OWMQaXRE7<` VN56=@P8OWMQaXRE7<`VL DN56=@P8OWMQaXRE< DN56=@P8OWMQaXRE<D7 /N56=@P8OWMQaXRE<D7/4 ;N56=@P8OWMQaXRE<D7/4;. 0N56=@P8OWMQaXREL"
" DN56=@P8OWMQaXRELD7 /N56=@P8OWMQaXRELD7/. 0N56=@P8OWV QN56=@P8OWVQ. XN56=@P8OWVQ/ XN56=@P8OWVQ/X7 &N56=@P8OWVQ/X` EN56=@P8OWVQ/X`EM .N56=@P8OWVQI XN56=@P8OWVQIX. ^N56=@P8OWVQIX.^7 /N56=@P8OWVQIX.^R BN56=@P8OWVQIX.^RBJ ZN56=@P8OWVQIX.^` EN56=@P8OWVQIX/ EN56=@P8OWVQIX/E7 9N56=@P8OWVQIX/E79: AN56=@P8O"
"WVQIX/E79:AB 0N56=@P8OWVQIX/E79:AB0U JN56=@P8OWVQIX/E79A JN56=@P8OWVQIX/E79AJ: .N56=@P8OWVQIX/E79AJ:.Z 'N56=@P8OWVQIX/E79AJ:.Z'a UN56=@P8OWVQIX/E79AJ:.a BN56=@P8OWVQIX/E79AJ:.aBU `N56=@P8OWVQIX/E79AJ:.aBU`_ ^N56=@P8OWVQIX/E79AJa .N56=@P8OWVQIX/E79AJa.: BN56=@P8OWVQIX/E79AJa.:BU ^N56=@P8OWVQIX/E79AJa"
".:BU^R MN56=@P8OWVQIX/E79AJa.D LN56=@P8OWVQIX/E79AJa.DLU MN56=@P8OWVQIX/E79AJa.DLUM` ^N56=@P8OWVQIX/E79AJa.DL` 4N56=@P8OWVQIX/E79AJa.DL`4U MN56=@P8OWVQIX/E79AJa.DL`4UM< CN56=@P8OWVQIX/E79AJa.DL`4UM<CB RN56=@P8OWVQIX/E79AJa.DL`4UM<CBR: 2N56=@P8OWVQIX/E79AJa.` (N56=@P8OWVQIX/EL ^N56=@P8OWVQIX/EL^7 9N5"
"6=@P8OWVQIX/EM LN56=@P8OWVQIX/EMLa 7N56=@P8OWVQIX7 AN56=@P8OWVQIX9 BN56=@P8OWVQIX9BA 7N56=@P8OWVQIXR ^N56=@P8OWVQIXR^/ EN56=@P8OWVQIXR^/E7 AN56=@P8OWVQIXR^` EN56=@P8OWVQIXR^`EL UN56=@P8OWVQIXR^`ELUM _N56=@P8OWVQIXR^`ELUM_] JN56=@P8OWVQIX` EN56=@P8OWVQIX`EL RN56=@P8OWVQIX`ELR. MN56=@P8OWVQIX`EM AN56="
"@P8OWVQIX`EMA. ^N56=@P8OWVQIX`EMA.^R JN56=@P8OWVQIX`EMA.^RJ7 LN56=@P8OWVQIX`EMA.^RJ7La 9N56=@P8OWVQIX`EMA.^RJ7La9B /N56=@P8OWVQIX`EMA.^U LN56=@P8OWVQIX`EMA.^UL7 aN56=@P8OWVQIX`EMA.^UL7aD ]N56=@P8OWVQIX`EMA.^UL7ab DN56=@P8OWVQIX`EMA.^UL7abD< _N56=@P8OWVQIX`EMA.^UL7abD<_] 3N56=@P8OWVQIX`EMA.^UL7abD<_]"
"3J RN56=@P8OWVQIX`EMA.^_ aN56=@P8OWVQIX`EMA.^_a7 UN56=@P8OWVQIX`EMA.^_a7UY LN56=@P8OWVQIX`EMA.^_a7UYLJ /N56=@P8OWVQIX`EMA.^_a7UYLJ/B cN56=@P8OWVQIX`EMA.^_a7UYLJ/Bc9 :N56=@P8OWVQIX`EMA.^_a7UYLJ/Bc9:Z RN56=@P8OWVQIX`EMA/ ^N56=@P8OWVQIX`EMA/^U .N56=@P8OWVQIXa RN56=@P8OWVQIXaR. _N56=@P8OWVQIXaR._7 /N56="
"@P8OWVQIXaR._7/( ^N56=@P8OWVQIXaR/ _N56=@P8OWVQIXaR/_7 ^N56=@P8OWVQIXaR/_7^9 MN56=@P8OWVQIXaR/_7^Y UN56=@P8OWVQIXaR/_7^YU` EN56=@P8OWVQIXaR/_7^YU`EL DN56=@P8OWVQIXaR/_7^YU`ELDK 0N56=@P8OWVQIXaRB AN56=@P8OWVQIXaRBA. _N56=@P8OWVQIXaRBA._7 ^N56=@P8OWVQIXaRBA: JN56=@P8OWVQIXaRBA:JZ EN56=@P8OWVQIXaRb ^N5"
"6=@P8OWVQIXaRb^_ EN56=@P8OWVQIXaRb^_E< UN56=@P8OWVQIXaRb^_E<U. DN56=@P8OWVQIXaRb^_EM UN56=@P8OWVQIXaRb^_EMU. DN56=@P8OWVQIXaRb^_EMU.D7 <N56=@P8OWVQR ^N56=@P8OWVQR^X IN56=@P8OWVQR^XI` EN56=@P8OWVQX EN56=@P8OWVQXEL DN56=@P8OWVQXEM UN56=@P8OWVQXEMUI DN56=@P8OWVQXEMU^ `N56=@P8OWVQXEMU^`_ aN56=@P8OWVQ_ E"
"N56=@P8OWVQ_E7 <N56=@P8OWVQ_E< XN56=@P8OWVQ_E<X. MN56=@P8OWVQ_EL <N56=@P8OWVQ_EM UN56=@P8OWVQ_EMU. 4N56=@P8OWVQ_EMU.47 -N56=@P8OWVQ_EMU.4< 7N56=@P8OWVQ_EMU.4<7^ ;N56=@P8OWVQ_EMU7 ^N56=@P8OWVQ_EMUI DN56=@P8OWVQ_EMUID4 LN56=@P8OWVQ_EMUID4L^ 7N56=@P8OWVQ_EMUID4L^7< RN56=@P8OWVQ_EMUID< LN56=@P8OWVQ_EMUI"
"D<LK `N56=@P8OWVQ_EMUIDA <N56=@P8OWVQ_EMUIDA<7 LN56=@P8OWVQ_EMUID^ LN56=@P8OWVQ_EMUID^LK JN56=@P8OWVQ_EMUID^LKJ7 /N56=@P8OWVQ_EMU^ XN56=@P8OWVQ_EMU^X. 4N56=@P8OWVQ_EMU^X.47 -N56=@P8OWVQ_EMU^X.47-< ;N56=@P8OWVQ_EMU^X.4< 7N56=@P8OWVQ_EMU^X.4<7- LN56=@P8OWVQ_EMU^X7 aN56=@P8OWVQ_EMU^X7a9 DN56=@P8OWVQ_EM"
"U^X7a` ]N56=@P8OWVQ_EMU^X7a`]9 LN56=@P8OWVQ_EMU^X7a`]9LA IN56=@P8OWVQ_EMU^X7a`]9LD ;N56=@P8OWVQ_EMU^X7a`]9LD;A IN56=@P8OWVQ_EMU^X7a`]9LD;AIJ <N56=@P8OWVQ_EMU^X7a`]9LD;AIJ<3 +N56=@P8OWVQ_EMU^XI DN56=@P8OWVQ_EMU^XR `N56=@P8OWVQ_EMU^XR`a LN56=@P8OWVQ_EMU^XR`aL7 DN56=@P8OWVQ_EMU^XR`aL7D< 0N56=@P8OWVQ_EM"
"U^Xb DN56=@P8OWVQ_EMU^XbD7 /N56=@P8OWVQ_EMU^XbD7/4 <N56=@P8OWVQ_EMU^XbD7/4<; KN56=@P8OWVQ_EMU^XbD7/< LN56=@P8OWVQ_EMU^XbD7/<LK CN56=@P8OWVQ_EMU^XbD7/<LKC; 0N56=@P8OWVQ_EMU^XbD< LN56=@P8OWVQ_EMU^XbD<L7 3N56=@P8OWVQ_EMU^XbD<L73R /N56=@P8OWVQ_EMU^XbD<LK CN56=@P8OWVQ_EMU^XbD<LKC; 7N56=@P8OWVQ_EMU^XbDL <"
"N56=@P8OWVQ_EMU^XbDL<3 ;N56=@P8OWVQ_EMU^XbDL<3;. /N56=@P8OWVQ_EMU^XbDL<3;7 /N56=@P8OWVQ_EMU^XbDL<3;7/- KN56=@P8OWVQ_EMU^XbDL<3;7/-K4 $N56=@P8OWVQ_EMU^XbDL<3;7/. &N56=@P8OWVQ_EMU^XbDL<3;C TN56=@P8OWVQ_EMU^XbDL<3;CT. /N56=@P8OWVQ_EMU^XbDL<3;CT./4 &N56=@P8OWVQ_EMU^XbDL<3;CT./7 $N56=@P8OWVQ_EMU^XbDL<3;C"
"T4 KN56=@P8OWVQ_EMU^XbDL<3;CT4K[ 7N56=@P8OWVQ_EMU^XbDL<3;CT7 /N56=@P8OWVQ_EMU^XbDL<7 SN56=@P8OWVQ_EMU^XbDL<7S4 TN56=@P8OWVQ` EN56=@P8OWVQ`E< XN56=@P8OWVQ`EL XN56=@P8OWVQ`ELX. MN56=@P8OWVQ`ELX.MD aN56=@P8OWVQ`EM UN56=@P8OWVQ`EMUI _N56=@P8OWVQa XN56=@P8OWVQaX. _N56=@P8OWVQaX._7 /N56=@P8OWVQaX._7/( ^N5"
"6=@P8OWVQaX._7/(^I BN56=@P8OWVQaX._7/(^IBY AN56=@P8OWVQaX._7/(^IBYA0 $N56=@P8OWVQaX._7/(^IBYA0$- 'N56=@P8OWVQaX._7/(^IBYA0$-'& )N56=@P8OWVQaX._7/(^IBYA0$-'&)` 9N56=@P8OWVQaX._7/(^IBYA0$-'&)`9U #N56=@P8OWVQaX/ EN56=@P8OWVQaX/EM .N56=@P8OWVQaX/EM.- <N56=@P8OWVQaX/EM.7 -N56=@P8OWVQaX/EM.7-& $N56=@P8OWV"
"QaX/EM.I RN56=@P8OWVQaX/EM.IR7 -N56=@P8OWVQaXI RN56=@P8OWVQaXIR. `N56=@P8OWVQaXIR.`_ AN56=@P8OWVQaXIR.`_A7 EN56=@P8OWVQaXIR.`_A7E< /N56=@P8OWVQaXIR.`_A7E</( MN56=@P8OWVQaXIR/ `N56=@P8OWVQaXIR/`_ AN56=@P8OWVQaXIR/`_A7 .N56=@P8OWVQaXIR/`_A7.& EN56=@P8OWVQaXIRB AN56=@P8OWVQaXIRBA. _N56=@P8OWVQaXIRBA._7"
" ^N56=@P8OWVQaXIRBA._7^Y /N56=@P8OWVQaXIRBA._7^Y/J UN56=@P8OWVQaXIRBA._7^` bN56=@P8OWVQaXIRBA._: JN56=@P8OWVQaXIRBA._:JZ EN56=@P8OWVQaXIRBA._:JZEM LN56=@P8OWVQaXIRBA/ _N56=@P8OWVQaXIRBA/_7 ^N56=@P8OWVQaXIRBA/_7^Y 'N56=@P8OWVQaXIRBA/_7^Y'J UN56=@P8OWVQaXIRBA/_7^` bN56=@P8OWVQaXIRBA/_7^`bU 'N56=@P8OWV"
"QaXIRBA/_7^`bU': JN56=@P8OWVQaXIRBA/_7^`bU':JZ EN56=@P8OWVQaXIRBA/_7^`bU':JZE0 <N56=@P8OWVQaXIRBA/_7^`bU':JZE0<9 ]N56=@P8OWVQaXIRBA: JN56=@P8OWVQaXIRBA:JZ EN56=@P8OWVQaXIRBA:JZE< `N56=@P8OWVQaXIRBA:JZE<`_ DN56=@P8OWVQaXIRBA:JZE<`_D7 ^N56=@P8OWVQaXIRBA:JZE<`_D7^] MN56=@P8OWVQaXIRBA:JZE<`_D7^]M0 .N56="
"@P8OWVQaXIRBA:JZE<`_D7^]MC .N56=@P8OWVQaXIRBA:JZE<`_D7^]MC.L UN56=@P8OWVQaXIRBA:JZE<`_D7^]MC.LU4 9N56=@P8OWVQaXIRBA:JZE<`_D7^]MK CN56=@P8OWVQaXIRBA:JZE<`_D9 7N56=@P8OWVQaXIRBA:JZE<`_D97M ;N56=@P8OWVQaXIRBA:JZE<`_D97M;3 -N56=@P8OWVQaXIRBA:JZE<`_DM LN56=@P8OWVQaXIRBA:JZE<`_DMLb 7N56=@P8OWVQaXIRBA:JZEM"
" `N56=@P8OWVQaXIRBA:JZEM`_ <N56=@P8OWVQaXIRBA:JZEM`_<. /N56=@P8OWVQaXIRBA:JZEM`_<7 DN56=@P8OWVQaXIRBA:JZEM`_<7D0 ^N56=@P8OWVQaXIRBA:JZEM`_<7D0^] UN56=@P8OWVQaXIRBA:JZEM`_<7D0^]U9 .N56=@P8OWVQaXIRBA:JZEM`_<9 DN56=@P8OWVQaXIRBA:JZEM`_<9D0 4N56=@P8OWVQaXIRBA:JZEM`_<9D047 ^N56=@P8OWVQaXIRBA:JZEM`_<9D047"
"^] UN56=@P8OWVQaXIRb EN56=@P8OWVQaXR EN56=@P8OWVQaXRE< `N56=@P8OWVQaXRE<`_ MN56=@P8OWVQaXRE<`_MD LN56=@P8OWVQaXREM _N56=@P8OWVQaXREM_. `N56=@P8OWVQaXREM_.`^ IN56=@P8OWVQaXREM_.`^Ib LN56=@P8OWVQaXREM_.`^IbLU DN56=@P8OWVQaXREM_.`^IbLUD< 7N56=@P8OWVQaXREM_7 LN56=@P8OWVQaXREM_7LA <N56=@P8OWVQaXREM_7LA<D"
" :N56=@P8OWVQaXREM_7LA<D:B 9N56=@P8OWVQaXREM_7LA<` IN56=@P8OWVQaXREM_I AN56=@P8OWVQaXb ^N56=@P8OWVQaXb^_ EN56=@P8OWVQaXb^_E< MN56=@P8OWVQaXb^_E<MD CN56=@P8OWVQaXb^_E<MDC; 7N56=@P8OWVQaXb^_E<MDC;7K UN56=@P8OWVQaXb^_E<MDC;7KU. 3N56=@P8OWVQaXb^_E<MDC;7KU.3+ LN56=@P8OWVQaXb^_E<MDC;7KU.3+L/ 0N56=@P8OWVQa"
"Xb^_E<MDC;7KU.3+L/09 &N56=@P8OWVQaXb^_EM UN56=@P8OWVQaXb^_EMU. 4N56=@P8OWX EN56=@P8OWXE. 7N56=@P8OWXE.7/ 0N56=@P8OWXE/ 7N56=@P8OWXE7 /N56=@P8OWXE7/( AN56=@P8OWXE7/(A9 QN56=@P8OWXE7/(A9QB 0N56=@P8OWXE7/(A9QB0' `N56=@P8OWXE7/9 0N56=@P8OWXE7/90A .N56=@P8OWXE7/90A.$ &N56=@P8OWXE7/90A.$&- <N56=@P8OWXE7/9"
"0A.$&-<_ #N56=@P8OWXE7/90A.$&-<_#U MN56=@P8OWXE7/90A.$&-<_#UMV ^N56=@P8OWXE7/90A.- QN56=@P8OWXE7/90A.-Q$ &N56=@P8OWXE7/90A._ IN56=@P8OWXE7/90A._I- QN56=@P8OWXE7/90A._I-QJ BN56=@P8OWXE7/90A._I-QJB( RN56=@P8OWXE7/90A._I-QJB(RL DN56=@P8OWXE7/90A._I-QJB(RLDM VN56=@P8OWXE7/90A._I-QJB(RLDMV^ KN56=@P8OWXE7"
"/90A._I-QJB(RLDMV^K' 1N56=@P8OWXE7/90A._I-QJB: RN56=@P8OWXE7/90A._I-QJB:RL 1N56=@P8OWXE7/90A._I-QJB:RL1V 2N56=@P8OWXE7/90A._I-QJB:RL1V2* )N56=@P8OWXE7/90A._I-QJB:RL1V2*)' ^N56=@P8OWXE7/90A._I-QJB:RL1V2*)'^Z MN56=@P8OWXE7/90A._IB RN56=@P8OWXE7/90A._IBRQ DN56=@P8OWXE7/90A._IBRQD( aN56=@P8OWXE7/90A._IB"
"RQD(a< bN56=@P8OWXE7/90A._IBRQD(a<bU MN56=@P8OWXE7/90A._IBRQD(a<bUMV -N56=@P8OWXE7/90A._IBRQD(a<bUMV-` ^N56=@P8OWXE7/90A._IBRQD(a<bUMV-`^; KN56=@P8OWXE7/90A._IBRQDV :N56=@P8OWXE7/90A._IBRQDV:$ &N56=@P8OWXE7/90A._IBRQDV:- JN56=@P8OWXE7/90A._IBRQDV:-J& ZN56=@P8OWXE7/90A._IBRQDV:-J&Z' (N56=@P8OWXE7/90A"
"._IBRQDV:-J&Z'() 4N56=@P8OWXE7/90A._IBRV QN56=@P8OWXE7/90A._IBRVQJ :N56=@P8OWXE7/90A._IBRVQJ:- DN56=@P8OWXE7/90A._IBRVQJ:-D( 'N56=@P8OWXE7/90A._IBRVQM 2N56=@P8OWXE7/90A._IBRVQM2J :N56=@P8OWXE7/90A._IBRVQM2J:- DN56=@P8OWXE7/90A._IBRVQM2J:-D4 <N56=@P8OWXE7/90A._IQ DN56=@P8OWXE7/90A._IQD( BN56=@P8OWXE7"
"/90A._IQD(BJ RN56=@P8OWXE7/90A._IQD(BJRU :N56=@P8OWXE7/90A._IQD(BJRV :N56=@P8OWXE7/90A._IQD(BJRV:L -N56=@P8OWXE7/90A._IQDV :N56=@P8OWXE7/90A._IU MN56=@P8OWXE7/90A._IV JN56=@P8OWXE7/90A._IVJ: QN56=@P8OWXE7/90A._IVJ:QB RN56=@P8OWXE7/90A._IVJ:QBRZ DN56=@P8OWXE7/90A._IVJU BN56=@P8OWXE7/90A._IVJUB- DN56="
"@P8OWXE7/90A._IVJUB-D4 MN56=@P8OWXE7/90A._IVJUB-D4M< $N56=@P8OWXE7/90A._IVJUB-D4M<$R CN56=@P8OWXE7/90A._IVJUB-DR ZN56=@P8OWXE7/A .N56=@P8OWXE7/A.- 9N56=@P8OWXE7/A.0 QN56=@P8OWXE7/A.0Q$ &N56=@P8OWXE7/A.0Q$&' <N56=@P8OWXE7/A.0Q& IN56=@P8OWXE7/A.0Q&I` VN56=@P8OWXE7/A.0Q&I`V' $N56=@P8OWXE7/A.0Q' IN56=@P"
"8OWXE7/A.0Q'IR aN56=@P8OWXE7/A.0QU IN56=@P8OWXE7/A.0QUI9 aN56=@P8OWXE7/A.0QUI9aV MN56=@P8OWXE7/A.0QUIa BN56=@P8OWXE7/A.0QUIaB9 :N56=@P8OWXE7/A._ 9N56=@P8OWXE7/A._9V DN56=@P8OWXE7/A._9VD0 BN56=@P8OWXE7/A._9VD0BL MN56=@P8OWXE7/L AN56=@P8OWXE7/LAM VN56=@P8OWXE7/LAMVI RN56=@P8OWXE7/LAMVIR^ _N56=@P8OWXE7"
"/LAQ IN56=@P8OWXE7/_ AN56=@P8OWXE7/_AQ IN56=@P8OWXEA 7N56=@P8OWXEA7V 9N56=@P8OWXED MN56=@P8OWXEDM. IN56=@P8OWXEDM.I7 VN56=@P8OWXEDM.I7VA /N56=@P8OWXEDM.I` VN56=@P8OWXEDM.I`V7 aN56=@P8OWXEDM.I`V7ab /N56=@P8OWXEDM.I`V7ab/_ UN56=@P8OWXEL QN56=@P8OWXELQ. aN56=@P8OWXELQ.aA 7N56=@P8OWXELQ.aI MN56=@P8OWXEL"
"Q.aIMV UN56=@P8OWXELQ.aIMVU7 BN56=@P8OWXELQ.aIM` _N56=@P8OWXELQ.aV ^N56=@P8OWXELQ/ aN56=@P8OWXELQ/aI MN56=@P8OWXELQ/aIMV UN56=@P8OWXELQ7 /N56=@P8OWXELQ7/( .N56=@P8OWXELQ7/(.' $N56=@P8OWXELQ7/(.'$- DN56=@P8OWXELQ7/(.'$-D9 `N56=@P8OWXELQ7/(.'$-D9`K <N56=@P8OWXELQ7/(.'$-D9`K<C 3N56=@P8OWXELQ7/(.'$-D9`K"
"<C34 0N56=@P8OWXELQ7/(.'$-D9`K<C340M TN56=@P8OWXELQ7/(.'$-D9`K<C340MTI AN56=@P8OWXELQ7/(.'$-DK <N56=@P8OWXELQ7/(.'$-DK<C 3N56=@P8OWXELQ7/(.'$-DK<C3M 4N56=@P8OWXELQ7/(.'$9 0N56=@P8OWXELQ7/(.'$90A IN56=@P8OWXELQ7/(.'$90AIM -N56=@P8OWXELQ7/(.'$90AIM-U VN56=@P8OWXELQ7/(.'$90AIM-UVa JN56=@P8OWXELQ7/(.'$9"
"0AIM-UVaJ_ BN56=@P8OWXELQ7/(.'$90AIM-UVaJ` BN56=@P8OWXELQ7/(.'$A IN56=@P8OWXELQ7/(.- DN56=@P8OWXELQ7/(.-DK <N56=@P8OWXELQ7/(.-DK<C 3N56=@P8OWXELQ7/(.-DK<C39 `N56=@P8OWXELQ7/9 AN56=@P8OWXELQ7/9AB .N56=@P8OWXELQ7/9AB.M JN56=@P8OWXELQ7/9AB.MJ& <N56=@P8OWXELQ7/9AB.MJ&<' (N56=@P8OWXELQ7/9AB.MJ&<'(4 0N56="
"@P8OWXELQ7/9AB.MJ&<'(40) DN56=@P8OWXELQ7/9AB.MJ&<'(40)D` VN56=@P8OWXELQ7/9AB.MJ&<'(40)D`V- :N56=@P8OWXELQ7/A IN56=@P8OWXELQ7/AIM JN56=@P8OWXELQ7/AIMJ' DN56=@P8OWXELQ7/AIMJ( DN56=@P8OWXELQ7/AIMJ(D` VN56=@P8OWXELQ7/AIMJ9 0N56=@P8OWXELQ7/AIMJ90B :N56=@P8OWXELQ7/AIMJ90B:' `N56=@P8OWXELQ7/AIMJ90B:'`U DN5"
"6=@P8OWXELQ7/AIMJB 9N56=@P8OWXELQ7/AIMJB9_ :N56=@P8OWXELQ7/AIMJB9_:. <N56=@P8OWXELQ7/AIMJY .N56=@P8OWXELQ7/AIMJY.- cN56=@P8OWXELQ7/AIMJY.-c& bN56=@P8OWXELQ7/AIMJY.-c&bV ^N56=@P8OWXELQ7/AIMJY.-c&bV^_ $N56=@P8OWXELQ7/AIMJY.-c&bV^_$# `N56=@P8OWXELQ7/AIMJ` VN56=@P8OWXELQ7/AIMJ`V_ -N56=@P8OWXELQ7/AIMJa -"
"N56=@P8OWXELQ7/AIMJa-$ _N56=@P8OWXELQ7/AIMJa-$_V <N56=@P8OWXELQ7/AIMJa-$_V<3 4N56=@P8OWXELQ7/AIMJa-$_V<34. DN56=@P8OWXELQ7/AIMJa-$_V<34.D( UN56=@P8OWXELQ7/AIMJa-$_V<34.D(UK ,N56=@P8OWXELQ7/I MN56=@P8OWXELQ7/IMA .N56=@P8OWXELQ7/IM_ RN56=@P8OWXELQ7/IM_RA :N56=@P8OWXELQ7/IM_RA:V KN56=@P8OWXELQ7/IM_RA:V"
"K. aN56=@P8OWXELQ7/IM_RV KN56=@P8OWXELQ7/IM_RVKA 9N56=@P8OWXELQ7/IM_RVKA9. aN56=@P8OWXELQ7/IM_RVKA9.aD `N56=@P8OWXELQ7/IM_RVKA9.aD`( &N56=@P8OWXELQ7/IM_RVKA9.aD`(&J BN56=@P8OWXELQ7/IM_RVKA9.aD`(&JB' )N56=@P8OWXELQ7/IM` AN56=@P8OWXELQ7/IM`AV aN56=@P8OWXELQ7/IM`AVaU _N56=@P8OWXELQ7/IM`AVab ^N56=@P8OWX"
"ELQ7/IM`AVab^R UN56=@P8OWXELQ7/IM`AVab^RUJ KN56=@P8OWXELQ7/J .N56=@P8OWXELQ7/a _N56=@P8OWXELQ7/a_A IN56=@P8OWXELQ7/a_AI` UN56=@P8OWXELQ7/a_AI`U^ VN56=@P8OWXELQ7/a_AI`U^VM JN56=@P8OWXELQ7/a_AI`U^VMJR -N56=@P8OWXELQ7/a_AI`U^VMJR-$ <N56=@P8OWXELQ7/a_AI`U^VMJR-$<B DN56=@P8OWXELQ7/a_AI`U^VMJR-$<BD: 4N56="
"@P8OWXELQI MN56=@P8OWXELQIM_ DN56=@P8OWXELQIM_DA 7N56=@P8OWXELQIM_DA7V KN56=@P8OWXELQIM_DA7VK0 aN56=@P8OWXELQIM_DA7VK0aT `N56=@P8OWXELQIM_DA7VK0aT`/ [N56=@P8OWXELQIM_DA7VKC -N56=@P8OWXELQIM` DN56=@P8OWXELQIM`D4 7N56=@P8OWXELQIM`D47A RN56=@P8OWXELQIM`D7 .N56=@P8OWXELQIM`D7.4 JN56=@P8OWXELQIM`D7.4JA 0"
"N56=@P8OWXELQIM`DA 7N56=@P8OWXELQIM`DA74 <N56=@P8OWXELQIM`DV 7N56=@P8OWXELQIM`DV7; 4N56=@P8OWXELQIM`DV7J UN56=@P8OWXELQIM`DV7JU_ ^N56=@P8OWXELQIM`DV7JU_^] 4N56=@P8OWXELQIM`DV7JU_^]4; bN56=@P8OWXELQJ RN56=@P8OWXELQJRZ DN56=@P8OWXELQJRZD7 0N56=@P8OWXELQJRZD709 /N56=@P8OWXELQa _N56=@P8OWXELQa_7 /N56=@P"
"8OWXELQa_7/A IN56=@P8OWXELQa_7/AIM :N56=@P8OWXELQa_7/I AN56=@P8OWXELQa_7/IAR MN56=@P8OWXELQa_I UN56=@P8OWXELQa_R IN56=@P8OWXELQa_V MN56=@P8OWXELQa_VMD <N56=@P8OWXELQa_VMD<7 UN56=@P8OWXELQa_VMD<7U3 ^N56=@P8OWXELQa_VMD<7UC ^N56=@P8OWXELQa_VMD<7U] /N56=@P8OWXELQa_VMD<7U]/. 0N56=@P8OWXELQa_VMD<7U]/.0' 4"
"N56=@P8OWXELQa_VMD<7U]/.0'43 ;N56=@P8OWXELQa_VMD<7U]/.0'43;K CN56=@P8OWXELQa_VMD<7U]/.0'43;KC- (N56=@P8OWXELQa_VMD<7U]/.03 ;N56=@P8OWXELQa_VMD<7U]/.03;K CN56=@P8OWXELQa_VMD<7U]/.03;KC' 4N56=@P8OWXELQa_VMD<7U]/.03;KC'4- (N56=@P8OWXELQa_VMD<I UN56=@P8OWXEM VN56=@P8OWXE_ 7N56=@P8OWXE_7A 9N56=@P8OWXE` 7"
"N56=@P8OWXE`7- .N56=@P8OWXE`70 AN56=@P8OWXE`70A/ IN56=@P8OWXE`7A 9N56=@P8U ON56=@P8UOE VN56=@V 7N56=@V7- EN56=@V7-EO .N56=@V7-EO.M 4N56=@V70 /N56E @N56E@- 7N56E@. MN56E@7 MN56E@7M8 =N56E@7MA =N56E@7MO XN56E@7MOXV 8N56E@7MP ON56E@8 AN56E@8A7 ON56E@8AP MN56E@8APM- ON56E@8APM. IN56E@8APM7 =N56E@8APM7=9"
" .N56E@8APM7=I .N56E@8APM7=I.- /N56E@8APM7=I.-/< DN56E@8APM9 DN56E@8APMV UN56E@8APMVUW DN56E@P =N56E@P=8 ON56E@V 4N56E@V47 8N56E@W MN56M @N56M@7 EN56M@7E8 AN56M@7E8AP =N56M@7E8AP=O IN56M@7E= DN56M@7E=DL <N56M@7E=DV <N56M@7E=DV<4 0N56M@7EP ON56M@8 EN56M@8EP 7N56M@O IN56M@OIP =N56M@OIP=E <N56M@P ON56M"
"@PO7 EN56M@PO7E= AN56M@PO8 7N56M@V 4N56U 8N= 6N=6- @N=6/ .N=65 @N=65@. 7N=65@.70 /N=65@/ PN=65@/PM EN=65@7 8N=65@789 /N=65@789/0 AN=65@78A PN=65@78AP/ .N=65@78AP/.' $N=65@78AP9 BN=65@78AP9BM EN=65@78AP9BMEL DN=65@78AP9BMELDO UN=65@78AP9BMELDOUJ :N=65@78AP9BMELDOUJ:2 <N=65@78AP9BMELDOUJ:2<I WN=65@78A"
"P9BMELDOUJ:2<IW; 3N=65@78AP9BMELDOUJ:2<IW;3K -N=65@78AP9BMELDOUJ:2<IW;3K-V XN=65@78AP9BMELDOUJ:2<IW;3K-VXQ _N=65@78AP9BMELDOUJ:2<IW;3K-VXQ_4 /N=65@78AP9BO /N=65@78AP9BO/: 2N=65@78AP9BO/:2I JN=65@78AP9BO/:2IJQ VN=65@78API :N=65@78API:B 9N=65@78API:B92 RN=65@78API:B92RQ /N=65@78APO MN=65@78I AN=65@78I"
"AP /N=65@78IAP/. ON=65@78IAP/.OQ 9N=65@78O PN=65@78OPA :N=65@78OPA:9 0N=65@78OPA:I 9N=65@78OPA:I9B MN=65@78OPA:I9BM/ 0N=65@78OPA:I9BM/0Q EN=65@78OPU /N=65@78OPU/( &N=65@78OPU/(&0 .N=65@78P /N=65@78V /N=65@8 7N=65@870 /N=65@87E ON=65@87P ON=65@87PO0 /N=65@A 7N=65@A7E ON=65@E 4N=65@E4- <N=65@E4-<7 ON="
"65@E4-<7O8 .N=65@E4-<7OD AN=65@E4-<7OL MN=65@E4-<7OLM3 8N=65@E4-<7OLM8 .N=65@E4-<7OLM8.$ PN=65@E4-<7OLMC DN=65@E4-<7OLMD ;N=65@E4-<7OLMD;8 KN=65@E4-<7OLMD;8K3 .N=65@E4-<7OLMD;8K3.$ PN=65@E4-<7OLMD;8K3.$P/ IN=65@E4-<7OLMD;8K3.$P/IC +N=65@E4-<7OLMD;8K3.$P/IR BN=65@E4-<7OLMD;8K3.$P/IRBC AN=65@E4-<7OLMD"
";8KX PN=65@E4-<7OLMD;8KXP3 IN=65@E4-<7OLMD;8KXP3IC 9N=65@E4-<7OLMD;8KXP3IC9S 0N=65@E4-<7OLMD;8KXP3IC9S0: BN=65@E4-<7OLMD;8KXP3IC9S0:BA 2N=65@E4-<7OLMD;8KXP3IC9S0U .N=65@E4-<7OLMD;8KXP3IC9S0U.W `N=65@E4-<7OLMD;X PN=65@E4-<7OLMD;XP9 .N=65@E4-<7OLMD;XP9.I QN=65@E4-<7OLMD;XP9.IQC JN=65@E4-<7OLMD;XP9.IQC"
"JV UN=65@E4-<7OLMD;XP9.IQCJVUW AN=65@E4-<7OLMD;XP9.IQCJVUWA8 0N=65@E4-<7OLMD;XPV WN=65@E4-<7OM PN=65@E4-<7OMPA .N=65@E4-<7OMPC ;N=65@E4-<7OMPC;3 LN=65@E4-<7OMPC;3LA 9N=65@E4-<7OMPC;3LA9I 8N=65@E4-<7OMPC;3LD 8N=65@E4-<7OMPC;3LD8U KN=65@E4-<7OMPC;3LD8UKS ^N=65@E4-<7OMPC;D KN=65@E4-<7OMPC;DK3 LN=65@E4-"
"<7OMPC;DK3LS 8N=65@E4-<7OMPC;DK3LS8U .N=65@E4-<7OMPC;DKV UN=65@E4-<7OMPC;DKVUW `N=65@E4-<7OMPC;DKVUW`L _N=65@E4-<7OMPC;DKVUW`L_+ SN=65@E4-<7OMPC;DKVUW`L_+S9 8N=65@E4-<7OMPC;DKVUW`L_+S98A IN=65@E4-<7OMPC;DKVUW`L_+S98AIX JN=65@E4-<7OMPC;DKVUW`L_+S98I /N=65@E4-<7OMPC;DKVUW`L_+S98I/X .N=65@E4-<7OMPC;DKV"
"UW`L_+S98I/X.& TN=65@E4-<7OMPC;DKVUW`L_+SI 8N=65@E4-<7OMPC;DKVUW`L_+SI8A 9N=65@E4-<7OMPC;DKVUW`L_9 XN=65@E4-<7OMPC;DKVUW`L_9XI &N=65@E4-<7OMPC;DKVUW`L_9XI&$ #N=65@E4-<7OMPC;DKVUW`L_9XI&$#Q .N=65@E4-<7OMPC;DKVUW`L_9XI&$#Q.+ SN=65@E4-<7OMPC;DKVUW`L_9XI&$#Q.+S/ 0N=65@E4-<7OMPC;DKVUW`L_9XI&$#Q.+Sb 0N=65"
"@E4-<7OMPC;DKVUW`L_9XI&$#Q.+Sb0] TN=65@E4-<7OMPC;DKVUW`L_9XI&$#Q./ 0N=65@E4-<7OMPC;DKVUW`L_9XI&$#Q./0+ SN=65@E4-<7OMPC;DKVUW`L_9XI&$#Q./0+S8 'N=65@E4-<7OMPC;DKVUW`L_9XI&$#Q./0+S8'a YN=65@E4-<7OMPC;DKVUW`L_9XI&Q .N=65@E4-<7OMPC;DKVUW`L_9XI&Q.+ 0N=65@E4-<7OMPC;DKVUW`L_9XI&Q.+0' SN=65@E4-<7OMPC;DKVUW`_"
" ^N=65@E4-<7OMPC;DKVUW`_^L aN=65@E4-<7OMPC;DKVUW`_^La9 SN=65@E4-<7OMPC;DKVUW`_^La9SI JN=65@E4-<7OMPC;DKVUW`_^X LN=65@E4-<7OMPC;DKVUW`_^XL9 $N=65@E4-<7OMPC;DKVUW`_^XL9$A IN=65@E4-<7OMPC;DKVUX LN=65@E4-<7OMPC;DKVUXL^ WN=65@E4-<7OMPC;DKVUXL^Wa &N=65@E4-<7OMPC;V KN=65@E4-<7OMPC;VKD UN=65@E4-<7OMPC;VKDUW"
" `N=65@E4-<7OMPC;VKDUW`L aN=65@E4-<7OMPV DN=65@E4-<7OMPVD9 UN=65@E4-<7OMPVD9UW `N=65@E4-<7OMPVDL WN=65@E4-<7OMPVDLWC ^N=65@E4-<7OMPVDLWC^K 3N=65@E4-<7OMPVDLWC^K3; SN=65@E4-<7OMPVDLWC^K3;S_ +N=65@E4-<7OMPVDW UN=65@E4-<7OMPVDWUC ;N=65@E4-<7OMPVDWUC;L KN=65@E4-<7OMPVDWUC;LKQ aN=65@E4-<7OMPVDWUL aN=65@E"
"4-<7OMPVDWULaI ^N=65@E4-<7OMPVDWULaI^3 CN=65@E4-<7OMPVDWULaK XN=65@E4-<7OMPVDWULaKX_ CN=65@E4-<7OMPVDWULaKX_C` ^N=65@E4-<7OMPVDWULaKX_C`^9 .N=65@E4-<7OMPVDWULaKX` ^N=65@E4-<7OMPVDWULaKX`^_ CN=65@E4-<7OMPVDWULa_ `N=65@E4-<7OMPVDWULa_`^ XN=65@E4-<7OMPVDX UN=65@E4-<7OMPVDXUL CN=65@E4-<7OMPVDXULC^ aN=65"
"@E4-<D .N=65@E4-<D.7 8N=65@E4-<M .N=65@E4-<M.& 7N=65@E4-<M.3 ;N=65@E4-<M.3;C ON=65@E4-<M.7 ON=65@E4-<M.7O$ LN=65@E4-<M.7O$LV 8N=65@E4-<M.7O& DN=65@E4-<M.7O&DL VN=65@E4-<M.O DN=65@E4-<M.OD7 8N=65@E4-<M.OD78A $N=65@E4-<M.OD78A$C KN=65@E4-<M.OD78A$CKS LN=65@E4-<M.OD78A$CKSLU VN=65@E4-<M.OD78A$CKSLUV; X"
"N=65@E4-<M.OD78A$CKSLUV;XP WN=65@E4-<M.OD78C QN=65@E4-<M.OD78CQI RN=65@E4-<M.OD78CQIRA JN=65@E4-<M.OD78CQIRAJ$ KN=65@E4-<M.OD78CQIRAJ$KS PN=65@E4-<M.OD78CQIRAJ$KSP9 /N=65@E4-<M.OD78CQIRAJ$KSP9/0 XN=65@E4-<M.OD78CQIRAJ$KSP9/0XW LN=65@E4-<M.OD78CQIRAJ; $N=65@E4-<M.OD78CQIRAJ;$9 UN=65@E4-<M.OD78CQIRAJ;"
"$9UB XN=65@E4-<M.OD78CQIRAJ;$9UBXP WN=65@E4-<M.ODC KN=65@E4-<M.ODCK3 LN=65@E4-<M.ODCK3L9 UN=65@E4-<M.ODCK3L; +N=65@E4-<M.ODCK3L;+7 8N=65@E4-<M.ODCK3L;+78A $N=65@E4-<M.ODCK3L;+78A$0 9N=65@E4-<M.ODCK3L;+78A$09& /N=65@E4-<M.ODCK3L;+78A$09B :N=65@E4-<M.ODCK3L;+78A$09B:2 (N=65@E4-<M.ODCK3L;+78A$09B:2(/ U"
"N=65@E4-<M.ODCK3L;+78A$09B:2(/UV `N=65@E4-<M.ODCK3L;+78A$09B:2(1 IN=65@E4-<M.ODCK3L;+78A$09B:2(1IP VN=65@E4-<M.ODCK3L;+78A$09B:2(1IPVU ^N=65@E4-<M.ODCK3L;+78A$09B:2(1IPV^ `N=65@E4-<M.ODCK3L;+9 AN=65@E4-<M.ODCK3L;+9A' $N=65@E4-<M.ODCK3L;+9A'$/ &N=65@E4-<M.ODCK3L;+9A'$/&7 IN=65@E4-<M.ODCK3L;+9A7 8N=65"
"@E4-<M.ODCK3L;+9A78: VN=65@E4-<M.ODCK3L;+9A78:V' XN=65@E4-<M.ODCK3L;+9A78:VU _N=65@E4-<M.ODCK3L;+9A78:VU_` XN=65@E4-<M.ODCK3L;+9A8 IN=65@E4-<M.ODCK3L;+9A: IN=65@E4-<M.ODCK3LU IN=65@E4-<M.ODCK3LUI' ;N=65@E4-<M.ODCK3LUI';S ,N=65@E4-<M.ODCKS LN=65@E4-<M.ODCKSL8 UN=65@E4-<M.ODCKSLU IN=65@E4-<M.ODCKSLUI+"
" $N=65@E4-<M.ODCKSLUI+$' /N=65@E4-<M.ODCKSLUI+$'/( ^N=65@E4-<M.ODCKSLUI+$'/(^8 7N=65@E4-<M.ODCKSLUI+$'/(^870 WN=65@E4-<M.ODCKSLUI+$/ 7N=65@E4-<M.ODCKSLUI+$7 /N=65@E4-<M.ODCKSLUI+$7/( ^N=65@E4-<M.ODCKSLUI+$7/(^V 8N=65@E4-<M.ODCKSLUI+$7/(^V8' 0N=65@E4-<M.ODCKSLUI+$7/(^V8'0Q PN=65@E4-<M.ODCKSLUI+$7/(^V"
"8'0QPJ XN=65@E4-<M.ODCKSLUI; +N=65@E4-<M.ODCKSLUI;+' $N=65@E4-<M.ODCKSLUI;+'$8 /N=65@E4-<M.ODCKSLUI;+'$8/P 9N=65@E4-<M.ODCKSLUI;+'$8/P9A BN=65@E4-<M.ODCKSLUIA PN=65@E4-<M.ODCKSLUIAP7 8N=65@E4-<M.ODCKSLUIAP789 BN=65@E4-<M.ODCKSLUIAP789B0 /N=65@E4-<M.ODCKSLUIAP789B0/& (N=65@E4-<M.ODCKSLUIAP789B0/&(' $"
"N=65@E4-<M.ODCKSLUIAP789B0/&('$1 *N=65@E4-<M.ODCKSLUIAP789B0/&('$1*) 2N=65@E4-<M.ODCKSLUIAP789B0/&('$1*)2; WN=65@E4. MN=65@E4.M8 ON=65@E4.M< DN=65@E4.M<D3 -N=65@E4.M<D3-; CN=65@E4.M<D3-;CK LN=65@E4.M<DL -N=65@E4.MD 7N=65@E4.MD7< KN=65@E4.ML <N=65@E4.ML<D ;N=65@E4.MU WN=65@E4.MV WN=65@E4.MVWL ON=65@E"
"4/ 7N=65@E4/7- 'N=65@E4/7-'( .N=65@E4/7-'(.& 8N=65@E4/7-'(.&80 #N=65@E4/7-'(.&8A LN=65@E4/7-'0 8N=65@E4/7. <N=65@E4/7.<3 -N=65@E4/7.<3-C MN=65@E4/7.<8 0N=65@E4/7.<803 &N=65@E4/7.<803&$ 'N=65@E4/7.<803&$'( ON=65@E4/7.<803&$'(O- PN=65@E4/7.<803&$'(O-P; IN=65@E4/7.<803&$'(O-P;IR BN=65@E4/78 DN=65@E4/78"
"D0 MN=65@E48 DN=65@E48D< MN=65@E48D<MV 7N=65@E48D<MV7/ WN=65@E4W DN=65@E4WDC MN=65@E4WDCML ON=65@M 7N=65@M7V PN=65@O 4N=65@O4- .N=65@O4-.7 <N=65@O4. 7N=65@O4.7- EN=65@O4.70 8N=65@O4/ EN=65@O4/E< 7N=65@O4/EM 7N=65@O4/EM7. PN=65@O4/EM7.P8 0N=65@O4/EM7.P80D <N=65@O4/EM7.P80D<- XN=65@O4/EM7.P80D<-XW KN="
"65@O4/EM7.P80D<-XWK) $N=65@O4/EM7.P80D<-XWK)$& (N=65@O4/EM7.P80D<-XWK)$&(Q LN=65@O4/EM7.P80I AN=65@O4/EM7.P80IAQ JN=65@O4/EM7.P80IAQJD $N=65@O4/EM7.PD -N=65@O4/EM7.PD-< 'N=65@O4/EM7.PD-<'$ WN=65@O4/EM7.PD-<'$W3 &N=65@O4/EM7.PD-<'$W8 AN=65@O4/EM7.PD-<'$W8A0 (N=65@O4/EM7.PD-<'$Wa VN=65@O4/EM7.PD-<'$Wa"
"V8 0N=65@O4/EM7.PD-<'3 &N=65@O4/EU 7N=65@O4/EV PN=65@O48 EN=65@O48E. /N=65@O48EL <N=65@O4M DN=65@O4MD; KN=65@O4MD;K< EN=65@O4MD;K<EL VN=65@O4MD< CN=65@P 8N=65@P8. ON=65@P8.OE MN=65@P87 <N=65@P87<C ON=65@P87<D ON=65@P87<DOA /N=65@P87<DOA/. IN=65@P87<DOE MN=65@P87<DOW 0N=65@P87<DOW0/ IN=65@P87<DOW0/IE"
" -N=65@P87<DOW0/IQ 'N=65@P87<DOW0/IQ'M XN=65@P87<DOW0/IQ'MXB -N=65@P87<DOW0/IQ'MXB-E AN=65@P87<DOW0/IQ'MXB-EA. 4N=65@P87<DOW0/IQ'MXB-EA4 `N=65@P87<I ON=65@P87<IO9 /N=65@P87<IO9/. 0N=65@P87<IO9/.0- :N=65@P87<IO9/.0-:D 4N=65@P87<IOD /N=65@P87<IOD/C 9N=65@P87<IOD/C9E AN=65@P87<IOD/C9EAQ MN=65@P87<IOD/C"
"9EAQMB LN=65@P87<IOD/C9EAQMBLX WN=65@P8A 9N=65@P8A97 /N=65@P8A97/0 QN=65@P8A9E IN=65@P8A9EI7 ON=65@P8A9EI7OQ MN=65@P8A9O 4N=65@P8A9O4. EN=65@P8A9O4.EM 7N=65@P8A9O4.EM70 $N=65@P8A9O4.EM70$/ (N=65@P8A9O40 7N=65@P8A9O407/ EN=65@P8A9O407/EM WN=65@P8A9O4M BN=65@P8A9O4MB. 'N=65@P8A9O4MB/ 7N=65@P8E ON=65@P"
"8EOA 7N=65@P8EOM IN=65@P8O WN=65@P8OW. EN=65@P8OW.EL DN=65@P8OW7 AN=65@P8OW7AQ /N=65@P8OWM QN=65@P8OWMQ. VN=65@P8OWMQI XN=65@P8OWMQIX9 AN=65@P8OWMQIXA BN=65@P8OWMQIXR UN=65@P8OWMQIXRUV ^N=65@P8OWMQIXRUV^. LN=65@P8OWMQIXRUV^.La EN=65@P8OWMQIXRUV^.LaE7 BN=65@P8OWMQIXRUV^.LaE7B_ `N=65@P8OWMQIXRUV^.LaE7"
"B_`J ZN=65@P8OWMQIXRUV^.LaE7B_`JZ] AN=65@P8OWMQIXRUV^.LaE7B_`JZ]A9 4N=65@P8OWMQIXRUV^.LaE7B_`JZ]A943 -N=65@P8OWMQIXRUV^.LaE7B_`JZ]A943-: 2N=65@P8OWMQIX` EN=65@P8OWMQIX`E< DN=65@P8OWMQIX`ED UN=65@P8OWMQIX`EDUa LN=65@P8OWMQIX`EDUaLV 7N=65@P8OWMQIX`EL AN=65@P8OWMQIXa EN=65@P8OWMQR EN=65@P8OWMQ` EN=65@P"
"8OWMQ`EL DN=65@P8OWMQ`ELD7 VN=65@P8OWMQ`ELD7VC aN=65@P8OWMQ`ELD7VCab XN=65@P8OWMQa XN=65@P8OWMQaX. EN=65@P8OWMQaX.E7 <N=65@P8OWMQaX.E7<4 /N=65@P8OWMQaX.E7<4/( -N=65@P8OWMQaX.E7<4/(-I UN=65@P8OWMQaX.E7<4/C $N=65@P8OWMQaX.E7<C `N=65@P8OWMQaX.E7<C`_ VN=65@P8OWMQaX.E7<C`_VL UN=65@P8OWMQaX.E< _N=65@P8OWM"
"QaX.E<_7 UN=65@P8OWMQaX.E<_7UV ^N=65@P8OWMQaX.E<_7UV^A 9N=65@P8OWMQaX.E<_7UV^A9I BN=65@P8OWMQaX.E<_7UV^A9IB/ :N=65@P8OWMQaXI RN=65@P8OWMQaXIRB `N=65@P8OWMQaXIRB`_ JN=65@P8OWMQaXIRB`_JZ EN=65@P8OWMQaXIRB`_JZEA 7N=65@P8OWMQaXIRB`_JZEA7V <N=65@P8OWMQaXR EN=65@P8OWV QN=65@P8OWVQ. MN=65@P8OWVQ/ MN=65@P8O"
"WVQI XN=65@P8OWVQIX/ RN=65@P8OWVQIX/R7 ^N=65@P8OWVQIX/R7^9 0N=65@P8OWVQIX/R7^90A :N=65@P8OWVQIX/R7^90A:` _N=65@P8OWVQIX/R7^90A:`_] MN=65@P8OWVQIX/R7^90A:`_]MU <N=65@P8OWVQIX/R7^90A:`_]MU<E 'N=65@P8OWVQIX/R7^90A:`_]MU<E'C .N=65@P8OWVQIX/R7^90A:`_]MU<E'C.a 4N=65@P8OWVQIX/R7^90A:`_]MU<E'C.a4- &N=65@P8O"
"WVQIX/R7^90` &N=65@P8OWVQIX/R7^90`&( _N=65@P8OWVQIX/R7^90`&(_] UN=65@P8OWVQIX/R7^90`&(_]U' )N=65@P8OWVQIX/R7^90`&(_]U')A :N=65@P8OWVQIX/R7^90`&A _N=65@P8OWVQIX/R7^90`&_ .N=65@P8OWVQIX/R7^` EN=65@P8OWVQIX/R7^`E_ aN=65@P8OWVQIX/R7^`E_aM UN=65@P8OWVQIX/R` EN=65@P8OWVQIX/R`EM ^N=65@P8OWVQIX/R`EM^U .N=65"
"@P8OWVQIX/R`EM^U.7 LN=65@P8OWVQIX/R`EM^U.7LD aN=65@P8OWVQIX/R`EM^U.7LDab CN=65@P8OWVQIX/R`EM^U.7LDabC< &N=65@P8OWVQIX/R`EM^U.7LDabC<&( 0N=65@P8OWVQIX/Ra _N=65@P8OWVQIX/Ra_7 ^N=65@P8OWVQIX/Ra_7^Y UN=65@P8OWVQIX/Ra_7^YU` EN=65@P8OWVQIX/Ra_7^YU`EL DN=65@P8OWVQIX/Ra_7^YU`ELDK <N=65@P8OWVQIX/Ra_7^YU`ELDK"
"<] 0N=65@P8OWVQIX/Ra_7^YU`ELDK<]0T 'N=65@P8OWVQIX/Ra_7^YU`ELDK<]0T'C .N=65@P8OWVQIX/Ra_7^YU`ELDK<]0T'Z JN=65@P8OWVQIX/Ra_7^YU`ELDM CN=65@P8OWVQIX/Ra_7^YU`ELDMC9 4N=65@P8OWVQIX/Ra_7^YU`ELDMC94< 'N=65@P8OWVQIX/Ra_7^YU`E] <N=65@P8OWVQIX` ^N=65@P8OWVQIX`^U EN=65@P8OWVQIX`^UEL RN=65@P8OWVQIX`^UEM RN=65@P"
"8OWVQIX`^UEMR. <N=65@P8OWVQIXa RN=65@P8OWVQIXaR. _N=65@P8OWVQIXaR/ _N=65@P8OWVQIXaR/_7 MN=65@P8OWVQIXaR/_7MD LN=65@P8OWVQIXaR/_7MDLE ;N=65@P8OWVQIXaR/_7MDLE;U ^N=65@P8OWVQIXaR/_7MDLE;U^K `N=65@P8OWVQIXaR/_7MDLE;U^K`] &N=65@P8OWVQIXaRB EN=65@P8OWVQIXaRb ^N=65@P8OWVQIXaRb^_ EN=65@P8OWVQIXaRb^_EM UN=65"
"@P8OWVQX EN=65@P8OWVQXEM UN=65@P8OWVQXEMU^ LN=65@P8OWVQXEMU^LD 7N=65@P8OWVQXEMU^LD7. 4N=65@P8OWVQXEMU^LD7.4- CN=65@P8OWVQXEMU^LD7.4-C< ;N=65@P8OWVQXEMU^LD7.4-C<;R TN=65@P8OWVQXEMU^LD7< CN=65@P8OWVQXEMU^LD7<C. 3N=65@P8OWVQXEMU^LD7<C.3/ 0N=65@P8OWVQXEMU^LD7<C.3/0R `N=65@P8OWVQXEMU^LD7J `N=65@P8OWVQXEM"
"U^LD7J`_ ]N=65@P8OWVQXEMU^LD7R `N=65@P8OWVQXEMU^LD7R`_ ]N=65@P8OWVQXEMU^LD7R`_]< CN=65@P8OWVQXEMU^LD7R`_]<C. 3N=65@P8OWVQ_ EN=65@P8OWVQ_E< XN=65@P8OWVQ_E<X. MN=65@P8OWVQ_E<X.M7 ^N=65@P8OWVQ_EM UN=65@P8OWVQ_EMU. 4N=65@P8OWVQ_EMU.4< 7N=65@P8OWVQ_EMUI DN=65@P8OWVQ_EMU^ XN=65@P8OWVQ_EMU^X. aN=65@P8OWVQ_"
"EMU^X.a9 AN=65@P8OWVQ_EMU^X.a9A7 0N=65@P8OWVQ_EMU^X7 aN=65@P8OWVQ_EMU^X7a9 DN=65@P8OWVQ_EMU^XR `N=65@P8OWVQ_EMU^XR`a LN=65@P8OWVQ_EMU^Xb LN=65@P8OWVQ_EMU^XbL7 DN=65@P8OWVQ_EMU^XbL7D< 0N=65@P8OWVQ_EMU^XbLC DN=65@P8OWVQ_EMU^XbLCD< -N=65@P8OWVQ_EMU^XbLCD<-I ;N=65@P8OWVQ_EMU^XbLD 7N=65@P8OWVQ_EMU^XbLD7<"
" CN=65@P8OWVQ` EN=65@P8OWVQa XN=65@P8OWVQaX. _N=65@P8OWVQaX._7 /N=65@P8OWVQaX._7/( ^N=65@P8OWVQaX/ _N=65@P8OWVQaX/_7 ^N=65@P8OWVQaX/_7^9 MN=65@P8OWVQaX/_I AN=65@P8OWVQaX/_IA7 ^N=65@P8OWVQaXI RN=65@P8OWVQaXIR/ `N=65@P8OWVQaXIR/`_ AN=65@P8OWVQaXIRB AN=65@P8OWVQaXIRBA. _N=65@P8OWVQaXIRBA/ EN=65@P8OWVQa"
"XIRBA/EM .N=65@P8OWVQaXIRBA/EM.' (N=65@P8OWVQaXIRBA/EM.7 -N=65@P8OWVQaXIRBA: JN=65@P8OWVQaXIRBA:JZ EN=65@P8OWVQaXIRBA:JZE< `N=65@P8OWVQaXIRBA:JZEM `N=65@P8OWVQaXIRBA:JZEM`_ <N=65@P8OWVQaXIRBA:JZEM`_<7 /N=65@P8OWVQaXIRBA:JZEM`_<7/4 DN=65@P8OWVQaXIRBA:JZEM`_<7/4D0 -N=65@P8OWVQaXIRBA:JZEM`_<7/4D0-9 ^N="
"65@P8OWVQaXIRBA:JZEM`_<7/4D0-9^] UN=65@P8OWVQaXR ^N=65@P8OWVQaXR^_ EN=65@P8OWVQaXR^_EM `N=65@P8OWVQaXR^_EM`] IN=65@P8OWVQaXR^_EM`]Ib DN=65@P8OWVQaXb EN=65@P8OWVQaXbEM <N=65@P8OWVQaXbEM<4 DN=65@P8OWX EN=65@P8OWXE7 /N=65@P8OWXE7/9 0N=65@P8OWXE7/90A .N=65@P8OWXE7/90A.$ &N=65@P8OWXE7/90A.$&- <N=65@P8OWX"
"E7/90A.$&-<_ #N=65@P8OWXE7/90A.$&-<_#U MN=65@P8OWXE7/90A.$&-<_#UMV ^N=65@P8OWXE7/90A.- QN=65@P8OWXE7/90A._ IN=65@P8OWXE7/90A._I- JN=65@P8OWXE7/90A._IB RN=65@P8OWXE7/90A._IBRQ DN=65@P8OWXE7/90A._IBRQDV :N=65@P8OWXE7/90A._IBRV JN=65@P8OWXE7/90A._IQ DN=65@P8OWXE7/90A._IQD( BN=65@P8OWXE7/90A._IQD(BJ RN="
"65@P8OWXE7/90A._IQD(BJRV :N=65@P8OWXE7/90A._IQD(BJRV:L -N=65@P8OWXE7/90A._IQD(BJRV:L-M 'N=65@P8OWXE7/90A._IQD(BJRV:L-M'1 aN=65@P8OWXE7/90A._IQD(BJRV:L-M'1a& ^N=65@P8OWXE7/90A._IV JN=65@P8OWXE7/90A._IVJU BN=65@P8OWXE7/90A._IVJUB- DN=65@P8OWXE7/90A._IVJUB-D4 MN=65@P8OWXE7/90A._IVJUB-D4M< $N=65@P8OWXE7"
"/A .N=65@P8OWXE7/A.0 QN=65@P8OWXE7/A.0Q$ &N=65@P8OWXE7/A.0Q$&' <N=65@P8OWXE7/A.0QU 9N=65@P8OWXE7/A.0QU9I MN=65@P8OWXE7/A._ 9N=65@P8OWXE7/L AN=65@P8OWXE7/LAI MN=65@P8OWXED MN=65@P8OWXEDM. IN=65@P8OWXEL QN=65@P8OWXELQ. aN=65@P8OWXELQ.aA 7N=65@P8OWXELQ.aA7/ IN=65@P8OWXELQ.aA7/IM 0N=65@P8OWXELQ.aI MN=65"
"@P8OWXELQ7 /N=65@P8OWXELQ7/( .N=65@P8OWXELQ7/(.' DN=65@P8OWXELQ7/(.'D9 `N=65@P8OWXELQ7/(.'DM VN=65@P8OWXELQ7/(.J DN=65@P8OWXELQ7/9 AN=65@P8OWXELQ7/9AB .N=65@P8OWXELQ7/9AB.I 0N=65@P8OWXELQ7/9AB.I0J MN=65@P8OWXELQ7/A IN=65@P8OWXELQ7/AIM JN=65@P8OWXELQ7/AIMJ( DN=65@P8OWXELQ7/AIMJ(D` VN=65@P8OWXELQ7/AIM"
"J9 0N=65@P8OWXELQ7/AIMJ90B :N=65@P8OWXELQ7/AIMJa _N=65@P8OWXELQ7/I AN=65@P8OWXELQ7/IAU MN=65@P8OWXELQ7/IAUMV aN=65@P8OWXELQ7/IAUMVa0 .N=65@P8OWXELQ7/IAUM` .N=65@P8OWXELQ7/IAUM`.: BN=65@P8OWXELQ7/IAUM`.:B- VN=65@P8OWXELQ7/IAUM`.:B-V_ KN=65@P8OWXELQ7/IAUM`.:B-V_KD aN=65@P8OWXELQ7/IAUM`.R ^N=65@P8OWXEL"
"Q7/IAUM`.R^- VN=65@P8OWXELQI MN=65@P8OWXELQIM_ DN=65@P8OWXELQIM_DA 7N=65@P8OWXELQIM` DN=65@P8OWXELQIM`D4 7N=65@P8OWXELQIM`D47A RN=65@P8OWXELQIM`D47ARV UN=65@P8OWXELQIM`D7 .N=65@P8OWXELQIM`D7.4 <N=65@P8OWXELQIM`D7.4<V 3N=65@P8OWXELQIM`D7.4<V30 KN=65@P8OWXELQJ aN=65@P8OWXELQa VN=65@P8OWXE` 7N=65@P8OWX"
"E`7A 9N=6E @N=6E@. DN=6E@/ <N=6E@7 8N=6E@78- MN=6E@78-MA .N=6E@78-MA.O 9N=6E@78-MA.O90 /N=6E@78-MA.O90/5 'N=6E@78-MA.O90/5'P #N=6E@78-MA.O90/P JN=6E@78-MA.O90/PJ: IN=6E@78-MA.O90/PJ:I' &N=6E@78-MA.O90/PJ:I'&$ (N=6E@78-MA.O9U BN=6E@78-MA.O9V BN=6E@78-MA.O9VB& LN=6E@78-MO .N=6E@78-MO./ DN=6E@78-MO.A L"
"N=6E@78-MO.AL0 5N=6E@78-MO.AL05< DN=6E@78-MO.ALD 9N=6E@78-MO.ALD9& PN=6E@78-MO.ALD9&P/ :N=6E@78-MO.ALD9&PQ JN=6E@78-MO.ALD9&PU CN=6E@78-MO.ALD9&PUC; 3N=6E@78-MO.ALD9&PUCW 4N=6E@78-MO.ALD9&PUCW45 <N=6E@78-MO.ALD9&PUCW4V /N=6E@78-MO.ALD90 5N=6E@78-MO.ALD905< PN=6E@78-MO.ALD905<P$ /N=6E@78-MO.ALD905<P4"
" /N=6E@78-MO.ALD9U CN=6E@78-MO.ALD9UC/ VN=6E@78-MO.ALD9UC0 PN=6E@78-MO.ALD9UC0PW <N=6E@78-MO.ALD9UC; 3N=6E@78-MO.ALD9UC;30 5N=6E@78-MO.ALD9UC;305K SN=6E@78-MO.ALD9UC;305KS< VN=6E@78-MO.ALD9UC;305KS<VP WN=6E@78-MO.ALD9UCK <N=6E@78-MO.ALD9UCK<5 4N=6E@78-MO.ALD9UCK<54; XN=6E@78-MO.U 4N=6E@78-MO.U45 <N="
"6E@78-MO.U45<D KN=6E@78-MO.U45<DKC ;N=6E@78-MO.U45<DKC;3 +N=6E@78-MO.U45<DKV LN=6E@78-MO.U45<DKVL3 CN=6E@78-MO.U45<DKVL3C, XN=6E@78-MO.U45<DKVL3C,X; ^N=6E@78-MO.U45<DKVL3C,X;^_ `N=6E@78-MO.U45<DKVL3C,X;^_`A PN=6E@78-MO.U45<DKVL3C,X;^_`APW ]N=6E@78-MO.U4W PN=6E@78-MO.U4WP5 <N=6E@78-MO.U4WP5<3 XN=6E@7"
"8-MO.U4WPA IN=6E@78-MO.U4WPAI5 <N=6E@78-MO.U4WPAI5<0 DN=6E@78-MO.U4WPAI5<0DL _N=6E@78-MO.U4WPAI5<0DL_C /N=6E@78-MO.W VN=6E@78-MO.WVA LN=6E@78-MO.WVALD 9N=6E@78-MO.WVALD9^ _N=6E@78-MP ON=6E@78-MV PN=6E@78-MVPD 5N=6E@78-MVPD5A IN=6E@78-MVPD5AIO QN=6E@78-MVPU 5N=6E@78-MVPU5. DN=6E@78-MVPU5.D4 <N=6E@78-"
"MVPU5.DO <N=6E@78-MVPU5/ DN=6E@78. -N=6E@78.-/ 5N=6E@78.-/5O 'N=6E@78.-/5O'4 QN=6E@78.-/5O'4QI (N=6E@78.-/5O'4QI(0 DN=6E@78.-A 5N=6E@78.-A5O DN=6E@78.-P ON=6E@78/ MN=6E@78/MA LN=6E@78/MO PN=6E@78/MOPV UN=6E@78/MOPVUW _N=6E@78/MOPVUW_^ DN=6E@78A QN=6E@78AQI JN=6E@78AQIJ: RN=6E@78AQIJ:R9 BN=6E@78AQIJ:"
"R9B- 2N=6E@78AQIJ:R9BV 2N=6E@78AQIJ:R9BZ PN=6E@78AQIJ:R9BZPO XN=6E@78AQIJ:R9BZPV ON=6E@78AQIJ:R9BZPVO- UN=6E@78AQO 5N=6E@78AQO5. /N=6E@78AQO5./9 PN=6E@78AQO5./9PU <N=6E@78AQO5./P $N=6E@78AQO5M <N=6E@78AQO5M<4 3N=6E@78AQO5U <N=6E@78AQO5V 9N=6E@78AQO5V9. PN=6E@78AQO5W 9N=6E@78AQP 9N=6E@78P ON=6E@78PO-"
" IN=6E@78PO. IN=6E@78PO.IA MN=6E@78PO/ MN=6E@78POA 9N=6E@78POA9. MN=6E@78POM 5N=6E@78POM5. -N=6E@78POM5A 9N=6E@78POW IN=6E@78POWIX 5N=6E@78POWIX5. -N=6E@78POX 5N=6E@78POX5M <N=6E@78V 5N=6E@78V5. PN=6E@78V5.P- UN=6E@78W 5N=6E@8 7N=6E@875 /N=6E@87V AN=6E@A 7N=6E@A7- 5N=6E@A7-5O DN=6E@A7/ MN=6E@A78 5N="
"6E@A785O PN=6E@A785OPU IN=6E@A7P ON=6E@V PN=6E@VP. 5N=6E@VP.5- MN=6E@VP.5-MU <N=6E@VP.5-MU<D 4N=6E@VP.57 8N=6E@VP.578/ &N=6E@VP.578< $N=6E@VP.5< MN=6E@VP.5<MD 4N=6E@VP.5<MD47 /N=6E@VP.5<MD47/3 &N=6E@VP.5<MD47/3&O ;N=6E@VP/ .N=6E@VP/.7 8N=6E@VP/.8 7N=6E@VP7 8N=6E@VP78. $N=6E@VP78.$0 /N=6E@W MN=6E@WM7"
" 8N=6M @N=6M@8 7N=6M@87O EN=6M@O EN=6M@OE7 IN=6M@OE8 5N=6M@OE85< 7N=6M@OE< DN=6M@OED PN=6M@V EN=6O @N=6O@7 8N=6O@78. EN=6O@78.E5 MN=6O@785 EN=6O@785E. MN=6O@785E.M/ 0N=6O@785E.M/0I AN=6O@785E.M/0IAB 9N=6O@785E.M/0IAB9P RN=6O@785E.M/0V &N=6O@785E.M/0V&I WN=6O@785E.M/0V&IW9 QN=6O@785E.M/0V&IW9QP JN=6O"
"@785E.MA XN=6O@785E.MAXD LN=6O@785E.MAXV IN=6O@785E.MAXW aN=6O@785E.MAXWaV `N=6O@785E.MAXWaV`< IN=6O@785E.MAXWaV`<IU PN=6O@785E.MAXWaV`<I^ LN=6O@785E.MAXWaV`D IN=6O@785E.MAXWaV`DI^ PN=6O@785E.MAXWaV`^ UN=6O@785E.MAXWaV`^UD IN=6O@785E.MAXWaV`^UDIL PN=6O@785E.MAXWaV`^U] PN=6O@785E.MAXWaV`^U]P/ :N=6O@7"
"85E.MAXWaV`^U]P/:9 0N=6O@785E.MAXWaV`^U_ ]N=6O@785E.MD LN=6O@785E.MV 4N=6O@785E.MV4- <N=6O@785E.MV4-</ 0N=6O@785E.MV4/ <N=6O@785E.MV4/<L QN=6O@785E.MV4/<LQD PN=6O@785E.MV4/<LQDP- 0N=6O@785E.MV4< DN=6O@785E.MV4<DL -N=6O@785E.MV4A DN=6O@785EA MN=6O@785EAM- .N=6O@785EAMD <N=6O@785EM PN=6O@785EV MN=6O@7"
"8A 9N=6O@78A9. PN=6O@78A9.PQ JN=6O@78A9.PQJW RN=6O@78A90 EN=6O@78A90E. 5N=6O@78A90E.5- /N=6O@78A90E.5-/L MN=6O@78A90E/ WN=6O@78M PN=6O@78MPI QN=6O@78MPIQ5 AN=6O@78MPIQ5A. EN=6O@78MPIQ9 WN=6O@78MPIQA BN=6O@78MPIQABJ RN=6O@78MPIQR EN=6O@78MPIQRE5 <N=6O@78MPIQW RN=6O@78MPIQWR5 XN=6O@78MPIQWR5X. EN=6O@7"
"8MPIQWR5X.E< $N=6O@78MPIQWR5X.EA _N=6O@78MPIQWR5X.EA_` aN=6O@78MPIQWR5X.EA_`aD JN=6O@78MPIQWR5X.EA_`aDJ: -N=6O@78MPIQWR5XA EN=6O@78MPIQWR5XAE0 :N=6O@78MPIQWR9 AN=6O@78MPIQWR9A: JN=6O@78MPIQWR9A:J0 /N=6O@78MPIQWRB JN=6O@78MPIQWRBJZ XN=6O@78MPIQWRBJZX5 `N=6O@78U MN=6O@78UME DN=6O@78V <N=6O@8 7N=6O@87M"
" EN=6O@A 5N=6O@M EN=6O@ME7 5N=6O@MED PN=6O@MEDPI VN=6O@MEDPQ XN=6O@MEDPQXI WN=6O@MEDPQXIWV 4N=6O@V EN=6U 8NE @NE@6 7NE@670 MNE@670M8 /NE@670MO XNE@670MOX8 =NE@678 5NE@67A INE@67AI/ =NE@67AI/=5 MNE@67AI/=5MV DNE@67AI/=8 MNE@67AI/=O PNE@67AI/=OP5 :NE@67AI/=OP5:V UNE@67AI/=OPJ .NE@67AI0 :NE@67AI0:O PNE"
"@67AI0:P -NE@67AI0:P-5 MNE@67AI0:P-5MV /NE@67AI8 =NE@67AI8=5 MNE@67AI8=O PNE@67AI8=OP5 /NE@67AI8=OPJ /NE@67AI8=OPJ/- .NE@67AI8=OPR -NE@67AI8=OPR-5 :NE@67AI8=OPR-5:9 JNE@67AI8=OPR-5:9JB ZNE@67AI8=OPR-5:9JBZU .NE@67AI8=OPR-5:9JBZU.0 2NE@67AI8=OPR-5:9JBZU.02/ (NE@67AI8=OPR-5:9JBZU.02/(' &NE@67AI8=OPR-5"
":9JBZU.02/('&$ MNE@67AI8=OPR-5:9JBZU.02/('&$MX WNE@67AI8=OPR-5:9JBZU.02/('&$MXW) aNE@67AI8=OPW -NE@67AI8=OPW-9 BNE@67AI8=OPW-9BX QNE@67AI8=P ONE@67AIJ ONE@67AIO =NE@67AIO=/ .NE@67AIO=/.8 WNE@67AIO=/.8W0 5NE@67AIO=/.8W05< 'NE@67AIO=5 MNE@67AIO=5MV QNE@67AIO=8 PNE@67AIO=8P5 /NE@67AIO=9 MNE@67AIO=9M5 ."
"NE@67AIO=9M5.- 4NE@67AIO=9M5.-4< 8NE@67AIO=9M5.4 DNE@67AIO=9M5.4DV <NE@67AIO=9M5.4DV<Q BNE@67AIO=9M5.8 <NE@67AIO=9M5.8<V -NE@67AIO=9M5.8<V-Q /NE@67AIO=9M5.D XNE@67AIO=9M5.DX0 8NE@67AIO=9M5.DX08< PNE@67AIO=9M5.J -NE@67AIO=9M5.Q -NE@67AIO=9M5.Q-4 <NE@67AIO=9M5.Q-4<V _NE@67AIO=9M8 -NE@67AIO=9M8-5 PNE@6"
"7AIO=9M8-5P$ QNE@67AIO=9M8-5P$QR .NE@67AIO=9M8-5P$QR./ 0NE@67AIO=9M8-5P$QR./0< (NE@67AIO=9M8-5P< QNE@67AIO=9M8-5P<QR 3NE@67AIO=9M8-5PQ RNE@67AIO=9M8-5PR QNE@67AIO=9M8-Q /NE@67AIO=9M8-Q/. &NE@67AIO=9M8-R QNE@67AIO=9M8-RQ5 VNE@67AIO=9M8-RQ5V< /NE@67AIO=9M8-RQ5V</. &NE@67AIO=9M8-RQ5V</.&L 0NE@67AIO=9M8"
"-RQ5V</.&L0X PNE@67AIO=9M8-RQ5V</.&L0XPJ KNE@67AIO=9MQ 5NE@67AIO=9MQ5U VNE@67AIO=9MV XNE@67AIO=J QNE@67AIO=JQ/ RNE@67AIO=M 5NE@67AIO=M59 PNE@67AIO=M59P8 QNE@67AIP ONE@67AIPOB QNE@67AIPOBQX JNE@67AIPOJ QNE@67AIPOW XNE@67AIPOWXV QNE@67AIPOX =NE@67AIQ PNE@67AIV =NE@67AIV=5 .NE@67AIV=5./ MNE@67AIV=5./MD"
" <NE@67AIV=O PNE@67AIV=OP/ UNE@67AIW =NE@67AIW=O MNE@67AIW=OM9 5NE@67AIW=OM95D VNE@67AIW=OM95DVL UNE@67P ONE@67PO0 MNE@67PO0MA INE@67PO0MU =NE@67PO8 =NE@67POA =NE@67POA=/ 8NE@67POA=5 8NE@67POA=58X MNE@67POA=8 MNE@67POA=8MW XNE@67POA=< 8NE@67POA=U 8NE@67POA=U85 MNE@67POA=U8W MNE@67POA=U8WM4 <NE@67POM"
" INE@67POV =NE@67POW INE@67POWIA 8NE@67POWIX VNE@67POX WNE@67POXWA QNE@67POXWM QNE@67POXW_ `NE@67POXW_`a INE@67POXW_`aIQ ANE@67V =NE@67V=4 5NE@67V=5 PNE@67V=8 ONE@67W MNE@7 MNE@7M6 PNE@7M6PA =NE@7M6PA=< 9NE@7M6PA=<9I BNE@7M6PA=<9IBO 8NE@7M6PA=D 9NE@7M6PA=D9O 8NE@7M6PA=O 5NE@7M6PA=O5< 4NE@7M6PA=O5<4D"
" ;NE@7M6PA=Q 5NE@7M6PA=Q5< 4NE@7M6PA=Q5X /NE@7M6PA=Q5X/. 0NE@7M6PA=Q5X/.0- 8NE@7M6PA=Q5X/.0-89 :NE@7M6PA=Q5X/.0-89:( &NE@7M6PA=Q5X/.0-89:(&$ 'NE@7M6PI <NE@7M6PI<D =NE@7M6PI<L =NE@7M6PI<L=4 5NE@7M6PO VNE@7M6POV9 =NE@7M6POV9=< 4NE@7M6POV9=<4D ;NE@7M6POV9=<4D;+ 3NE@7M6POV9=<4D;+3C 8NE@7M6POV9=<4D;+3C85"
" /NE@7M6POV9=<4D;+3C85/' :NE@7M6POV9=<4D;+3C85/':U LNE@7M6POV9=<4D;+3C85/':UL` 0NE@7M6POV9=<4D;+3C85/U LNE@7M6POV9=<4D;+3C85/ULI ANE@7M6POV9=<4D;+3C85/ULIAQ -NE@7M6POV9=<4D;+3C8A .NE@7M6POV9=<4D;+3C8A.- /NE@7M6POV9=<4D;+3C8A.-/5 LNE@7M6POV9=<4D;+3C8A.-/5L' &NE@7M6POV9=<4D;+3C8A.-/I 5NE@7M6POV9=<4D;C"
" KNE@7M6POV9=<4D;CK+ 8NE@7M6POV9=<4D;CK+8A 5NE@7M6POV9=<4D;CK+8A50 INE@7M6POV9=<4D;CK+8A50IL /NE@7M6POV9=<4D;CK+8A50IL/& -NE@7M6POV9=<4D;CK+8A50IL/&-. $NE@7M6POV9=<4D;CK+8A50IL/&-.$# BNE@7M6POV9=<4D;I 3NE@7M6POV9=<4D;I3L 5NE@7M6POV9=<4D;I3L5C JNE@7M6POV9=<4D;K 3NE@7M6POV9=<4D;K3I LNE@7M6POV9=<4D;K3L"
" 5NE@7M6POV9=<4D;K3L5I .NE@7M6POV9=<4D;K3L5I./ CNE@7M6POV9=<4D;K3L5I./C+ (NE@7M6POV9=<4D;K3L5U WNE@7M6POV9=<4D;U 5NE@7M6POV9=<4D;W 5NE@7M6POV9=<4D;W5. 3NE@7M6POV9=<4D;W5C KNE@7M6POV9=<4D;W5K 3NE@7M6POV9=<4D;W5K3- CNE@7M6POV9=<4I 5NE@7M6POV9=<4I5. 3NE@7M6POV9=<4I5.3C ;NE@7M6POV9=<4I5.3C;+ -NE@7M6POV9"
"=<4I5.3C;+-D 'NE@7M6POV9=<4I5.3X -NE@7M6POV9=<4I5.3X-W aNE@7M6POV9=<4I53 -NE@7M6POV9=<4I53-. &NE@7M6POV9=<4I5D CNE@7M6POV9=<4I5DC- ;NE@7M6POV9=<4I5DC; 3NE@7M6POV9=<4I5DC;3- $NE@7M6POV9=<4I5DC;3-$L KNE@7M6POV9=<4I5DC;3L KNE@7M6POV9=<4I5DC;3LK- $NE@7M6POV9=<4I5DC;3LK-$X .NE@7M6POV9=<4I5DC;3LK-$X./ 8NE"
"@7M6POV9=<4I5DC;3LK-$X./8A (NE@7M6POV9=<4I5DC;3LK-$X./8A(^ `NE@7M6POV9=<4I5DC;3LK-$X./8A(^`U WNE@7M6POV9=<4I5DC;3LKW QNE@7M6POV9=<4I5DC;3LKX QNE@7M6POV9=<4I5DC;3LKXQ- .NE@7M6POV9=<4I5DC;3LKXQ-.$ 'NE@7M6POV9=<4I5DC;3LKXQ-.$'/ JNE@7M6POV9=<4I5DC;3LKXQ-.$'W 0NE@7M6POV9=<4I5DC;3LKXQ-.$'W0& 8NE@7M6POV9=<"
"4I5DC;3LKXQ-.W 0NE@7M6POV9=<4I5DC;3LKXQ-.W0^ _NE@7M6POV9=<4I5DC;3LKXQ-.W0^_` UNE@7M6POV9=<4I5DC;3LKXQ-.W0^_`U$ 8NE@7M6POV9=<4I5DC;3LKXQW aNE@7M6POV9=<4I5DC;3LKXQWaJ ANE@7M6POV9=<4I5DC;3LKXQWaJA8 RNE@7M6POV9=<4I5DC;3LKXQWaR ANE@7M6POV9=<4I5DC;3LKXQWaRA8 JNE@7M6POV9=<4I5DC;3LKXQWaRA8JB -NE@7M6POV9=<4I"
"5DC;3LKXQWaRA8JB-_ /NE@7M6POV9=<4I5DC;3LKXQWaRA8JB-_/' bNE@7M6POV9=<4I5DC;3LKXQWaRA8JB-_/^ YNE@7M6POV9=<4I5DC;3LKXQWaRA8JB-_/^Y' `NE@7M6POV9=<4I5DC;3LKXQWaRA8JB-_/^Y'`U ]NE@7M6POV9=<4I5DC;3LKXQWaRA8JB-_/^YU `NE@7M6POV9=<4I5DC;3LKXQWa^ _NE@7M6POV9=<4I5DC;3LKXQWa^_R ANE@7M6POV9=<4I5DC;3LKXQWa^_RA` ]NE"
"@7M6POV9=<4I5DC;3LKXQWa^_RA`]U -NE@7M6POV9=<4I5DC;3LKXQWa^_RA`]U-8 bNE@7M6POV9=<4I5DC;3LKXQWa^_` ]NE@7M6POV9=<4I5DC;3LKXQWa_ bNE@7M6POV9=<4I5DC;3LKXQWa_bZ YNE@7M6POV9=<4I5DC;3LKXQWa_bZY` ^NE@7M6POV9=<4I5DC;3LKXQWa_bZY`^T SNE@7M6POV9=<4I5DC;3LKXQ^ _NE@7M6POV9=<4I5DC;3LKXQ^_` JNE@7M6POV9=<4I5DC;3LKXQ_"
" WNE@7M6POV9=<4I5DC;3LK_ WNE@7M6POV9=<4I5DC;3LK_WU JNE@7M6POV9=<4I5DCU WNE@7M6POV9=<4I5DCUWL KNE@7M6POV9=<4I5L .NE@7M6POV9=<4I5L.- CNE@7M6POV9=<4I5L.-C; KNE@7M6POV9=<4I5L.X 0NE@7M6POV9=<4I5L.X0/ (NE@7M6POV9=<4I5L.X0/(- 8NE@7M6POV9=<4I5X .NE@7M6POV9=<4I5X.D 0NE@7M6POV9=<4I5X.D0U CNE@7M6POV9=<4L 5NE@7"
"M6POV9=<4L5. KNE@7M6POV9=<4L5.KI -NE@7M6POV9=<4L5.KI-D CNE@7M6POV9=<4L5.KI-DC; 3NE@7M6POV9=<4L5D KNE@7M6POV9=<4L5DK. CNE@7M6POV9=<4L5DK.CQ INE@7M6POV9=<4L5DK.CU ;NE@7M6POV9=<4L5DK.CU;3 +NE@7M6POV9=<4L5DKI CNE@7M6POV9=<4L5DKIC; 3NE@7M6POV9=<4L5DKIC;3- $NE@7M6POV9=<4L5DKIC;3-$X .NE@7M6POV9=<4L5DKIC;3-"
"$X./ 8NE@7M6POV9=<4L5DKIC;3-$X./8A (NE@7M6POV9=<4L5DKIC;3-$X./8U &NE@7M6POV9=<4L5DKIC;3-$_ .NE@7M6POV9=<4L5DKIC;3-$_./ (NE@7M6POV9=<4L5DKIC;3-$_./(, WNE@7M6POV9=<4L5DKIC;3-$_./(,WU `NE@7M6POV9=<4L5DKIC;3X /NE@7M6POV9=<4L5DKIC;3X/' 0NE@7M6POV9=<4L5DKIC;3X/. 'NE@7M6POV9=<4L5DKIC;3X/.'( 0NE@7M6POV9=<4L"
"5DKIC;3X/.'(0& 8NE@7M6POV9=<4L5DKIC;3X/.'(0&8A $NE@7M6POV9=<4L5DKIC;3X/.'(0&8A$# -NE@7M6POV9=<4L5DKIC;3_ .NE@7M6POV9=<4L5DKIC;3_./ (NE@7M6POV9=<4L5DKIC;3_./(- $NE@7M6POV9=<4L5DKICX ;NE@7M6POV9=<4L5DKICX;W /NE@7M6POV9=<4L5DKQ RNE@7M6POV9=<4L5DKQRU ^NE@7M6POV9=<4L5DKQRW XNE@7M6POV9=<4L5DKQRWX_ UNE@7M6"
"POV9=<4L5DKU ^NE@7M6POV9=<4L5DKU^I WNE@7M6POV9=<4L5DKU^IWX QNE@7M6POV9=<4L5DKU^IWXQa _NE@7M6POV9=<4L5DKU^W aNE@7M6POV9=<4L5DKU^Wa+ ;NE@7M6POV9=<4L5DKU^Wa+;, XNE@7M6POV9=<4L5DKU^Wa+;,X_ `NE@7M6POV9=<4L5DKU^Wa+;,X_`3 .NE@7M6POV9=<4L5DKU^WaI XNE@7M6POV9=<4L5DKU^` ;NE@7M6POV9=<4L5DKU^`;- CNE@7M6POV9=<4L"
"5DKW ANE@7M6POV9=<4L5DKWAB .NE@7M6POV9=<4L5DKWAU -NE@7M6POV9=<4L5DKWAU-8 /NE@7M6POV9=<4L5DKWA_ UNE@7M6POV9=<4L5DKWA_U^ 8NE@7M6POV9=<4L5DKWA_U^8Q 3NE@7M6POV9=<4L5I CNE@7M6POV9=<4L5ICD KNE@7M6POV9=<4Q 5NE@7M6POV9=<4U WNE@7M6POV9=<4UWL 5NE@7M6POV9=<4W 5NE@7M6POV9=D <NE@7M6POV9=D<; 4NE@7M6POV9=I 5NE@7M6"
"POV9=I54 BNE@7M6POV9=I54BR 8NE@7M6POV9=I54BR8L DNE@7M6POV9=I54BR8LDA :NE@7M6POV9=I54BR8LDA:0 JNE@7M6POV9=I54BR8LDA:0J2 /NE@7M6POV9=I54BR8LDA:0J2/. -NE@7M6POV9=I54BR8LDA:0J2/.-W YNE@7M6POV9=I54BR8LDA:0J2/.-WYQ `NE@7M6POV9=I54BR8LDA:0J2/.-WYQ`X <NE@7M6POV9=I5< 4NE@7M6POV9=I5<4. 3NE@7M6POV9=I5<4.3C DNE"
"@7M6POV9=I5<4.3D ;NE@7M6POV9=I5<4.3X -NE@7M6POV9=I5<4.3X-D ;NE@7M6POV9=I5<4.3X-D;W 'NE@7M6POV9=I5<4.3X-D;W'K 8NE@7M6POV9=I5<4.3X-W aNE@7M6POV9=I5<4.3X-WaD 'NE@7M6POV9=I5<43 .NE@7M6POV9=I5<43.$ -NE@7M6POV9=I5<43.- $NE@7M6POV9=I5<4D CNE@7M6POV9=I5<4DC- .NE@7M6POV9=I5<4DC-.$ 'NE@7M6POV9=I5<4DC; 3NE@7M6"
"POV9=I5<4DC;3L KNE@7M6POV9=I5<4DC;3LK- $NE@7M6POV9=I5<4DC;3LK-$U WNE@7M6POV9=I5<4DC;3LK-$X 0NE@7M6POV9=I5<4DC;3LK-$X08 /NE@7M6POV9=I5<4DC;3LKU WNE@7M6POV9=I5<4DC;3LKX 0NE@7M6POV9=I5<4DC;3LKX0/ (NE@7M6POV9=I5<4DC;3LKX08 /NE@7M6POV9=I5<4DC;3LKX08/& (NE@7M6POV9=I5<4DC;3LKX08/&(. WNE@7M6POV9=I5<4DC;3LKX"
"08/&(.W^ `NE@7M6POV9=I5<4DC;3LKX08/( WNE@7M6POV9=I5<4DC;3LKX08/(W^ `NE@7M6POV9=I5<4DC;3LKX08/(W^`' ANE@7M6POV9=I5<4DC;3LKX08/(W^`'A. RNE@7M6POV9=I5<4DC;3LK_ ^NE@7M6POV9=I5<4DC;3LK_^] WNE@7M6POV9=I5<4DC;3LK_^]WU .NE@7M6POV9=I5<4DCU WNE@7M6POV9=I5<4DCUWL KNE@7M6POV9=I5<4DCX .NE@7M6POV9=I5<4L .NE@7M6PO"
"V9=I5D LNE@7M6POV9=I5DLX CNE@7M6POV9=I5L DNE@7M6POV9=I5LDX .NE@7M6POV9=I5U WNE@7M6POV9=I5X .NE@7M6POV9=I5X.$ 8NE@7M6POV9=I5X.W 0NE@7M6POV9=I5X.W0' 8NE@7M6POV9=I5X.W0'8/ ANE@7M6POV9=I5X.W0'8^ (NE@7M6POV9=I5X.W0'8^() :NE@7M6POV9=I5X.W0/ (NE@7M6POV9=I5X.W0/($ 8NE@7M6POV9=I5X.W0/($8' ANE@7M6POV9=I5X.W0/"
"($8^ ANE@7M6POV9=I5X.W0/(- $NE@7M6POV9=I5X.W0/(-$^ 8NE@7M6POV9=I5X.W0/(-$^8U :NE@7M6POV9=I5X.W0/(^ 8NE@7M6POV9=I5X.W0/(^8$ 'NE@7M6POV9=I5X.W0/(^8U :NE@7M6POV9=I5X.W0/(^8U:' &NE@7M6POV9=I5X.W0/(^8U:L ANE@7M6POV9=I5X.W0/(^8U:LA$ 'NE@7M6POV9=I5X.W0/(^8U:LA$'B _NE@7M6POV9=I5X.W0/(^8U:LA$'B_` KNE@7M6POV9"
"=I5X.W0/(^8U:LA$'B_`K2 RNE@7M6POV9=I5X.W0/(^8U:LA$'B_`K2RD QNE@7M6POV9=I5X.W0/(^8U:LA$'B_`K2RDQ< JNE@7M6POV9=I5X.W0/(^8U:LA$'B_`K2RDQ<JZ 3NE@7M6POV9=I5X.W0/(^8U:LA$'B_`K< CNE@7M6POV9=I5X.W0/(^8U:LA$'B_`K<CD 4NE@7M6POV9=I5X.W0/(^8U:LA' RNE@7M6POV9=I5X.W0/(^8U:LA'R& -NE@7M6POV9=I5X.W0/(^8U:LAD RNE@7M6"
"POV9=I5X.W0/(^8U:LADR4 QNE@7M6POV9=I5X.W0/(^8U:LADR4Q' &NE@7M6POV9=I5X.W0^ 8NE@7M6POV9=I5X.W0^8' :NE@7M6POV9=I5X.W0^8':B ANE@7M6POV9=I5X.W0^8':BA2 RNE@7M6POV9=I5X.W0^8':BA2RU QNE@7M6POV9=I5X.W0^8':BA2RUQ/ &NE@7M6POV9=I5X.W0^8':BA2RUQ/&1 (NE@7M6POV9=I5X.W0^8( :NE@7M6POV9=I5X.W0^8/ (NE@7M6POV9=I5X.W0^"
"8/(- $NE@7M6POV9=I5X.W0^8/(-$U BNE@7M6POV9=I5X.W0^8/(U BNE@7M6POV9=I5X.W0^8U :NE@7M6POV9=I5X.W0^8U:/ ANE@7M6POV9=I5X.W0^8U:L ANE@7M6POV9=I5X.W0^8U:LA/ RNE@7M6POV9=I5X.W0^8U:LA/RB (NE@7M6POV9=I5X.W0^8U:LA/RB(2 QNE@7M6POV9=I5X.W0^8U:LA/RD &NE@7M6POV9=I5X.W0^8U:LA/RD&4 aNE@7M6POV9=I5X.W0^8U:LA/RD&4a( -"
"NE@7M6POV9=I5X.W0^8U:LAD RNE@7M6POV9=I5X.W0^8U:LADR4 QNE@7M6POV9=I5X.W0^8U:LADR4Q- aNE@7M6POV9=Q 5NE@7M6POV9=Q5< INE@7M6POV9=Q5W .NE@7M6POV9=Q5X .NE@7M6POV9=Q5X.$ /NE@7M6POV9=Q5X.W 0NE@7M6POV9=Q5X.W0' 8NE@7M6POV9=Q5X.W0'8/ &NE@7M6POV9=Q5X.W0/ (NE@7M6POV9=Q5X.W0/(- 8NE@7M6POV9=Q5X.W0/(-8^ $NE@7M6POV9"
"=Q5X.W0/(-8^$U :NE@7M6POV9=Q5X.W0/(-8^$U:L ANE@7M6POV9=Q5X.W0/(-8^$U:LAD _NE@7M6POV9=Q5X.W0/(^ 8NE@7M6POV9=Q5X.W0/(^8- $NE@7M6POV9=Q5X.W0/(^8-$U :NE@7M6POV9=Q5X.W0/(^8-$U:L ANE@7M6POV9=Q5X.W0/(^8U BNE@7M6POV9=Q5X.W0/(^8UBL ANE@7M6POV9=Q5X.W0^ 8NE@7M6POV9=Q5X.W0^8' :NE@7M6POV9=Q5X.W0^8':U ANE@7M6POV9"
"=Q5X.W0^8':UA/ INE@7M6POV9=Q5X.W0^8':UA/IJ BNE@7M6POV9=Q5X.W0^8U BNE@7M6POV9=Q5X.W0^8UBL ANE@7M6POV9=Q5X.W0^8UBLA: JNE@7M6POV9=Q5X.W0^8UBLA:JR INE@7M6POV9=Q5X.W0^8UBLAD _NE@7M6POV9=W 5NE@7M6POV9=W5A aNE@7M6POV9=W5AaQ XNE@7M6POV9=W5AaQX< CNE@7M6POV9=W5AaQX<C4 RNE@7M6POV9=W5AaQX<C4R; 3NE@7M6POV9=W5AaQ"
"X<C4R;3D +NE@7M6POV9=W5AaQX<C4R;3D+U INE@7M6POV9=W5AaQX<C4R;3D+UI^ /NE@7M6POV9=W5AaU ^NE@7M6POV9=W5AaX QNE@7M6POV9=W5AaXQJ RNE@7M6POV9=W5AaXQJRZ INE@7M6POV9=W5AaXQJR^ _NE@7M6POV9=W5AaXQJR` _NE@7M6POV9=W5AaXQ^ UNE@7M6POV9=W5AaXQ^U` INE@7M6POV9=W5Aa^ UNE@7M6POV9=W5Aa^U` _NE@7M6POV9=W5Aa^U`_b XNE@7M6PO"
"V9=W5Aa` _NE@7M6POV9=W5Aa`_X bNE@7M6POV9=W5Aa`_Xb< QNE@7M6POV9=W5Aa`_Xb<QD 4NE@7M6POV9=W5Aa`_Xb<QD4U INE@7M6POV9=W5Aa`_Xb<QD4UI. CNE@7M6POV9=W5I aNE@7M6POV9=W5Ia< 4NE@7M6POV9=W5Ia<4. 3NE@7M6POV9=W5Ia<4U ^NE@7M6POV9=W5IaA XNE@7M6POV9=W5IaAX4 QNE@7M6POV9=W5IaAX4Q< BNE@7M6POV9=W5IaAX4Q<BR :NE@7M6POV9=W"
"5IaAX4Q<BR:Z 3NE@7M6POV9=W5IaAX4Q<BR:Z3C ;NE@7M6POV9=W5IaAX4Q<BR:Z3C;+ 8NE@7M6POV9=W5IaAX< 4NE@7M6POV9=W5IaAX^ _NE@7M6POV9=W5IaQ XNE@7M6POV9=W5IaQX4 -NE@7M6POV9=W5IaQX4-< ANE@7M6POV9=W5IaQX< 4NE@7M6POV9=W5Ia^ _NE@7M6POV9=W5Ia^_U LNE@7M6POV9=W5^ ANE@7M6POV9=W5^AQ XNE@7M6POV9=X QNE@7M6POV9=XQW _NE@7M6"
"POV9=XQW_` aNE@7M6POV9=_ WNE@7M6POVA =NE@7M6POVA=< 9NE@7M6POVA=<9I BNE@7M6POVA=<9IB2 8NE@7M6POVA=<9IBR :NE@7M6POVA=<9IBR:Q 4NE@7M6POVA=<9Q 8NE@7M6POVA=<9Q8I RNE@7M6POVA=I 5NE@7M6POVA=I54 -NE@7M6POVA=I54-& $NE@7M6POVA=I54-< /NE@7M6POVA=I54-</D .NE@7M6POVA=I54-</D.0 QNE@7M6POVA=I54-</D.0Q( 8NE@7M6POVA"
"=I54-</D.0Q(8' $NE@7M6POVA=I5< 4NE@7M6POVA=I5<4. 3NE@7M6POVA=I5<4.3C ;NE@7M6POVA=I5<4.3C;+ -NE@7M6POVA=I5<4.3X -NE@7M6POVA=I5<4.3X-W aNE@7M6POVA=I5<4.3X-WaD ;NE@7M6POVA=I5<4D CNE@7M6POVA=I5<4DC- ;NE@7M6POVA=I5<4DC; 3NE@7M6POVA=I5<4DC;3L UNE@7M6POVA=I5<4DC;3LUS :NE@7M6POVA=I5<4DC;3LUS:B .NE@7M6POVA=I"
"5<4DC;3LUS:B./ 9NE@7M6POVA=I5<4DC;3LUS:B./9X 8NE@7M6POVA=I5<4DC;3LUS:B./9X8W 'NE@7M6POVA=I5<4DC;3LUS:B./9X8W'- &NE@7M6POVA=I5<4DC;3LUS:X JNE@7M6POVA=I5<4DC;3LUX .NE@7M6POVA=I5<4DC;3LUX./ 8NE@7M6POVA=I5<4DC;3LUX./8W KNE@7M6POVA=I5<4DC;3LUX./8WK0 (NE@7M6POVA=I5<4DC;3LUX./8WK0(' &NE@7M6POVA=I5<4DC;3LUX"
"./8WK0(_ 9NE@7M6POVA=I5<4DC;3LUX./8WK0(_9' &NE@7M6POVA=I5<4DC;3LUX./8WK_ 9NE@7M6POVA=I5<4L .NE@7M6POVA=I5U WNE@7M6POVA=I5W .NE@7M6POVA=I5X .NE@7M6POVA=I5X.$ /NE@7M6POVA=I5X.$/- &NE@7M6POVA=I5X.$/-&W 8NE@7M6POVA=I5X.$/-&W8^ 9NE@7M6POVA=I5X.$/-&W8^94 <NE@7M6POVA=I5X.$/-&W8^94<; DNE@7M6POVA=I5X.$/0 (NE"
"@7M6POVA=I5X.$/0(- &NE@7M6POVA=I5X.$/0(-&' #NE@7M6POVA=I5X.$/0(-&W 9NE@7M6POVA=I5X.$/0(-&W98 BNE@7M6POVA=I5X.$/0(-&W98B^ 'NE@7M6POVA=I5X.$/0(-&W98B^'J RNE@7M6POVA=I5X.$/0(-&W98B^'JRQ ZNE@7M6POVA=I5X.$/0(-&W98B^'JRQZ: 2NE@7M6POVA=I5X.$/D 9NE@7M6POVA=I5X.$/D94 <NE@7M6POVA=I5X.$/D98 RNE@7M6POVA=I5X.$/D"
"98R& WNE@7M6POVA=I5X.$/D98R&W0 QNE@7M6POVA=I5X.$/D98RB 0NE@7M6POVA=I5X.$/D98RB0& (NE@7M6POVA=I5X.$/D98RB0&(: 'NE@7M6POVA=I5X.$/D98RB04 :NE@7M6POVA=I5X.$/D98RB04:2 YNE@7M6POVA=I5X.$/D98RJ BNE@7M6POVA=I5X.$/D98RJB( 0NE@7M6POVA=I5X.$/D98RJB(0U 4NE@7M6POVA=I5X.$/D98RJBW LNE@7M6POVA=I5X.$/D98RW UNE@7M6PO"
"VA=I5X.- $NE@7M6POVA=I5X.-$< 4NE@7M6POVA=I5X.-$<4D ;NE@7M6POVA=I5X.< 0NE@7M6POVA=I5X.D <NE@7M6POVA=I5X.D<W 0NE@7M6POVA=I5X.W 0NE@7M6POVA=I5X.W0/ 8NE@7M6POVA=I5X.W0/8$ 9NE@7M6POVA=I5X.W0/8$9( aNE@7M6POVA=I5X.W0/8$9(a- JNE@7M6POVA=I5X.W0/8$9(a-JR :NE@7M6POVA=I5X.W0/8$9- &NE@7M6POVA=I5X.W0/8$9^ (NE@7M6"
"POVA=I5X.W0/8$9^(& RNE@7M6POVA=I5X.W0/8$9^(&R- BNE@7M6POVA=I5X.W0/8$9^(' BNE@7M6POVA=I5X.W0/8$9^('B- RNE@7M6POVA=I5X.W0/8( `NE@7M6POVA=I5X.W0/8(`$ JNE@7M6POVA=I5X.W0/8(`$JB :NE@7M6POVA=I5X.W0/8(`$JB:9 2NE@7M6POVA=I5X.W0/8^ 'NE@7M6POVA=I5X.W0/8^'U _NE@7M6POVA=I5X.W0/8^'U_` 9NE@7M6POVA=I5X.W0/8^'U_`9L"
" DNE@7M6POVA=I5X.W0/8^'U_`9LDC BNE@7M6POVA=I5X.W0/8^'U_`9LDCB$ &NE@7M6POVA=I5X.W0/8^'U_`9LDCB$&< #NE@7M6POVA=I5X.W0U 8NE@7M6POVA=I5X.W0^ 8NE@7M6POVA=I5X.W0^8/ (NE@7M6POVA=I5X.W0^8/(- $NE@7M6POVA=I5X.W0^8/(-$U _NE@7M6POVA=I5X.W0^8/(-$U_` 9NE@7M6POVA=I5X.W0^8U _NE@7M6POVA=I5X.W0^8U_/ 9NE@7M6POVA=I5X.W"
"0^8U_/9$ (NE@7M6POVA=I5X.W0^8U_/9$(' LNE@7M6POVA=I5X.W0^8U_/9$('L` BNE@7M6POVA=I5X.W0^8U_/9( JNE@7M6POVA=I5X.W0^8U_/9` RNE@7M6POVA=I5X.W0^8U_/9`RB (NE@7M6POVA=I5X.W0^8U_L `NE@7M6POVA=I5X.W0^8U_L`/ DNE@7M6POVA=I5X.W0^8U_L`D <NE@7M6POVA=I5X.W0^8U_` 9NE@7M6POVA=I5X.W0^8U_`9( 'NE@7M6POVA=I5X.W0^8U_`9('/"
" BNE@7M6POVA=I5X.W0^8U_`9(': QNE@7M6POVA=I5X.W0^8U_`9(':Q/ JNE@7M6POVA=I5X.W0^8U_`9/ BNE@7M6POVA=I5X.W0^8U_`9/B$ 'NE@7M6POVA=I5X.W0^8U_`9/B( 'NE@7M6POVA=I5X.W0^8U_`9/B('$ &NE@7M6POVA=I5X.W0^8U_`9/B('$&L DNE@7M6POVA=I5X.W0^8U_`9/B('$&LDC RNE@7M6POVA=I5X.W0^8U_`9/B('$&LDCRJ QNE@7M6POVA=I5X.W0^8U_`9/B("
"'& $NE@7M6POVA=I5X.W0^8U_`9/B('&$# 1NE@7M6POVA=I5X.W0^8U_`9/B('&$#1J -NE@7M6POVA=I5X.W0^8U_`9/B('J )NE@7M6POVA=I5X.W0^8U_`9/B('J)$ 1NE@7M6POVA=I5X.W0^8U_`9/BJ (NE@7M6POVA=I5X.W0^8U_`9/BJ(' &NE@7M6POVA=I5X.W0^8U_`9/BJ('&L RNE@7M6POVA=I5X.W0^8U_`9/BJ(: QNE@7M6POVA=I5X.W0^8U_`9/BJ(:QL RNE@7M6POVA=I5X.W"
"0^8U_`9/BJ(D RNE@7M6POVA=I5X.W0^8U_`9/BJ(DRQ ZNE@7M6POVA=I5X.W0^8U_`9/BJ(L DNE@7M6POVA=I5X.W0^8U_`9/BJ(LDC RNE@7M6POVA=I5X.W0^8U_`9/BJ(LDCR: QNE@7M6POVA=I5X.W0^8U_`92 RNE@7M6POVA=I5X.W0^8U_`92RQ bNE@7M6POVA=I5X.W0^8U_`94 <NE@7M6POVA=I5X.W0^8U_`9L DNE@7M6POVA=I5X.W0^8U_`9LD( BNE@7M6POVA=I5X.W0^8U_`9L"
"D/ BNE@7M6POVA=I5X.W0^8U_`9LD4 <NE@7M6POVA=I5X.W0^8U_`9LD4<C ;NE@7M6POVA=I5X.W0^8U_`9LD; KNE@7M6POVA=I5X.W0^8U_`9LDC BNE@7M6POVA=Q INE@7M6POVA=QIR JNE@7M6POVA=QIRJB 5NE@7M6POVA=QIW aNE@7M6POVA=QIWa4 RNE@7M6POVA=QIWa4RD :NE@7M6POVA=QIWaX RNE@7M6POVA=QIWaXRJ 5NE@7M6POVA=QIWaXRJ5Z 9NE@7M6POVA=QIWaXRJ5Z"
"98 :NE@7M6POVA=QIWaXRJ5Z98:< 4NE@7M6POVA=QIWaXRJ5Z98:<4. 3NE@7M6POVA=QIWaXRJ5Z98:<4.3D LNE@7M6POVA=QIWaXRJ5Z98:<4D CNE@7M6POVA=QIWaXRJ5Z98:<4DCL ;NE@7M6POVA=QIWaXRJ5Z98:<4DCL;K SNE@7M6POVA=QIWaXRJ5Z98:<4DCL;KS- UNE@7M6POVA=QIWaXRJ5Z98:<4DCL;KS-U+ $NE@7M6POVA=QIWaXRJ5Z98:<4DCL;KS^ `NE@7M6POVA=QIWaXRJ"
"5Z98:<4DCL;KS^`_ ]NE@7M6POVA=QIWaXRJ5Z98:<4DCL;KS^`_]U bNE@7M6POVA=QIWaXR^ _NE@7M6POVA=QIWaXR^_` ]NE@7M6POVA=QIWaXR^_`]4 5NE@7M6POVA=QIWaXR_ 5NE@7M6POVA=QIWaXR_5J 9NE@7M6POVA=QIWaXR_5J98 BNE@7M6POVA=QIWaXR` _NE@7M6POVA=QIWaXR`_J 5NE@7M6POVA=U WNE@7M6POVA=UWI 5NE@7M6POVA=W 5NE@7M6POVA=W5I aNE@7M6POVA"
"=W5IaQ XNE@7M6POVA=W5^ XNE@7M6POVI DNE@7M6POVIDX QNE@7M6POVQ INE@7M6POVQIU 8NE@7M6POVQIU8` ^NE@7M6POVQIU8`^J ANE@7M6POVQIU8`^JAX WNE@7M6POVQIW XNE@7M6POVU 8NE@7M6POVU8A LNE@7M6POVU8_ =NE@7M6POVU8_=W XNE@7M6POVU8` ^NE@7M6POVW UNE@7M6POVWU= 8NE@7M6POVWU=8A aNE@7M6POVWU=8L DNE@7M6POVWU=8LD< /NE@7M6POVW"
"U=8LDA INE@7M6POVWU=8LDK ;NE@7M6POVWU=8LDK;< -NE@7M6POVWU=8LDK;I ANE@7M6POVWU=8_ ^NE@7M6POVWU=8_^A :NE@7M6POVWU=8_^A:9 0NE@7M6POVWU=8_^A:I 9NE@7M6POVWU=8_^A:I9B aNE@7M6POVWU=8_^A:I9Ba/ `NE@7M6POVWU=8_^A:I9Ba/`X DNE@7M6POVWU=8_^A:I9X RNE@7M6POVWU=8_^X aNE@7M6POVWU=8_^XaA INE@7M6POVWU=8_^XaAIJ 9NE@7M6"
"POVWU=8_^Xa] /NE@7M6POVWU=8_^Xa]/' -NE@7M6POVWU=8_^Xa]/( &NE@7M6POVWU=8_^Xa]/(&0 -NE@7M6POVWU=8_^Xa]/(&0-' )NE@7M6POVWU=8_^Xa]/0 DNE@7M6POVWU=8_^Xa]/0D4 <NE@7M6POVWU=8_^Xa]/0D4<5 (NE@7M6POVWU=8_^Xa]/0D4<5(K CNE@7M6POVWU=8_^Xa]/0D4<5(KC3 ;NE@7M6POVWU=8_^Xa]/0D4<5(KC3;L `NE@7M6POVWU=8_^Xa]/0D4<5(KC3;L"
"`b TNE@7M6POVWU=8_^Xa]/0D< .NE@7M6POVWU=8_^Xa]/0D<.K LNE@7M6POVWU=8_^Xa]/L -NE@7M6POVWU=8_^Xa]/L-. $NE@7M6POVWU=8_^] /NE@7M6POVWU=8_^]/( `NE@7M6POVWU=8_^]/(`a XNE@7M6POVWU=8_^]/0 (NE@7M6POVWU=8_^]/L `NE@7M6POVWU=8_^]/L`a XNE@7M6POVWU=8_^]/L`aX' 4NE@7M6POVWU=8_^]/L`aX( &NE@7M6POVWU=8_^]/L`aX(&A INE@7"
"M6POVWU=8_^]/L`aX(&AIJ 4NE@7M6POVWU=8_^]/L`aX(&I RNE@7M6POVWU=8_^]/L`aX(&IRA QNE@7M6POVWU=8_^]/L`aX(&IRAQB :NE@7M6POVWU=8_^]/L`aX(&IRAQB:J 5NE@7M6POVWU=8_^]/L`aX(&IRAQB:J59 YNE@7M6POVWU=8_^]/L`aXA 5NE@7M6POVWU=8_^]/L`aXA5. INE@7M6POVWU=8_^]/L`aXA5I 4NE@7M6POVWU=8_^]/L`aXA5I4< 9NE@7M6POVWU=8_^]/L`aXA"
"5b 9NE@7M6POVWU=8_^]/L`aXI ANE@7M6POVWU=8_^]/L`aXb -NE@7M6POVWU=8_^]/L`aXb-. $NE@7M6POVWU=8_^]/L`aXb-.$' 5NE@7M6POVWU=8_^]/L`aXb-.$'5& (NE@7M6POVWU=8_^]/L`aXb-I RNE@7M6POVWU=8_^]/L`aXb-IRB <NE@7M6POVWU=8_^]/X -NE@7M6POVWU=8_^]/X-Q aNE@7M6POVWU=8_^]/X-Qa. $NE@7M6POVWU=8_^]/X-Qa.$0 (NE@7M6POVWUL `NE@7"
"M6POVWUL`^ 8NE@7M6POVWU_ 8NE@7M6POV_ 8NE@7M6PQ INE@7M6PQIJ 8NE@7M6PQIJ8A 9NE@7M6PQIJ8A90 RNE@7M6PQIJ8A90RZ BNE@7M6PQIO VNE@7M6PQIOV8 WNE@7M6PQIOV8WX ANE@7M6PQIOV8WXAa `NE@7M6PQIU 8NE@7M6PQIU8W =NE@7M6PU 8NE@7M6PU8V .NE@7M6PV ONE@7M6PVOQ INE@7M6PVOW UNE@7M6PVOWUL _NE@7M6PW ONE@7M8 5NE@7M856 =NE@7M856"
"=4 -NE@7M856=4-< .NE@7M856=4-<./ QNE@7M856=4-<./Q& (NE@7M856=4-<./Q&($ 'NE@7M856=4-<./Q&($') DNE@7M856=4-<./Q&($')DO ;NE@7M856=4-<./QO (NE@7M856=4-<.L /NE@7M856=4-<.L/P ONE@7M856=4-<.L/PO0 KNE@7M856=4-<.L/PO0KX WNE@7M856=4-<.L/PO0KXW` aNE@7M856=4-<.L/PO0KXW`ab ANE@7M856=4-<.L/PO0KXW`abAB JNE@7M856=4"
"-<.L/PO0KXW`abABJR VNE@7M856=4-<.L/PO0KXW`abABJRVQ 9NE@7M856=4-<.L/PO0KXW`abABJRVQ9I 'NE@7M856=< 4NE@7M856=<4. DNE@7M856=<4L .NE@7M856=<4O DNE@7M856=<4P .NE@7M856=O PNE@7M85= 6NE@7M85=6O DNE@7M85=6OD< PNE@7M85O =NE@7M85O=4 <NE@7M85O=U VNE@7M85U ONE@7MA 5NE@7MA5= 6NE@7MA5=6O 0NE@7MI ANE@7MO XNE@7MOX8"
" DNE@7MOX8DC <NE@7MOX8DL =NE@7MOX8DU VNE@7MOX8DV WNE@7MOX8DVWU PNE@7MOX8DVWUPL _NE@7MOX8DVWUPL_^ `NE@7MOX8DW PNE@7MOX8DWPI `NE@7MOX8DWPI`a bNE@7MOX8DWPU VNE@7MOX8DWPV =NE@7MOX8DWPV=L 5NE@7MOX8DWPV=Q 6NE@7MOX8DWPV=Q6- INE@7MOX8DWPV=Q6-I5 ANE@7MOX8DWPa INE@7MOX8DWPaIQ ANE@7MOX9 =NE@7MOXA INE@7MOXI =NE"
"@7MOXV PNE@7MOXVP9 <NE@7MOXVPD LNE@7MOXVPDLU 6NE@7MOXVPDLU6W 5NE@7MOXVPDLU6W58 =NE@7MOXVPDLW _NE@7MOXVPQ ANE@7MOXVPW `NE@7MOXVPW`^ DNE@7MOXVPW`^D= 8NE@7MOXVPW`^D=8< /NE@7MOXVPW`^Db UNE@7MOXW PNE@7MOXWP9 DNE@7MOXWP9DV =NE@7MOXWP9DV=L 8NE@7MOXWP9DV=L8Q /NE@7MOXWP9DV=L8Q/I 6NE@7MOXWP9DV=L8Q/I6C `NE@7MO"
"XWPA VNE@7MOXWPQ INE@7MOXWPQIB aNE@7MOXWPQIV _NE@7MOXWPQIV_a ^NE@7MOXWPQI` 8NE@7MOXWPQI`86 ANE@7MOXWPQI`86A= VNE@7MOXWPQI`86A=VJ BNE@7MOXWPV _NE@7MOXWPV_9 ^NE@7MOXWPV_9^I ANE@7MOXWPV_D ^NE@7MOXWPV_L ^NE@7MOXWPV_U ^NE@7MOXWPV_U^9 LNE@7MOXWPV_U^a ]NE@7MOXWPV_U^a]Q JNE@7MOXWPV_U^a]QJ8 ANE@7MOXWPV_U^a]Q"
"J8AI 9NE@7MOXWPV_U^a]QJZ ANE@7MOXWPV_U^a]QJZAb INE@7MOXWPV_U^a]b LNE@7MOXWPV_U^a]bLK DNE@7MOXWPV_U^a]bLKD9 8NE@7MOXWPV_U^a]bLKD98Q =NE@7MOXWPV_U^a]bLKD< ;NE@7MOXWPV_U^a]bLKD<;= 8NE@7MOXWPV_U^a]bLKD<;=8Q RNE@7MOXWPV_U^a]bLKD<;=8QRA 9NE@7MOXWPV_U^a]bLKD<;=8QRA9/ :NE@7MOXWPV_U^a]bLKD<;=8QRA9/:I 4NE@7MO"
"XWPV_U^a]bLKD= ;NE@7MOXWPV_` aNE@7MOXWPV_a ^NE@7MOXWPV_a^9 DNE@7MOXWPV_a^9DU 5NE@7MOXWPV_a^9DU56 =NE@7MOXWPV_a^9DU56=< .NE@7MOXWPV_a^9DU5A INE@7MOXWPV_a^A 5NE@7MOXWPV_a^A56 UNE@7MOXWPV_a^L UNE@7MOXWPV_a^LUA 9NE@7MOXWPV_a^LU` bNE@7MOXWPV_a^LU`bA INE@7MOXWPV_a^U /NE@7MOXWPV_a^b UNE@7MP ONE@7MPO6 8NE@7"
"MPO689 5NE@7MPO6895I =NE@7MPO68A =NE@7MPO68A=W INE@7MPO68A=WIR VNE@7MPO68I QNE@7MPO8 6NE@7MPO86= INE@7MPOI 6NE@7MPOU 6NE@7MPOV WNE@7MPOW INE@7MPOWI= 8NE@7MU PNE@7MV ONE@7MVOP 6NE@7MVOP68 =NE@7MVOP68=W XNE@7MVOP68=WX- ^NE@7MVOP68=WXU aNE@7MVOP68=WXUa_ ^NE@7MVOP69 INE@7MVOP69I= QNE@7MVOP6D INE@7MVOP6D"
"IW UNE@7MVOP6I =NE@7MVOP6I=W XNE@7MVOP6I=WXU _NE@7MVOP6I=WXU_D ^NE@7MVOP6U _NE@7MVOP6U_^ WNE@7MVOP6U_^WX `NE@7MVOP6U_^Wa `NE@7MVOP6U_^Wa`X DNE@7MVOP6U_^Wa`XD= ANE@7MVOP6W =NE@7MVOP6W=D 8NE@7MW ONE@7MWOP INE@7MWOPIV DNE@8 7NE@87. INE@876 9NE@87= 6NE@87=6. 0NE@87=60 /NE@87=65 /NE@87=65/( ANE@87=65/M ."
"NE@87=6M ANE@87=6MA- .NE@87=6MA. ONE@87=6MA0 5NE@87=6MA05. /NE@87=6MA05I ONE@87=6MA: INE@87=6MA:IP BNE@87=6MA:IPBJ 9NE@87=6MAB 5NE@87=6MAB50 /NE@87=6MAB50/& (NE@87=6MAB5O 9NE@87=6MAB5O9. PNE@87=6MAO 9NE@87=6MAV INE@87=6MAW ONE@87=6O WNE@87=6OW0 5NE@87=6OWV MNE@87U MNE@87V ANE@87VA9 INE@87VA: 9NE@87V"
"A:9/ 0NE@87VAJ INE@87W ANE@V PNE@VP6 5NE@VP654 DNE@VP654D< LNE@VP654D<LQ 7NE@VP654D<LQ78 ANE@VP654D<LQ78AM INE@VP657 =NE@VP657=8 ONE@VP657=8OM DNE@VP658 7NE@VP6587. =NE@VP6587I ONE@VP65Q 7NE@VP7 =NE@VP8 7NE@VP876 /NE@VP876/0 .NE@VP87I ONE@VP87IO9 ANE@VP9 ONE@VPI ONE@VPIO7 QNE@VPIOQ WNE@W PNE@WP6 MNE"
"@WP6MU 5NE@WP8 7NE@WP876 .NE@WP879 ONE@WPI QNM @NM@6 7NM@670 ENM@67A ENM@67AE8 /NM@67O =NM@67P ONM@67POX =NM@67V =NM@8 ENM@8E= 7NM@8E=7O ANM@8E=7P ONM@8ED PNM@8EDP7 INM@8EDPO VNM@8EDPOVW INM@8EO PNM@8EOP6 7NM@8EOP< 7NM@8EOPD VNM@O INM@OI6 =NM@OI6=7 ENM@OI6=7E4 0NM@OI6=7EL 5NM@OI6=7EQ 8NM@OI6=8 7NM@O"
"I6=9 PNM@OI6=A ENM@OI6=D <NM@OI7 6NM@OI76. /NM@OI768 PNM@OI768P= /NM@OI768PE WNM@OI768PEW- =NM@OI768PEW-=5 VNM@OI768PEW-=5VA LNM@OI768PEW-=5VALR BNM@OI768PEW-=5VALRBU _NM@OI76E 8NM@OI76E8A PNM@OI76E8AP9 WNM@OI76E8AP9WR =NM@OI8 7NM@OI87A QNM@OI87AQ9 PNM@OI87P ENM@OI87PE/ ANM@OI87PED ANM@OIA PNM@OIAP7"
" 6NM@OIAP9 8NM@OIJ QNM@OIJQA RNM@OIJQARZ PNM@OIJQARZP= 8NM@OIJQARZP=89 7NM@OIJQX PNM@V PNM@VP6 =NM@VP6=I QNM@VP8 7NM@VP876 /NM@VP879 ONM@VP87I ONM@VP87IO9 QNM@VP9 ONU @NU@5 ONU@6 =NU@6=7 ONU@7 8NU@786 =NU@786=E ONU@789 ONU@789OP INU@78I ANU@8 ONU@9 PNU@E ONU@EO7 PNU@EO7P8 6NU@EO7P86A MNU@EOP INU@EOV"
" INU@EOX PN 5N56 =N56=4 MN56=4M@ ON56=4M@OE DN56=4M@OEDK 7N56=4M@OEDK78 <N56=4M@OEDK78<; CN56=4M@OEDK78<C 3N56=4M@OEDK78<C3L ;N56=4M@OEDK78<C3L;+ AN56=4M@OEDK78<L CN56=4M@OEDK78<LCV WN56=4M@OEDL <N56=4M@OEDL<; 7N56=4M@OEDQ 8N56=4M@OEDQ8L KN56=4M@OEDQ8LKP 7N56=4M@OEDQ8P 7N56=4M@OI .N56=4M@OI.7 PN56=4"
"M@OI.7PE AN56=4M@OQ 7N56=4ML VN56=4MLV@ 7N56=4MLV@78 EN56=4MLVO UN56=4MLV_ EN56=4MLV_E8 7N56=4MLV` -N56=4MLV`-$ 3N56=4MLV`-$3O <N56=4MLV`-. EN56=4MLV`-.E< DN56=4MLV`-.ED 7N56=4MLV`-/ EN56=4MLV`-/ED 7N56=4MLV`-/ED7@ UN56=4MLV`-7 EN56=4MLV`-O EN56=4MO PN56=4MOPV UN56=4MP ON56=4MPO@ IN56=8 @N56=8@4 EN5"
"6=8@4E< LN56=8@4E<LO MN56=8@4EO MN56=8@4EOM7 <N56=8@9 PN56=8@9PD <N56=8@9PI 7N56=8@9PI7/ AN56=8@9PI7/AE 0N56=8@9PI7/AE0. &N56=8@D <N56=8@D<4 0N56=8@D<40P ON56=8@D<9 EN56=8@D<9EM ON56=8@D<I 7N56=8@D<I79 PN56=8@D<P EN56=8@D<Q EN56=8@I 7N56=8@I7- .N56=8@I7. -N56=8@I7.-< &N56=8@I7/ .N56=8@I7/.< 9N56=8@I"
"7/.A (N56=8@I7/.A($ 9N56=8@I7/.A(' &N56=8@I7/.A('&E 0N56=8@I7/.A('&E0< 9N56=8@I7/.A(- $N56=8@I7/.A(-$< DN56=8@I7/.A(-$D <N56=8@I7/.A(-$D<' &N56=8@I7/.A(< 9N56=8@I7/.A(<90 'N56=8@I7/.A(<90'- EN56=8@I79 PN56=8@P 7N56=8@P7- ON56=8@Q 7N56=@ PN56=@P4 8N56=@P489 ON56=@P489O7 EN56=@P48A 7N56=@P48I ON56=@P4"
"8Q 7N56=@P8 ON56=@P8O4 7N56=@P8O47. MN56=@P8O47.MD /N56=@P8O47E MN56=@P8O47EMV <N56=@P8O47M EN56=@P8O< 4N56=@P8OD <N56=@P8OD<W VN56=@P8OM EN56=@P8OME4 7N56=@P8OME47D VN56=@P8OME< DN56=@P8OMED <N56=@P8OMED<3 7N56=@P8OMED<370 /N56=@P8OMED<I 7N56=@P8OMED<W XN56=@P8OMEW VN56=@P8OMEWVD <N56=@P8OMEWVU .N5"
"6=@P8OMEWVU.Q XN56=@P8OMEWV_ 7N56=@P8OMEWV` UN56=@P8OW VN56=@P8OWV4 7N56=@P8OWV47. /N56=@P8OWV47/ AN56=@P8OWV47/A9 'N56=@P8OWV47/A9'. 0N56=@P8OWV47Q aN56=@P8OWV< EN56=@P8OWV<EM DN56=@P8OWV<EMD_ 7N56=@P8OWVD EN56=@P8OWVDEM UN56=@P8OWVE MN56=@P8OWVEM< DN56=@P8OWVEMD <N56=@P8OWVEMD<U aN56=@P8OWVEMD<_ ."
"N56=@P8OWVEMI aN56=@P8OWVEMIa< .N56=@P8OWVEMIa<./ 7N56=@P8OWVEMIaX QN56=@P8OWVEMIaXQ< 7N56=@P8OWVEMIaXQ<7D LN56=@P8OWVEMIaXQ<7DL^ _N56=@P8OWVEMIaXQ<7DL^_` ]N56=@P8OWVEMIaXQ<7DL^_`]U KN56=@P8OWVEMIaXQ<7DL^_`]UK; bN56=@P8OWVEMIaXQ^ _N56=@P8OWVEMQ aN56=@P8OWVEMQa< IN56=@P8OWVEMQa<I4 7N56=@P8OWVEMQa<IR "
".N56=@P8OWVEMQa<IR./ 4N56=@P8OWVEMQa<IR./4D 7N56=@P8OWVEMQa<IR./4D70 'N56=@P8OWVEMQa<IR./4D70'$ &N56=@P8OWVEMQa<IR./4D70'$&( AN56=@P8OWVEMQa<I` XN56=@P8OWVEMQaU ^N56=@P8OWVEMQaU^< 4N56=@P8OWVEMQaU^<4. DN56=@P8OWVEMQaU^<4.D; 3N56=@P8OWVEMQaU^<4.D;3+ 7N56=@P8OWVEMQaU^<4D ;N56=@P8OWVEMQaU^<4D;- 3N56=@P"
"8OWVEMQaU^<4D;3 7N56=@P8OWVEMQaU^<4D;37. RN56=@P8OWVEMQaU^<4D;37.RK CN56=@P8OWVEMQaU^<4D;37.RKCX -N56=@P8OWVEMQaU^<4D;37.RKCX-L bN56=@P8OWVEMQaU^<4D;37.RKCX-Lb/ &N56=@P8OWVEMQaU^<4D;37.RKCX-Lb/&' ,N56=@P8OWVEMQaU^<4D;37.RKCX-Lb/&',` _N56=@P8OWVEMQaU^<4D;37.RX +N56=@P8OWVEMQaU^<4D;37.RX+K CN56=@P8OWV"
"EMQaU^<4D;37.RX+KC! bN56=@P8OWVEMQaU^<4D;37.RX+KC!bT LN56=@P8OWVEMQaU^<4D;37.RX+KC!bTLS IN56=@P8OWVEMQaU^<4D;37.RX+KC!bTLSIJ BN56=@P8OWVEMQaU^<4D;37.RX+KC!bTLSIJB_ `N56=@P8OWVEMQaU^<4D;37.RX+KC!bTLSIJB_`Z cN56=@P8OWVEMQaU^<4D;37.RX+KC!bTLSIJB_`ZcA 0N56=@P8OWVEMQaU^<4D;37.RX+KC!bTLSIJB_`ZcA0/ :N56=@P"
"8OWVEMQaU^<4D;37.RX+KC!bTLSIJB_`ZcA0/:Y 'N56=@P8OWVEMQaU^<4D;37K CN56=@P8OWVEMQaU^<4D;37KC0 RN56=@P8OWVEMQaU^<4D;37KC0RX AN56=@P8OWVEMQaU^<4D;37KC0RXA_ bN56=@P8OWVEMQaU^<4D;37KC0RXA_b] YN56=@P8OWVEMQaU^<4D;37KC0RXA_b]Y9 `N56=@P8OWVEMQaU^<4D;37KC0RXA_b]Y9`c IN56=@P8OWVEMQaU^<4D;37KC0RXA_b]Y9`cIZ JN56"
"=@P8OWVEMQaU^<4D;37KC0RXA_b]Y9`cIZJB ,N56=@P8OWVEMQaU^D <N56=@P8OWVEMQaU^D<_ `N56=@P8OWVEMQaU^D<_`L 7N56=@P8OWVEMQaU^L DN56=@P8OWVEMQaU^LD7 <N56=@P8OWVEMQaU^LD7<C ;N56=@P8OWVEMQaU^LD7<C;3 KN56=@P8OWVEMQaU^LD7<C;K SN56=@P8OWVEMQaU^LD7<C;KS3 +N56=@P8OWVEMQaU^LD7<C;KS3+4 RN56=@P8OWVEMQaU^LD7<C;KS3+4RX "
"bN56=@P8OWVEMQaU^LD7<X CN56=@P8OWVEMQaU^LD7<XC4 IN56=@P8OWVEMQaU^LD7<XC4IR BN56=@P8OWVEMQaU^X 7N56=@P8OWVEMQaU^X7- .N56=@P8OWVEMQaU^X7D LN56=@P8OWVEMQaU^X7DL/ 4N56=@P8OWVEMQaU^X7DL/4< ;N56=@P8OWVEMQaX IN56=@P8OWVEMQaXIR BN56=@P8OWVEMQa` _N56=@P8OWVEMU .N56=@P8OWVEMU.L 7N56=@P8OWVEMU.L7D `N56=@P8OWVE"
"MX aN56=@P8OWVEMXa< QN56=@P8OWVEM_ .N56=@P8OWVEM_.A XN56=@P8OWVEM_.AXI 7N56=@P8OWVEM_.AXQ IN56=@P8OWVEM_.L 7N56=@P8OWVEM_.L7D UN56=@P8OWVEM_.L7DUI aN56=@P8OWVEM_.L7DUIa< 4N56=@P8OWVI aN56=@P8OWVIaE MN56=@P8OWVIaEM< .N56=@P8OWVIaEM<./ 7N56=@P8OWVIaEM<.D LN56=@P8OWVIaEMU .N56=@P8OWVIaEMU.L 7N56=@P8OWV"
"IaEMU.L7` _N56=@P8OWVIaEMX QN56=@P8OWVIaEMXQ< 4N56=@P8OWVIaEMXQ` _N56=@P8OWVIaEMXQ`_< .N56=@P8OWVIaEMXQ`_<.D 7N56=@P8OWVIaEMXQ`_<.D7^ ]N56=@P8OWVIaEMXQ`_<.D7^]R UN56=@P8OWVIa^ _N56=@P8OWVM EN56=@P8OWVMED <N56=@P8OWVMED<` ^N56=@P8OWVMED<`^U aN56=@P8OWVMEI QN56=@P8OWVMEIQU .N56=@P8OWVMEU aN56=@P8OWVME"
"` ^N56=@P8OWVQ IN56=@P8OWVQIA /N56=@P8OWVQIE LN56=@P8OWVQIEL< MN56=@P8OWVQIELD MN56=@P8OWVQIELDM< UN56=@P8OWVQIELM DN56=@P8OWVQIELMD< UN56=@P8OWVQIELMD<U_ ^N56=@P8OWVQIELMD<U_^] 9N56=@P8OWVQIELMDC UN56=@P8OWVQIELMDCU7 KN56=@P8OWVQIELMDJ .N56=@P8OWVQIELMDJ.X aN56=@P8OWVQIELMDJ.Xa< 7N56=@P8OWVQIELMDJ."
"Xa_ YN56=@P8OWVQIELMDJ.Xa_Y< `N56=@P8OWVQIELMDJ.Xa_Y<`^ UN56=@P8OWVQIELMDJ.Xa_Y<`^Uc ]N56=@P8OWVQIELMDJ.Xa_Y<`^Uc]/ CN56=@P8OWVQIELMDJ.Xa_YC 7N56=@P8OWVQIELMDJ.Xa_YC7< `N56=@P8OWVQIELMDJ.Xa_YC7<`^ UN56=@P8OWVQIELMDU aN56=@P8OWVQIELMDUa< -N56=@P8OWVQIELMDUaR `N56=@P8OWVQIELMDUaR`< .N56=@P8OWVQIELMDUa"
"R`_ XN56=@P8OWVQIELMDUaR`_XA ^N56=@P8OWVQIELMDUa` XN56=@P8OWVQIELMDUa`X4 7N56=@P8OWVQIELMDUa`XR .N56=@P8OWVQIELMDUa`XR.A ^N56=@P8OWVQIELMDUa`XR.A^4 <N56=@P8OWVQIELMDUa`XR.A^4<C _N56=@P8OWVQIELMDUa`XR.A^4<C_/ 9N56=@P8OWVQIELMDUa`XR.A^C _N56=@P8OWVQIELMDUa`XR.A^C_7 KN56=@P8OWVQIELMDUa`XR.A^b JN56=@P8O"
"WVQIELMDUa`XR.A^bJ- $N56=@P8OWVQIELMDUa`XR.A^bJ4 <N56=@P8OWVQIELMDUa`XR.A^bJ4<7 :N56=@P8OWVQIELMDUa`XR.A^bJ4<7:Y _N56=@P8OWVQIELMDUa`XR.A^bJB 7N56=@P8OWVQIELMDUa`XR.A^bJB7/ (N56=@P8OWVQIELMDUa`XR.A^bJB7/($ &N56=@P8OWVQIELMDUa`XR.A^bJB7/($&- 4N56=@P8OWVQIELMDUa`XR.A^bJB7/($&-4< CN56=@P8OWVQIELMDUa`XR"
".A^bJB7/($&-4<C, 9N56=@P8OWVQIELMDUa`XR.A^bJB7/($&-4<C,9' 0N56=@P8OWVQIELMDUa`XR.K _N56=@P8OWVQIELMDUa`XR.K_^ 7N56=@P8OWVQIELMDUa`XR.K_^7b AN56=@P8OWVQIELMDUa`XR.K_^7bA: 9N56=@P8OWVQIELMDUa`XR.b 7N56=@P8OWVQIELMDUa`XR.b7A ]N56=@P8OWVQIELMDUa`XR.b7A]Y 0N56=@P8OWVQIELMDX UN56=@P8OWVQIELMDXU^ .N56=@P8O"
"WVQIELMDXU^.< RN56=@P8OWVQIELMDXU^.K CN56=@P8OWVQIELMDXU^.KC; 9N56=@P8OWVQIELMDXU^.KC;94 0N56=@P8OWVQIELMDXU^.KC;940< `N56=@P8OWVQIELMDXU^.KC;940<`7 /N56=@P8OWVQIELMDXU^.KC;940<`7/S -N56=@P8OWVQIELMDXU^.KC;9B 0N56=@P8OWVQIELMDXU^.KC;9B04 `N56=@P8OWVQIELMDXU^.KC;9B04`A aN56=@P8OWVQIELMDXU^.KC;9B04`Aa"
"< _N56=@P8OWVQIELMDXU^.KC;9B04`Aa<_7 /N56=@P8OWVQIELMDXU^.KC;9B04`Aa<_7/b TN56=@P8OWVQIELMD_ UN56=@P8OWVQIELMD_UX RN56=@P8OWVQIELX MN56=@P8OWVQIEL_ ^N56=@P8OWVQIEL_^M DN56=@P8OWVQIEL_^MDU `N56=@P8OWVQIEL_^MDU`J /N56=@P8OWVQIEL_^MDU`a bN56=@P8OWVQIEL_^MDU`abX RN56=@P8OWVQIEL_^MDU`abXRT 9N56=@P8OWVQIE"
"L_^MDU`abXRT9A :N56=@P8OWVQIEL_^MDU`abXRT9A:- 0N56=@P8OWVQIEL_^MDU`abXRT9A:-0S 7N56=@P8OWVQIEL_^U `N56=@P8OWVQIEL_^U`M DN56=@P8OWVQIEL_^U`MDJ .N56=@P8OWVQIEL_^U`MDJ.4 7N56=@P8OWVQIEL_^U`MDR .N56=@P8OWVQIEL_^U`MDR.4 <N56=@P8OWVQIEL_^U`MDR.4<7 ]N56=@P8OWVQIEL_^U`R ]N56=@P8OWVQIEL_^U`R]M DN56=@P8OWVQIE"
"L_^U`R]MDA .N56=@P8OWVQIEL_^U`a bN56=@P8OWVQIR XN56=@P8OWVQIX /N56=@P8OWVQIX/& 7N56=@P8OWVQIX/- .N56=@P8OWVQIX/A aN56=@P8OWVQIX/Aa_ 7N56=@P8OWVQIX/Aa_7^ `N56=@P8OWVQIX/Aa_7^`b UN56=@P8OWVQIX/E 7N56=@P8OWVQIX/E7. 0N56=@P8OWVQIX/E79 AN56=@P8OWVQIX/E79AJ aN56=@P8OWVQIX/E79AJa. :N56=@P8OWVQIX/E79AJa.:B "
"UN56=@P8OWVQIX/E79AJa.:BU' RN56=@P8OWVQIX/E79AJa.:BU2 <N56=@P8OWVQIX/E79AJa.:BU2<' MN56=@P8OWVQIX/E79AJa.:BU2<'M0 _N56=@P8OWVQIX/E79AJa.:BU2<'M0_4 ZN56=@P8OWVQIX/E79AJa.:BU2<( MN56=@P8OWVQIX/E79AJa.:BU2<(M0 -N56=@P8OWVQIX/E79AJa.:BU2<(M0-^ _N56=@P8OWVQIX/E79AJa.:BU2<D MN56=@P8OWVQIX/E79AJa.:BU2<DM^ "
"0N56=@P8OWVQIX/E79AJa.:BU2<DM^0` $N56=@P8OWVQIX/E79AJa.:BU2<M 0N56=@P8OWVQIX/E79AJa.:BU2<M0_ $N56=@P8OWVQIX/E79AJa.:BUD <N56=@P8OWVQIX/E79AJa.:BU^ <N56=@P8OWVQIX/E79AJa.:BU` _N56=@P8OWVQIX/E79AJa.:BU`_^ ]N56=@P8OWVQIX/E79AJa` _N56=@P8OWVQIX/E7A :N56=@P8OWVQIX/E7^ 9N56=@P8OWVQIX/J 7N56=@P8OWVQIX/J7^ "
"`N56=@P8OWVQIX/J7^`_ ]N56=@P8OWVQIX/R 7N56=@P8OWVQIX/R7& AN56=@P8OWVQIX/R7^ 9N56=@P8OWVQIX/R7^90 AN56=@P8OWVQIX/R7^90A: JN56=@P8OWVQIX/R7^90A:JB `N56=@P8OWVQIX/R7^90A:JB`_ ]N56=@P8OWVQIX/R7^90A:JB`_]U (N56=@P8OWVQIX/R7^90A:JB`_]U(< 4N56=@P8OWVQIX/R7^90A:JB`_]U(<4M ;N56=@P8OWVQIX/R7^90A:JB`_]U(<4M;- "
".N56=@P8OWVQIX/R7^90A:JB`_]U(<4M;-.$ DN56=@P8OWVQIX/R7^9M UN56=@P8OWVQIX/R7^9MUE `N56=@P8OWVQIX/R7^9MUE`. <N56=@P8OWVQIX/R7^9MUE`.<L _N56=@P8OWVQIX/^ 7N56=@P8OWVQIX/^7A 0N56=@P8OWVQIX/^7A0M UN56=@P8OWVQIX/^7A0MUE BN56=@P8OWVQIX/^7A0MUEB4 <N56=@P8OWVQIX/^7A0MUEB4<3 aN56=@P8OWVQIX/^7A0MUEB4<3a` JN56=@"
"P8OWVQIX/^7A0MUEB4<3a`J] DN56=@P8OWVQIX/^7A0MUEB4<3a`J]D; KN56=@P8OWVQIX/^7B 9N56=@P8OWVQIX/^7B9M UN56=@P8OWVQIX/^7B9MUE `N56=@P8OWVQIX/^7J 9N56=@P8OWVQIX/^7J9- _N56=@P8OWVQIX/^7J9-_A ]N56=@P8OWVQIX/^7J9M UN56=@P8OWVQIX/^7J9MUE `N56=@P8OWVQIX/^7J9MUE`_ ]N56=@P8OWVQIX/^7M EN56=@P8OWVX aN56=@P8OWVXaE "
"MN56=@P8OX VN56=@P8OXVE MN56=@P8OXVEM^ `N56=@P8OXVW aN56=@P8OXVWaE MN56=@P8OXVWaEM< QN56=@P8OXVWaEM<Q_ UN56=@P< 8N56=@P<8I 7N56=@P<8O 7N56=@P<8O7. WN56=@P<8O7.WA IN56=@P<8O7/ -N56=@P<8O7/-. WN56=@P<8O70 .N56=@P<8O70./ -N56=@P<8O70./-4 (N56=@P<8O70./-4(' &N56=@P<8O70./-4('&$ EN56=@P<8O70./-4('&$E) DN"
"56=@P<8O70.9 WN56=@P<8O70.I QN56=@P<8O70.IQ9 WN56=@P<8O70.IQ9W/ 'N56=@P<8O70.IQA BN56=@P<8O70.IQABJ RN56=@P<8O70.IQABJR9 (N56=@P<8O70.IQABJR9($ &N56=@P<8O70.IQABJR9($&- :N56=@P<8O70.IQABJR9($&-:/ #N56=@P<8O70.IQABJR9($&-:/#2 'N56=@P<8O70.IQABJR9($&-:/#2'Z CN56=@P<8O70.IQABJR9(- :N56=@P<8O79 IN56=@P<"
"8O7I QN56=@P<8O7IQ- EN56=@P<8O7IQ-EA XN56=@P<8O7IQ-EAXW MN56=@P<8O7IQ-EAXWMV BN56=@P<8O7IQ-EAXWMVBJ RN56=@P<8O7IQ-EAXWMVBJR. _N56=@P<8O7IQ9 JN56=@P<8O7IQ9JA :N56=@P<8O7IQ9JA:- EN56=@P<8O7IQ9JA:X BN56=@P<8O7IQA BN56=@P<8O7IQAB0 DN56=@P<8O7IQABJ RN56=@P<8O7IQABJR9 :N56=@P<8O7IQABJR9:- EN56=@P<8O7IQABJ"
"R9:-E. /N56=@P<8O7IQABJR9:-E./( 'N56=@P<8O7IQABJR9:-E./('& 0N56=@P<8O7IQABJR9:-E./('&0M DN56=@P<8O7IQABJR9:-E./('X &N56=@P<8O7IQABJR9:-E./M $N56=@P<8O7IQABJR9:-E0 /N56=@P<8O7IQABJR9:-EM .N56=@P<8O7IQABJR9:-EM.& 4N56=@P<8O7IQABJR9:-EM.&4/ 0N56=@P<8O7IQABJR9:-EM.&4/0) CN56=@P<8O7IQABJR9:X aN56=@P<8O7I"
"QABJR9:Xa- EN56=@P<8O7IQABJR9:Xa-EM .N56=@P<8O7IQABJR9:Xa-EM.& WN56=@P<8O7IQABJR9:Xa-EM.&WV ^N56=@P<8O7IQABJR9:Xa-EM.&WV^/ UN56=@P<8O7IQABJR9:Xa-EM.&WV^/UY LN56=@P<8O7IQABJR9:Xa-EM.&WV^/UYLK DN56=@P<8O7IQABJR9:Xa-EM.&WV^/UYLKD0 3N56=@P<8O7IQABJR9:Xa-EM.&WV^/UYLKD03; 4N56=@P<8O7IQABJR9:Xa-EM.&WV^Y UN"
"56=@P<8O7IQABJR9:Xa. EN56=@P<8O7IQABJR9:Xa.EM DN56=@P<8O7IQABJR9:Xa0 'N56=@P<8O7IQABJRX 9N56=@P<8O7IQABJRX9/ .N56=@P<8O7IQABJRX9: 2N56=@P<8O7IQABX JN56=@P<8O7IQW RN56=@P<8O7IQWR- XN56=@P<8O7IQWR-X. EN56=@P<8O7IQWR-X.E/ MN56=@P<8O7IQWR-X.E/MV 0N56=@P<8O7IQWR-X.EA MN56=@P<8O7IQWR-X.EM &N56=@P<8O7IQWRA"
" JN56=@P<8O7IQWRAJ/ XN56=@P<8O7IQWRAJ: 9N56=@P<8O7IQWRAJ:9/ .N56=@P<8O7IQWRAJ:9/.0 -N56=@P<8O7IQWRAJ:90 /N56=@P<8O7IQWRAJ:90/' .N56=@P<8O7IQWRB JN56=@P<8O7IQWRBJZ XN56=@P<8O7IQWRBJZXA _N56=@PD <N56=@PD<8 7N56=@PD<I ON56=@PI ON56=@PIO4 QN56=@PIO< AN56=@PIO<AE 8N56=@PIO<AE87 /N56=@PIOE 8N56=@PIOE84 .N"
"56=@PIOE84./ -N56=@PIOE87 AN56=@PIOE87A9 .N56=@PIOE87A9.< /N56=@PIOE87A9.</0 RN56=@PIOE87AD 0N56=@PIOE87AD0/ 'N56=@PIOE87AD0/'. 9N56=@PIOE87AD0/'.9B :N56=@PIOE87AD0/'.9B:2 <N56=@PIOE87AQ LN56=@PIOE87AW /N56=@PIOE87AW/. MN56=@PIOE87AX WN56=@PIOE87AXWD /N56=@PIOE87AXWD/- .N56=@PIOE87AXWD/B 9N56=@PIOE8"
"7AXWD/B9. :N56=@PIOE87AXWD/Q RN56=@PIOE87AXWD/QR4 UN56=@PIOE87AXWD/QRJ BN56=@PIOE87AXWD/_ .N56=@PIOE87AXWQ LN56=@PIOE87AXWQL0 .N56=@PIOE87AXWQL0.D /N56=@PIOE87AXWQL0.` /N56=@PIOE87AXWQL: MN56=@PIOE87AXWQL:M/ UN56=@PIOE87AXWQLD /N56=@PIOE87AXW` aN56=@PIOE87AXW`ab QN56=@PIOE87AXWa RN56=@PIOE89 7N56=@P"
"IOE897/ AN56=@PIOE897/A. BN56=@PIOE897< :N56=@PIOE8D 7N56=@PIOE8D7A BN56=@PIOE8D7ABJ QN56=@PIOE8D7ABJQ: <N56=@PIOE8D7ABR :N56=@PIOE8D7ABR:9 <N56=@PIOE8D7ABR:9</ -N56=@PIOE8D7ABR:9</-. 4N56=@PIOE8D7ABR:9</-.4; QN56=@PIOE8D7ABW :N56=@PIOE8D7ABW:/ .N56=@PIOE8D7ABW:/.0 -N56=@PIOE8D7ABW:/.0-9 <N56=@PIOE8"
"D7ABW:Q ZN56=@PIOE8D7ABW:QZ- 4N56=@PIOE8D7ABW:QZ-4J RN56=@PIOE8D7ABW:QZ-4JR. <N56=@PIOE8D7ABW:R JN56=@PIOE8D7ABW:RJ2 `N56=@PIOE8D7ABW:RJ2`_ 9N56=@PIOE8D7ABW:RJ2`_9a XN56=@PIOE8D7ABW:RJ2`_9aXV <N56=@PIOE8D7X WN56=@PIOE8W 7N56=@PIOE8W7/ .N56=@PIOE8W7/.Q LN56=@PIOE8W7A BN56=@PIOE8W7AB/ .N56=@PIOE8X WN5"
"6=@PIOE8XW7 AN56=@PIOE8XW7A< QN56=@PIOE8XW7AD /N56=@PIOE8XW7AD/. 9N56=@PIOE8XW7AD/.9B :N56=@PIOE8XW7AQ RN56=@PIOE8XW7Aa `N56=@PIOE8XW9 AN56=@PIOE8XW9Aa 7N56=@PIOE8XWD VN56=@PIOE8XWDV7 AN56=@PIOE8XWQ 7N56=@PIOE8XWQ7/ .N56=@PIOE8XWV 7N56=@PIOE8XWV7A QN56=@PIOE8XWV7AQ9 RN56=@PIOE8XWV7Q JN56=@PIOE8XWV7Q"
"J9 AN56=@PIOE8XWa 7N56=@PIOE8XWa7Q JN56=@PIOE8XWa7QJ/ .N56=@PIOE8XWa7QJ/.< MN56=@PIOE8XWa7QJ/.<M9 AN56=@PIOE8XWa7QJ/.<M9A0 VN56=@PIOE8XWa7QJ/.<M9A0VR 'N56=@PIOE8XWa7QJ9 AN56=@PIOE8XWa7QJB :N56=@PIOE8XWa7QJB:A RN56=@PIOE8XWa7QJB:AR9 LN56=@PIOE8XWa7QJB:AR9L0 /N56=@PIOV 7N56=@PIOV7/ QN56=@PIOV7E 8N56=@"
"PIOV7E8/ 0N56=@PIOV7E8A QN56=@PIOV7E8AQ9 RN56=@PIOV7E8AQ9RJ WN56=@PIOV7E8AQ9RJW/ 0N56=@PIOV7E8AQW :N56=@PIOV7E8Q JN56=@PIOV7E8QJ/ .N56=@PIOV7E8QJ/.9 AN56=@PIOV7E8QJ/.9A: BN56=@PIOV7E8QJ4 <N56=@PIOV7E8QJ9 AN56=@PIOV7E8QJ9A: BN56=@PIOV7E8QJ9A:BR LN56=@PIOV7E8W JN56=@PIOV7E8WJ. /N56=@PIOV7E8WJ/ MN56=@P"
"IOV7E8WJ/M. -N56=@PIOV7E8WJ/M.-4 <N56=@PIOV7E8WJ/M.-4<Q ;N56=@PIOV7E8WJ/M.-4<Q;$ (N56=@PIOV7E8WJ/M.-4<Q;$(R 0N56=@PIOV7E8WJ/M.-4<Q;3 +N56=@PIOV7E8WJ/M.-4<Q;9 'N56=@PIOV7E8WJ/M.-4<Q;9'$ AN56=@PIOV7E8WJ/M.-4<Q;9'$AR (N56=@PIOV7E8WJ/M.-4<Q;9'$AR(0 :N56=@PIOV7E8WJ/M.-4<Q;9'$AR(0:B ZN56=@PIOV7E8WJ/M.-4<Q"
";R AN56=@PIOV7E8WJ/M.-4<Q;RA: UN56=@PIOV7E8WJ/M.-4<Q;RA:UB _N56=@PIOV7E8WJ/M.-4<Q;RA:UB_L &N56=@PIOV7E8WJ/M.-4<Q;RA:UB_L&C aN56=@PIOV7E8WJ/M.-4<Q;RA:UB_L&Ca9 DN56=@PIOV7E8WJ/M.-4<Q;RA:UB_L&Ca9D3 XN56=@PIOV7E8WJ/M.-4<Q;RA:UB_L&Ca9D3X' (N56=@PIOV7E8WJ/M.-4<Q;RA:UB_L&Ca9D3X'(0 SN56=@PIOV7E8WJ/M.-4<Q;RA"
":UB_L&Ca9D3X'(0S$ #N56=@PIOV7E8WJ/M.-4<Q;RA:UB_L&Ca9D3X'(0S$#Y )N56=@PIOV7E8WJ/M.-4<Q;RA:UB_L&Ca9D3X'(0S$#Y)K +N56=@PIOV7E8WJ/M.-4<Q;RAB UN56=@PIOV7E8WJ/M.-4<Q;RABU$ _N56=@PIOV7E8WJ/M.-4<Q;RABU$_: 0N56=@PIOV7E8WJ/M.-4<Q;RABU$_:09 &N56=@PIOV7E8WJ/M.-4<Q;RABU: _N56=@PIOV7E8WJ/M.-4<Q;RABU:_D LN56=@PIOV"
"7E8WJ/M.-4<Q;RABU:_DL9 &N56=@PIOV7E8WJ/M.-4<Q;RABU:_DL9&' $N56=@PIOV7E8WJ/M.-4<Q;RABU:_DL9&'$3 (N56=@PIOV7E8WJ/M.-9 (N56=@PIOV7E8WJ/M.-9(' &N56=@PIOV7E8WJ/M.-9('&A XN56=@PIOV7E8WJ/M.-9(A 0N56=@PIOV7E8WJ/M9 ^N56=@PIOV7E8WJ/M9^_ .N56=@PIOV7E8WJ/M9^_.L QN56=@PIOV7E8WJ/M9^_.LQ< (N56=@PIOV7E8WJ/M9^_.LQ<("
"X DN56=@PIOV7E8WJ/M9^_.LQ<(XDB 3N56=@PIOV7E8WJ/M9^_.LQ<(XDB30 UN56=@PIOV7E8WJ/MA UN56=@PIOV7E8WJ/MR ZN56=@PIOV7E8WJ4 <N56=@PIOV7E8WJ4<- .N56=@PIOV7E8WJ4<3 LN56=@PIOV7E8WJ4<3LD MN56=@PIOV7E8WJ4<3LDMK UN56=@PIOV7E8WJ4<3LDMKU_ ]N56=@PIOV7E8WJA BN56=@PIOV7E8WJAB: 2N56=@PIOV7E8WJAB:29 0N56=@PIOV7E8WJB :N"
"56=@PIOV7E8WJB:A LN56=@PIOV7E8WJB:ALD <N56=@PIOV7E8WJB:ALD<9 MN56=@PIOV7Q UN56=@PIOV7QUE MN56=@PIOV7QUEML 9N56=@PIOV7QUEML9D <N56=@PIOV7QUM EN56=@PIOV7QUMEL _N56=@PIOV7QUMEL_D <N56=@PIOV7QUMEL_D<8 WN56=@PIOV7W 9N56=@PIOV7W98 AN56=@PIOV7W98AE 0N56=@PIOV7W9E LN56=@PIOV7W9EL8 AN56=@PIOV7W9EL8A/ .N56=@P"
"IOV7W9EL8A/.B (N56=@PIOV7W9EL< ;N56=@PIOV7W9EL<;0 /N56=@PIOV7W9EL<;8 AN56=@PIOV7W9EL<;8A/ RN56=@PIOV7W9EL<;8A/R. MN56=@PIOV7W9ELD KN56=@PIOV7W9ELDK4 <N56=@PIOV7W9ELDK8 AN56=@PIOW UN56=@PIOWUE LN56=@PIOWUEL8 7N56=@PIOX VN56=@PIOXVE 9N56=@PIOXVE97 MN56=@PIOXVE97M< .N56=@PIOXVE97M<./ 4N56=@PIOXVE97M<./"
"4- 8N56=@PIOXVE97M<./4D -N56=@PIOXVE97M<./4D-8 (N56=@PIOXVE97M<./4D-8(& CN56=@PIOXVE97M<.D 8N56=@PIOXVE97M<.D8W AN56=@PIOXVE97M<.D8WAL RN56=@PIOXVE97MD .N56=@PIOXVE97MD./ -N56=@PIOXVE97MW aN56=@PIOXVE97MWa< 4N56=@PIOXVE97MWa<4D QN56=@PIOXVE97MWa<4DQ. `N56=@PIOXVE97MWa<4DQ.`U BN56=@PIOXVE97MWa^ _N56="
"@PIOXVE98 MN56=@PIOXVE98M< .N56=@PIOXVE98M<.D 7N56=@PIOXVE98M<.D7/ 0N56=@PIOXVE98M<.D7/04 LN56=@PIOXVE98MU WN56=@PIOXVE98MUW< QN56=@PIOXVE98MUW<Q` _N56=@PIOXVE98MUW<Q`_^ aN56=@PIOXVE98MUW<Q`_^ab BN56=@PIOXVE98MUW<Q`_^abB4 7N56=@PIOXVE98MUW<Q`_^abB47A :N56=@PIOXVE98MUW^ QN56=@PIOXVE98MUW^Q_ YN56=@PIO"
"XVE98MUW^Q_Y4 BN56=@PIOXVE98MW aN56=@PIOXVE98MWa< QN56=@PIOXVE98MWa<Q4 `N56=@PIOXVE98MWa<Q4`D BN56=@PIOXVE98MWa<Q4`DBR 7N56=@PIOXVE98MWa<Q4`DBR72 0N56=@PIOXVE98MWa<Q^ _N56=@PIOXVE98MWa<Q` _N56=@PIOXVE98MWa<Q`_D 7N56=@PIOXVE98MWa<Q`_D7^ ]N56=@PIOXVE98MWaL DN56=@PIOXVE98MWaLD7 <N56=@PIOXVE98MWa^ _N56="
"@PIOXVE98MWa` _N56=@PIOXVE9D <N56=@PIOXVE9M 7N56=@PIOXVE9M78 WN56=@PIOXVE9M78W: AN56=@PIOXVE9W .N56=@PIOXVE9W.8 7N56=@PIOXVE9W.87< QN56=@PIOXVM 9N56=@PIOXVM97 /N56=@PIOXVM97/E 8N56=@PIOXVM97/E80 WN56=@PIOXVM9E 7N56=@PIOXVU MN56=@PIOXVUM7 QN56=@PIOXVUM_ ^N56=@PQ 7N56=@PQ78 AN56=D <N56=D<4 MN56=D<8 @N"
"56=D<8@9 PN56=D<8@9PI 7N56=D<8@I 7N56=D<8@I79 PN56=D<8@Q 7N56=D<8@Q7- EN56=D<@ EN56=D<@E4 VN56=D<@E4VM LN56=D<@E4VMLO PN56=D<@E4VMLOPI CN56=D<@E4VMLOPICU ;N56=D<@E4VMLOPICU;3 +N56=D<@E; 4N56=D<@E;4. KN56=D<@E;4.K3 7N56=D<@EP 7N56=D<@EP74 ;N56=I 7N56=I7. EN56=I7.E@ MN56=I78 9N56=I789/ 0N56=Q @N56=Q@8"
" /N56=Q@8/. 7N56=Q@8/.7- ON56=Q@8/.7-OM EN58 @N58@6 =N58@6=4 EN58@6=4EO MN58@6=9 7N58@6=97/ ON58@6=D <N58@6=D<I 7N58@6=D<I79 AN58@6=D<Q 7N58@6=I 7N58@6=I7/ .N58@6=I7/.A (N58@6=I79 AN58@6=I79AJ 0N58@6=I79AJ0' MN58@6=Q 7N58@9 ON58@9OI PN58@9OP IN58@9OX AN58@I PN58@IP7 MN58@IP7MO 6N58@P MN5@ PN5@P6 8N5"
"@P684 EN5@P684E< DN5@P684E<DA 9N5@P684E<DA9O =N5@P684EA 9N5@P684EA9M 7N5@P684EA9O WN5@P684EA9OW= 7N5@P684EA9OW=7M DN5@P684EA9OW=7MDV <N5@P684EA9OW=7MDV<K CN5@P684EL DN5@P684EQ =N5@P689 ON5@P68A 9N5@P68A9/ 0N5@P68A9/0I 7N5@P68A9/0I7= &N5@P68A9I JN5@P68A9IJ7 BN5@P68A9O BN5@P68A9OB0 'N5@P68A9OB0'J RN5@"
"P68A9OB4 UN5@P68A9OB4UM IN5@P68A9OB4UMIJ RN5@P68A9OB4UMIJRX WN5@P68A9OBQ UN5@P68A9OBQU: JN5@P68A9OBQUM IN5@P68A9OBX 7N5@P68A9OBX7. WN5@P68A9OBX7.Wa -N5@P68A9OBX7/ .N5@P68A9OBX7/.= MN5@P68A9OBX7/.=ME -N5@P68A9Q BN5@P68A9QBO UN5@P68I ON5@P68IO7 QN5@P68IOA 7N5@P68Q 7N5@P68Q7O =N5@P8 6N5@P86. =N5@P86= 7"
"N5@P86=7- 4N5@P86=7-40 /N5@P86=7. /N5@P86=7/ AN5@P86=7/A- MN5@P86=7/A-M0 VN5@P86=7/A-M0VJ :N5@P86=7/A-M0VJ:I 9N5@P86=7/A-M0VJ:I9X .N5@P86=7/A-M0VJ:I9X.$ (N5@P86=7/A-M0VJ:I9X.$(& QN5@P86=7/A-M0VJ:I9X.$(&QR )N5@P86=7/A-M0VJ:I9X.$(&QR)E 1N5@P86=7/A-M0VJ:I9X.$(&QR)E1W 'N5@P86=7/A-M0VJ:I9X.$(&QR)E1W'_ UN"
"5@P86=7/A-M0VJ:I9X.$(&QR)E1W'_UD ON5@P86=7/A0 .N5@P86=7/A0.& -N5@P86=7/A0.O -N5@P86=7/A0.O-4 WN5@P86=7/A0.O-4W9 IN5@P86=7/A0.O-9 IN5@P86=7/A0.O-9IB WN5@P86=7/AO .N5@P86=7/AO.0 -N5@P86=7/AO.0-4 WN5@P86=7/AO.I <N5@P86=7/AO.I<Q -N5@P86=7/AO.I<Q-: 'N5@P86=7/AO.I<Q-:'9 BN5@P86=7/AO.I<Q-:'9BJ 0N5@P86=70 A"
"N5@P86=70AO .N5@P86=7I ON5@P86I ON5@P86IOV WN5@P86IOVWQ 9N5@P86IOVWa `N5@P86IOW 9N5@P86IOW9: EN5@P86IOW9V 7N5@P86X AN5@P86XA9 ON5@P= 6N5@P=6- MN5@P=6. 8N5@P=68 7N5@P=687/ AN5@P=687/A0 .N5@P=687/A0.O -N5@P=687/AO .N5@P=687/AO.0 -N5@P=687/AO.0-4 'N5@P=687/AO.0-4'$ WN5@P=6870 /N5@P=6870/( AN5@P=6870/(A"
"O .N5@P=6I ON5@P=6IOW 9N5@PI ON5@PIO6 AN5@PIO8 7N5@PIO87= MN5@PIO87=M6 /N5@PIO= AN5@PIOQ MN5@PIOV QN5@PIOVQ7 8N5@PIOVQW XN5@PIOW EN5@PIOWE7 8N5@PIOWEV 9N5@PIOWEV98 7N5@PIOWEV9870 AN5@PIOWEV9870AM RN5@PIOWEV987= MN5@PIOWEV9= MN5@PIOWEV9=M8 AN5@PIOWEV9=M8A7 XN5@PIOWEV9=M8A7XB 6N5@PIOWEV9=M8A7XB6L DN5@"
"PIOWEV9=ML 4N5@PIOWEV9D RN5@PIOWEV9DRB QN5@PX MN5I MN5IM6 ON5IM6OU =N5IM6OU=E @N5IM6OU=E@8 7N5IM6OV 7N5IM6OV7. EN5IM6OV7.E= 8N5IM6OV7.E=8@ 4N5IM6OW aN5IM6OX PN5IM6OXP= EN5IM6OXP=E< 4N5IM6OXPW 8N5IM6OXPW87 @N5IM7 8N5IM786 ON5IME =N5IME=6 @N5IME=7 VN5IME=7V< 6N5IME=7V<6O DN5IME=7VD ON5IME=7VDOW PN5IME"
"=8 VN5IME=8V< 4N5IME=8V<4D KN5IME=8V<4DKC ;N5IME=< VN5IME=<V6 7N5IME=<V@ 4N5IME=<V@4- 3N5IME=<V@4-3O .N5IME=<V@4-3O.D ;N5IME=<V@4-3O.D;K 6N5IME=D VN5IME=DVW ON5IME=DVWOU <N5IME=DVWOU<P KN5IME=DVWOU<PK@ XN5IME=L PN5IME=LP@ QN5IME=LP@Q8 6N5IME=LP@Q86D ON5IME=LPD @N5IME=LPD@6 7N5IME=LPD@678 <N5IME=LPW "
"ON5IME=LPX ON5IML PN5IMLP@ QN5IMLPE @N5IMLPW JN5IMLPWJ@ QN5IMLPWJ@Q7 6N5IMLPWJ@Q768 AN5IMLPWJ@QE XN5IMLPWJ@QEX6 ON5IMLPWJ@QEX6O7 =N5IMLPWJ@QEX8 7N5IMLPWJ@QEXa _N5IMLPWJ@QEXa_8 7N5IMLPWJ@QEXa_87= 6N5IMLPWJ@QEXa_87=6. ON5IMLPWJ@QR ZN5IMLPWJ@QRZE XN5IMLPX JN5IMLPXJ@ QN5IMLPXJ@QE AP OPO@ 7PO@7. 9PO@70 6"
"PO@706V 8PO@76 8PO@768. /PO@768./( APO@768./(A9 =PO@768./(A9=J :PO@768./9 IPO@768./9IQ APO@768./9IQAR JPO@768./E 5PO@768./E5- =PO@768./E5-=M NPO@768./E5-=MNU VPO@768./E5-=MNUV' 0PO@768./E5N =PO@768./E5N=' 0PO@768./E5N=4 <PO@768./V APO@768./VAI 9PO@7680 IPO@7689 APO@7689AJ QPO@768E =PO@768E=. /PO@768"
"E=/ APO@768E=5 .PO@768E=5.N APO@768E=< 5PO@768E=<5- .PO@768E=<5N WPO@768E=D 5PO@768E=D5. <PO@768E=D5N APO@768E=V 5PO@768E=W 5PO@768E=W5/ 0PO@768E=W5< 4PO@768N 5PO@768V APO@768VA= EPO@768VAN =PO@7E =PO@7E=. MPO@7E=.MW 0PO@7E=/ QPO@7E=/QM NPO@7E=/QMNV DPO@7E=/QMNVDX 8PO@7E=/QMNVDX80 UPO@7E=/QV XPO@7E="
"/QW 8PO@7E=/QW8A IPO@7E=/QW8AIX NPO@7E=/QW8AIXN6 :PO@7E=/QW8AIXN6:M 9PO@7E=/QW8AIXN6:M9D 0PO@7E=/QX IPO@7E=/QXIN 6PO@7E=0 6PO@7E=5 6PO@7E=56. /PO@7E=56./- 8PO@7E=560 <PO@7E=56N APO@7E=56V APO@7E=56W APO@7E=6 8PO@7E=68. IPO@7E=68/ 5PO@7E=68/5. 0PO@7E=68/5.0< $PO@7E=68/50 IPO@7E=68/5< 4PO@7E=68/5A MPO"
"@7E=68/5AM. 0PO@7E=68/5AM.0- (PO@7E=68/5AM.0-(' $PO@7E=68/5AM.0-('$) 9PO@7E=68/5AM.0-(4 $PO@7E=68/5AM.0-(4$< 3PO@7E=68/5AM.0-(4$<3C DPO@7E=68/5AM.0-(4$V &PO@7E=68/5AM.0-(4$V&< DPO@7E=68/5AM.0-(4$V&<D, 9PO@7E=68/5AM.0-(4$V&<D,9N LPO@7E=68/5AM.0-(4$V&<D,9NLU CPO@7E=68/5AM.0-(4$V&L DPO@7E=68/5AM.0-(4$V"
"&LDC 9PO@7E=68/5AM.0-(4$V&LDN ;PO@7E=68/5AM.0-(4$V&LDN;C KPO@7E=68/5AM.0-(4$W &PO@7E=68/5AM.0-(4$W&' 9PO@7E=68/5AM.0-(4$W&'9< DPO@7E=68/5AM.0-(4$W&'9<DK JPO@7E=68/5AM.0-(4$W&< DPO@7E=68/5AM.0-(4$W&<D, 9PO@7E=68/5AM.0-(4$W&<D,9: IPO@7E=68/5AM.0-(4$W&<D,9:IK VPO@7E=68/5AM.0-(4$W&<D,9:IKVQ !PO@7E=68/5A"
"M.0-(4$W&<D,9:IKVQ!N 1PO@7E=68/5AM.0-(4$W&<D,9:IKVQ!N1L UPO@7E=68/5AM.0-(4$W&L DPO@7E=68/5AM.0-(4$W&LDC 3PO@7E=68/5AM.0-(4$W&LDC3N 9PO@7E=68/5AM.0-(4$W&LDC3N9< QPO@7E=68/5AM.0-(4$W&LDC3N9<QI BPO@7E=68/5AM.0-(4$W&LDC3N9<QIBR XPO@7E=68/5AM.0-(4$W&LDC3N9<QIBRX2 `PO@7E=68/5AMW 'PO@7E=68/5N APO@7E=685 .P"
"O@7E=685.N APO@7E=685.NA- IPO@7E=68< QPO@7E=68D 5PO@7E=68D5N <PO@7E=68N APO@7E=68NA/ .PO@7E=68V 5PO@7E=68V5< LPO@7E=68W DPO@7E=< 6PO@7E=<65 8PO@7E=<658- .PO@7E=N DPO@7E=ND. IPO@7E=ND.I6 MPO@7E=ND.I6M4 APO@7E=ND.I6M4A< 5PO@7E=ND.IA QPO@7E=ND.IAQ5 6PO@7E=ND.IAQ564 /PO@7E=ND.IAQ564/J VPO@7E=ND.IAQ6 8PO"
"@7E=ND.IAQ68/ 5PO@7E=ND.IAQ68/59 -PO@7E=ND.IAQ689 :PO@7E=ND.IAQ68J RPO@7E=ND.IAQ68JR/ 9PO@7E=ND.IAQ68JR/9Z MPO@7E=ND.IAQ68JRB 9PO@7E=ND.IAQ68JRB94 :PO@7E=ND.IAQ68JRB94:0 VPO@7E=ND.IAQ68JRB94:0VW `PO@7E=ND.IAQ68JRB94:0VW`X MPO@7E=ND.IAQ68JRB94:0VW`XMY (PO@7E=ND.IAQ68JRB94:0VW`XMY(K ;PO@7E=ND.IAQ68JRB"
"94:0VW`XMY(K;L UPO@7E=ND.IAQ68JRB9Z MPO@7E=ND.IAQ68L JPO@7E=ND/ VPO@7E=ND/VM WPO@7E=ND/VMWU XPO@7E=ND/VMWUX6 QPO@7E=ND/VMWUX6Q_ 5PO@7E=ND/VU WPO@7E=ND/VUWM XPO@7E=ND/V^ 8PO@7E=ND/V^80 WPO@7E=ND4 MPO@7E=ND5 6PO@7E=ND564 <PO@7E=ND568 VPO@7E=ND568V< XPO@7E=ND568V<XK IPO@7E=ND568V<XKIW _PO@7E=ND568V<XKI"
"W_M /PO@7E=ND568V<XKIW_M/R UPO@7E=ND568V<XM IPO@7E=ND568V<XW IPO@7E=ND568V<XWIA `PO@7E=ND568V<XWI^ `PO@7E=ND568V<X^ 3PO@7E=ND568V<X^3L 4PO@7E=ND568V<X^3L4K APO@7E=ND568V<Xa 0PO@7E=ND568V<Xa0/ IPO@7E=ND56M <PO@7E=ND56M<. /PO@7E=ND6 5PO@7E=ND65- /PO@7E=ND65-/4 VPO@7E=ND65-/4V8 <PO@7E=ND65/ 0PO@7E=ND65"
"/0- (PO@7E=ND65/0-(4 <PO@7E=ND65/0-(4<8 .PO@7E=ND65/0-(4<8.' &PO@7E=ND65/0-(4<8.'&$ #PO@7E=ND65/0-(4<8.'&$#; CPO@7E=ND65/0-(4<; CPO@7E=ND65/0-(4<;CK 3PO@7E=ND65/0-(4<;CK3+ LPO@7E=ND65/0-(< $PO@7E=ND65/0-(<$. MPO@7E=ND65/0-(<$C MPO@7E=ND65/0-(<$CMV 3PO@7E=ND65/0-(<$CMV3L UPO@7E=ND65/0-(<$CMV3LU; 8PO@"
"7E=ND65/0-(<$CMV3LU;8+ APO@7E=ND65/0-(<$CMV3LU;8. 4PO@7E=ND65/0-(<$CMV3LU;8.4+ `PO@7E=ND65/0-(<$CMV3LU;8.4+`' &PO@7E=ND65/0-(<$CMV3LU;8.4+`'&, KPO@7E=ND65/0-(<$K VPO@7E=ND65/0-(<$KVM LPO@7E=ND65/0-(<$KVML_ ;PO@7E=ND650 8PO@7E=ND6508/ APO@7E=ND6508/AI MPO@7E=ND6508< MPO@7E=ND6508<M4 ;PO@7E=ND654 3PO@"
"7E=ND65< MPO@7E=ND65L MPO@7E=ND65LMC 8PO@7E=ND65LMC8U VPO@7E=V MPO@7E=VM. WPO@7E=VM/ 8PO@7E=VM/80 IPO@7E=VM/80IN APO@7E=VM/80INA9 .PO@7E=VM/86 5PO@7E=VM/865. 'PO@7E=VM/865.'$ 0PO@7E=VM/865.'$0- &PO@7E=VM/865.'$0A 9PO@7E=VM/865.'$0A9- NPO@7E=VM/865.'0 (PO@7E=VM/865.'0($ -PO@7E=VM/865.'0($-< )PO@7E=VM"
"/865.'0($-<)4 DPO@7E=VM/865.'0($-L DPO@7E=VM/865.'0($-LDC NPO@7E=VM/865.'0($-LDN )PO@7E=VM/865.'0($-LDN)4 ;PO@7E=VM/865.'0($-LDN)4;+ UPO@7E=VM/865.'0($-LDN)4;+UK _PO@7E=VM/865.'0($-LDN)4;3 +PO@7E=VM/865.'0($-LDN)4;3+K UPO@7E=VM/865.'0($-LDN)4;K <PO@7E=VM/865.'0($-LDN)4;K<+ 3PO@7E=VM/865.'0($-LDN)4;K"
"<+3C 9PO@7E=VM/865.'0($-LDN)4;K<+3C9# APO@7E=VM/865.'0($-LDN)4;K<+3C9#AB JPO@7E=VM/865.'0($-LDN)4;K<+3C9#ABJR IPO@7E=VM/865.'0(L DPO@7E=VM/865.'0(LDN )PO@7E=VM/865.'0(LDN)$ 9PO@7E=VM/865.'0(LDN)$94 -PO@7E=VM/865.'0(LDN)$9; -PO@7E=VM/865.'0(LDN); 9PO@7E=VM/865.'0(LDN);9$ -PO@7E=VM/865.'0(LDN)C 9PO@7E"
"=VM/865.'< 4PO@7E=VM/865.'<4N APO@7E=VM/865.'A (PO@7E=VM/865.'A(- 9PO@7E=VM/865.'A(-9$ &PO@7E=VM/865.'L (PO@7E=VM/865.'L(0 DPO@7E=VM/865.'L(0DN )PO@7E=VM/865.'L(0DN); 9PO@7E=VM/8A QPO@7E=VM/8AQI NPO@7E=VM/8AQINX JPO@7E=VM/8AQINXJB :PO@7E=VM/8AQINXJB:9 2PO@7E=VM/8AQINXJB:92R ZPO@7E=VM6 8PO@7E=W MPO@7"
"E=WM. /PO@7E=WM./5 6PO@7E=WM./568 0PO@7E=WM./5680' &PO@7E=WM./5680'&$ (PO@7E=WM./5680'&$() QPO@7E=WM./5680'&( )PO@7E=WM./5680'&()< DPO@7E=WM./56N APO@7E=WM./56NA9 IPO@7E=WM./56NA9I( &PO@7E=WM./56NA9I(&- 8PO@7E=WM./56NA9I(&-8' )PO@7E=WM./56NA9I(&-8')0 VPO@7E=WM./56NA9I(&-8')0VR JPO@7E=WM./56NA9I(&-8'"
")0VRJB QPO@7E=WM./6 5PO@7E=WM./65- &PO@7E=WM./65-&$ 8PO@7E=WM./65-&$8' 0PO@7E=WM./65-&$8'0( QPO@7E=WM/ 8PO@7E=WM/80 IPO@7E=WM/80IA QPO@7E=WM/80IAQN :PO@7E=WM/80IAQN:6 XPO@7E=WM/80IAQN:6XB JPO@7E=WM/80IAQN:6XBJV 9PO@7E=WM/80IAQN:6XJ 5PO@7E=WM/80IAQN:6XJ5V -PO@7E=WM/80IAQN:6XJ5V-. _PO@7E=WM/80IAQN:6XJ"
"5V-R 'PO@7E=WM/80IAQN:6XJ5V-R'& $PO@7E=WM/80IAQN:6XJ5V-R'&$. _PO@7E=WM/80IAQN:6XJ5V-R'&$._U LPO@7E=WM/80IAQN:6Xa VPO@7E=WM/80IAQN:6XaVU _PO@7E=WM/80IAQN:6XaVU_R `PO@7E=WM/80IAQN:6XaV^ 9PO@7E=WM/80IAQN:6XaV^94 (PO@7E=WM/80IAQN:6XaV^9R JPO@7E=WM/80IAQN:6XaV_ .PO@7E=WM/80IAQN:6XaV_.9 5PO@7E=WM/80IAQN:6"
"XaV_.95$ 'PO@7E=WM/80IAQN:6XaV_.95$'B RPO@7E=WM/80IAQN:6XaV_.95$'BRJ UPO@7E=WM/80IAQN:D `PO@7E=WM/80IAQN:J 9PO@7E=WM/80IAQN:J95 6PO@7E=WM/80IAQN:J956- XPO@7E=WM/80IAQN:J956-XR `PO@7E=WM/80IAQN:J956-XR`. BPO@7E=WM/80IAQN:J9564 <PO@7E=WM/80IAQN:J9564<- XPO@7E=WM/80IAQN:J9564<-XR `PO@7E=WM/80IAQN:J9564"
"<-XR`. BPO@7E=WM/80IAQN:J9564<-Xa .PO@7E=WM/80IAQN:J9564<-Xa.& _PO@7E=WM/80IAQN:J9564<-Xa.V $PO@7E=WM/80IAQN:J9564<-Xa.V$& UPO@7E=WM/80IAQN:J9564<-Xa.V$&U; ^PO@7E=WM/80IAQN:J956R XPO@7E=WM/80IAQN:J96 VPO@7E=WM/80IAQN:J96VX UPO@7E=WM/80IAQN:J96VXUR `PO@7E=WM/80IAQN:J96VXUR`< BPO@7E=WM/80IAQN:J9R 6PO@"
"7E=WM/80IAQN:J9R6- 5PO@7E=WM/80IAQN:J9R65 VPO@7E=WM/80IAQN:J9R65V- .PO@7E=WM/80IAQN:J9X ZPO@7E=WM/80IAQN:J9XZ5 6PO@7E=WM/80IAQN:J9XZ564 <PO@7E=WM/80IAQN:J9XZD VPO@7E=WM/80IAQN:R JPO@7E=WM/80IAQN:RJ6 .PO@7E=WM/80IAQN:RJ6.$ aPO@7E=WM/80IAQN:RJ6.$aV BPO@7E=WM/80IAQN:RJ6.$aVB2 XPO@7E=WM/80IAQN:RJ6.$aVB2"
"X_ YPO@7E=WM/80IAQN:RJ6.$aVB2X_Y` 5PO@7E=WM/80IAQN:RJ6.$aVB2X_Y`59 ^PO@7E=WM/80IN VPO@7E=WM/80INV5 6PO@7E=WM/80INV56R APO@7E=WM/80INV56RAJ :PO@7E=WM/80INVD QPO@7E=WM/80INVDQ4 LPO@7E=WM/80INVJ APO@7E=WM/80INVR 9PO@7E=WM/80INVU QPO@7E=WM/80INVUQX aPO@7E=WM/80IR VPO@7E=WM/80IRVN `PO@7E=WM/85 IPO@7E=WM/"
"85IN APO@7E=WM/85INAD VPO@7E=WM/86 5PO@7E=WM/865. 'PO@7E=WM/865.'$ 0PO@7E=WM/865.'$0( &PO@7E=WM/865.'$0(&- 4PO@7E=WM/865.'$0(&-4N VPO@7E=WM/865.'$0(&-4NV^ aPO@7E=WM/865.'$0- &PO@7E=WM/865.'$0-&( QPO@7E=WM/865.'$0-&(QI NPO@7E=WM/865.'$0-&(QINX JPO@7E=WM/865.'$0-&(QINXJV 9PO@7E=WM/865.'$0-&(QINXJV94 <"
"PO@7E=WM/865.'$0-&(QINXJV9A BPO@7E=WM/865.'$0-&(QN XPO@7E=WM/865.'$0-&(QNX4 `PO@7E=WM/865.'$0-&(QNX4`D IPO@7E=WM/865.'$0-&(QNXI JPO@7E=WM/865.'$0-&(QNXIJ4 `PO@7E=WM/865.'$0-&(QNXIJ4`D CPO@7E=WM/865.'$0-&(QNXIJD ;PO@7E=WM/865.'$0-&(QNXIJa VPO@7E=WM/865.'$0-&(QNXIJaVD <PO@7E=WM/865.'$0-&(QNXIJaVD<L `P"
"O@7E=WM/865.'$0-&(QNXIJaVD<L`_ 9PO@7E=WM/865.'$0-&(QNXIJaVD<L`_93 ;PO@7E=WM/865.'$0-&< DPO@7E=WM/865.'$0-&<DN LPO@7E=WM/865.'$0-&<DNL4 #PO@7E=WM/865.'$0-&<DNL4#9 3PO@7E=WM/865.'$0A &PO@7E=WM/865.'$0A&( QPO@7E=WM/865.'$0A&(QI JPO@7E=WM/865.'$0A&(QIJ- NPO@7E=WM/865.'$0A&(QIJ-N4 :PO@7E=WM/865.'$0A&(QIJ"
"-N4:D `PO@7E=WM/865.'$0A&(QIJ-N4:D`L KPO@7E=WM/865.'$0N &PO@7E=WM/865.'$0N&9 #PO@7E=WM/865.'0 (PO@7E=WM/865.'0($ &PO@7E=WM/865.'0($&) 9PO@7E=WM/865.'0(< 4PO@7E=WM/865.'A (PO@7E=WM/865.'A(- 9PO@7E=WM/865.'A(-9$ JPO@7E=WM/865.'A(-9$JB :PO@7E=WM/865.'A(-9< NPO@7E=WM/8650 'PO@7E=WM/8650'& $PO@7E=WM/8650"
"'&$. (PO@7E=WM/8650'&$.(- )PO@7E=WM/8650'&$.(-)N 9PO@7E=WM/8650'&$.(-)N9D VPO@7E=WM/8650'- .PO@7E=WM/8650'-.$ &PO@7E=WM/8650'-.$&( QPO@7E=WM/8650'-.$&(QI NPO@7E=WM/8650'-.$&(QINX JPO@7E=WM/8650'-.$&(QINXJV 9PO@7E=WM/8650'-.$&(QINXJV9L DPO@7E=WM/8650'-.$&(QN 9PO@7E=WM/8650'< DPO@7E=WM/8650'<DN LPO@7E"
"=WM/8650'<DNL- .PO@7E=WM/865< DPO@7E=WM/865<DA 9PO@7E=WM/865A IPO@7E=WM/865AI- .PO@7E=WM/865AI< VPO@7E=WM/8A IPO@7E=WM/8AIN `PO@7E=WM/8AIN`6 XPO@7E=WM/8AIN`6XR 5PO@7E=WM/8AIN`6XR5Q VPO@7E=WM/8AIN`6XR5QVU ^PO@7E=WM/8AIN`6XR5QVU^a .PO@7E=WM/8AIN`Q XPO@7E=WM/8AIN`QXV RPO@7E=WM/8AIN`R UPO@7E=WM/8AIN`V X"
"PO@7E=WM/8AIN`VXJ ^PO@7E=WM/8AIN`VXJ^5 6PO@7E=WM/8AIN`VXJ^56- QPO@7E=WM/8AIN`VXJ^56. QPO@7E=WM/8AIN`VXJ^56.QR -PO@7E=WM/8AIN`VXQ ^PO@7E=WM/8AIN`VX_ QPO@7E=WM/8AIQ 9PO@7E=WM/8AIQ96 JPO@7E=WM/8AIQ9: VPO@7E=WM6 5PO@7E=WM65. 0PO@7E=WM65/ APO@7E=WM65< 8PO@7E=WM65<8N VPO@7E=WM65<8NV^ 4PO@7E=WM65<8NV^4/ 0P"
"O@7E=WMD NPO@7N APO@7NA/ XPO@7NA/XQ UPO@7NA/XQUI WPO@7NA/XQUIWa JPO@7NA/XQUV IPO@7NA6 5PO@7NA65. =PO@7NA65.=/ 8PO@7NA65.=/8- (PO@7NA65.=/8I 0PO@7NA65.=/8I0- (PO@7NA65.=/8I0-(' MPO@7NA65.=/8I0-('M: WPO@7NA65.=8 0PO@7NA65.=80( 9PO@7NA65.=80(9/ MPO@7NA65.=80(9/ME -PO@7NA65.=80(9/ME-$ WPO@7NA65.=80(9/ME"
"-$WV _PO@7NA65.=80/ (PO@7NA65.=80/(' 9PO@7NA65.=80/('9- IPO@7NA65.=80/(E -PO@7NA65.=80/(E-' IPO@7NA65.=80/(E-'I& QPO@7NA65.=80/(E-'I&Q) 9PO@7NA65.=80/(E-'I&QD 9PO@7NA65.=9 IPO@7NA65.=< /PO@7NA65/ 8PO@7NA65/8. =PO@7NA65/8.=I 0PO@7NA65/8= EPO@7NA65/8=E9 .PO@7NA650 8PO@7NA6508. =PO@7NA6508.=/ MPO@7NA65"
"08/ .PO@7NA658 IPO@7NA658I. =PO@7NA658I/ QPO@7NA658I/Q9 0PO@7NA658I/Q= 0PO@7NA658I/Q=09 BPO@7NA658I/Q=09BZ &PO@7NA658I/Q=09BZ&R UPO@7NA658I/Q=09BZ&RU- JPO@7NA658I/Q=09BZ&RU-J: EPO@7NA658I/Q=09BZ&RU-J:EV <PO@7NA658I/Q=09BZ&RU-J:EV<; 3PO@7NA658I/Q=09BZ&RU-J:EV<;3+ DPO@7NA658I9 QPO@7NA658I9Q/ 0PO@7NA65"
"8I9Q: =PO@7NA659 IPO@7NA659I. $PO@7NA659I/ 8PO@7NA659I/8= <PO@7NA659I0 8PO@7NA659I8 =PO@7NA659I8=. QPO@7NA659I8=.Q/ 0PO@7NA659I8=.Q/0: &PO@7NA659I8=.Q/0D &PO@7NA659I8=/ .PO@7NA659I8=/.< DPO@7NA659I8=: BPO@7NA659I8=B :PO@7NA659I8=B:2 RPO@7NA659I8=B:2RQ UPO@7NA659I8=B:2RQUM EPO@7NA659I8=B:2RQUV EPO@7N"
"A659I8=B:2RQUVE0 XPO@7NA659I: BPO@7NA659I:B. =PO@7NA659I:B.=/ 8PO@7NA659I:B.=8 /PO@7NA659I:BJ =PO@7NA659I:BJ=8 WPO@7NA659IJ =PO@7NA659IJ=: QPO@7NA65: =PO@7NA65:=8 IPO@7NA65:=8IQ 9PO@7NA65:=9 IPO@7NA65:=9I8 BPO@7NA65:=9I8B/ .PO@7NA65:=9I8B/.< (PO@7NA65:=9I8B/.<(0 QPO@7NA65:=9I8B/.<(0QJ RPO@7NA65:=9I8"
"BJ RPO@7NA65:=9I8BJRZ QPO@7NA65:=9I8BJRZQX 0PO@7NA65:=9IB QPO@7NA65:=9IBQ8 RPO@7NA65:=9IBQ8R. /PO@7NA65:=9IBQ8R/ EPO@7NA65:=9IBQ8R/EM .PO@7NA65:=9IBQ8RJ 2PO@7NA65:=9IBQ8RJ2- .PO@7NA65:=9IBQ8RJ2-.& EPO@7NA65:=9IBQ8RJ2-.&E/ 0PO@7NA65:=9IBQ8RJ2-.< 0PO@7NA65:=9IBQ8RJ2-.<0& 'PO@7NA65:=9IBQ8RJ2-.<0&'$ ;PO"
"@7NA65:=9IBQ8RJ2-.<0&'$;D /PO@7NA65:=9IBQ8RJ2-.<0&'$;D/( EPO@7NA65:=9IBQ8RJ2-.<0&'$;D/(EL CPO@7NA65:=9IBQ8RJ2/ .PO@7NA65:=9IBQ8RJ2/.$ EPO@7NA65:=9IBQ8RJ2/.$EL (PO@7NA65:=9IBQ8RJ2/.$EL(' 0PO@7NA65:=9IBQ8RJ2/.$EL('0< ;PO@7NA65:=9IBQ8RJ2/.$EL('0<;D &PO@7NA65:=9IBQ8RJ2/.$EM (PO@7NA65:=9IBQ8RJ2/.< (PO@7N"
"A65:=9IQ UPO@7NA65:=9IQU/ 8PO@7NA65:=9IQU/80 'PO@7NA65:=9IQU8 EPO@7NA65:=9IQU8E/ WPO@7NA65:=9IQU8E/W- $PO@7NA65:=9IQU8E/W-$< 4PO@7NA65:=9IQU8E/W-$<4. 3PO@7NA65:=9IQUM EPO@7NA65:=9IQUME8 VPO@7NA65:=9IQUME8V/ BPO@7NA65:=9IQUME8V^ `PO@7NA65:=9IQUMED WPO@7NA65:=9IQUMEDWL XPO@7NA65:=9IQUMEDWLXV _PO@7NA65"
":=9IQUMEDWLXV_4 <PO@7NA65:=9IQUMEDWLXV_4<a JPO@7NA65:=9IQUMEDWLXV_4<aJB `PO@7NA65:=9IQUMEDWLXV_8 <PO@7NA65:=9IQUMEDWLXV_8<a JPO@7NA65:=9IQUMEDWLXV_8<aJB 2PO@7NA65:=9IQUMEDWX aPO@7NA65:=9IQUMEDWXaL VPO@7NA65:=9IQUMEDWXaLV8 <PO@7NA65:=9IQUMEDWa VPO@7NA65:=9IQUMEL VPO@7NA65:=9IQUMELVD <PO@7NA65:=9IQUME"
"LVD<8 WPO@7NA65:=9IQUMELVD<8WX JPO@7NA65:=9IQUMELVD<8WXJB 2PO@7NA65:=9IQUMELVD<8WXJB2. 'PO@7NA65:=9IQUMELV` _PO@7NA65:=9IQUMELV`_^ JPO@7NA65:=9IQUMELV`_^JB 8PO@7NA65:=9IQUMELV`_^JB8R WPO@7NA65:=9IQUMELV`_^JB8RWD <PO@7NA65:=9IQUMELV`_^JB8RWD<0 /PO@7NA65:=9IQUMELV`_^JB8RWD<0/a XPO@7NA65:=9IQUV JPO@7NA"
"65:=9IQUVJ8 BPO@7NA65:=9IQUVJ8BR /PO@7NA65:=9IQUVJ8BR/0 MPO@7NA65:=9IQUVJ8BR/0ME DPO@7NA65B EPO@7NA65BED LPO@7NA8 IPO@7NA8I6 UPO@7NA8I6UM 0PO@7NA8I6UM09 /PO@7NA8I6UM09/R BPO@7NA8I9 /PO@7NA9 QPO@7NA9Q8 IPO@7NA: MPO@7NA:ME UPO@7NA:MEU6 WPO@7NAB WPO@7V APO@7VA6 8PO@7VA68N =PO@7VA8 QPO@7VA8Q/ NPO@7VA8Q/"
"NB 5PO@7VA8Q/NB5= EPO@7VA8Q/NB5=EM 6PO@7VA8Q/NB5=EM6< DPO@7VA8Q/NB5=EM6<D; LPO@7VA8Q/NB5=EM6<D;LS :PO@7VA8Q/NB5=EM6<D;LS:2 9PO@7VA8Q/NB5=EM6<D;LS:29. $PO@7VA8Q/NB5=EM6<D;LS:29.$& _PO@7VA8Q/NB5=EM6<D;LS:29.$&_# 4PO@7VA8Q/NB5=EM6<D;LS:29.$&_#40 3PO@7VA8Q/NB5=EM6<D;LS:29.$&_#403+ KPO@7VA8Q/NB5=EM6U WPO"
"@7VA8Q/NB5=EM6UW. -PO@7VA8Q/NB5=EM6UW.-I JPO@7VA8Q/NB5=EM6UW.-IJa `PO@7VA8Q/NB5=EM6UW.-IJa`_ :PO@7VA8Q/NB5=EM6UW.-IJa`_:R ZPO@7VA8Q/NB5=EM6UW.-IJa`_:RZD 4PO@7VA8Q/NB5=EM6UW.-IJa`_:RZD4< ;PO@7VA8Q/NB5=EM6UW.-IJa`_:RZD4<;9 KPO@7VA8Q/NE 9PO@7VA8Q/NE9I 0PO@7VA8Q/NE9I0B :PO@7VA8Q/NE9I0B:2 JPO@7VA8Q/NE9I0"
"B:2JR MPO@7VA8Q/NE9I0B:2JRMW ^PO@7VA8Q/NE9I0B:J RPO@7VA8Q/NE9I0B:JR( _PO@7VA8Q/NE9I0B:JR(_^ ]PO@7VA8Q/NI JPO@7VA8Q/NIJ: WPO@7VA8Q/NIJ:WE =PO@7VA8Q/NIJW XPO@7VA8Q/NIJWX: 5PO@7VA8Q/NIJWX:5B 2PO@7VA8Q/NIJWX:5B2U 6PO@7VA8Q/NIJWX:5B2U6M =PO@7VA8Q/NIJWX:5B2U6M=E LPO@7VA8Q/NIJWX:5U 6PO@7VA8Q/NIJWX:5U6M EPO"
"@7VA8Q/NIJWX:5U6MEB 2PO@7VA8Q/NU MPO@7VA8QR IPO@7VA: 6PO@7VA= IPO@7VAE NPO@7VAEN/ =PO@7VAEN6 IPO@7VAN QPO@7VANQR 5PO@7VANQR5I W@ 7@7. /@7./0 '@7./0'N 8@7./0'N8I 6@7./0'N8I6- (@7./0'N8I6-(9 B@7./0'N8I6-(9B: 2@7./0'P 6@7./N 8@7./N89 A@7./N8A 6@7./P 6@7./P65 E@7./P6= E@7./P6=E5 8@7./P6=E58O <@7./P6E N@"
"7./P6EN= 8@70 /@70/. 6@70/N 8@70/N89 A@70/N89A6 .@70/N8I P@70/N8IPA O@70/P 6@70/P6' 8@70/P6'8. O@70/P6'8.O- 5@70/P6'8.O-5A #@70/P6'8.O-5A#= 4@70/P6'8.O9 I@70/P65 =@70/P65=8 O@70/P65=8O( I@70/P65=8O(IQ A@70/P65=8O(IQAB J@70/P65=8O(IQABJR 9@70/P65=8O(IQABJR9: X@70/P65=8O. I@70/P65=8O.IQ A@70/P65=8O.IQ"
"AB J@70/P65=8O.IQABJR 9@70/P6= O@70/P6=OE N@7N 8@7N8/ &@7N80 O@7N89 A@7N89A0 /@7N8A O@7N8AOI Q@7N8AOIQP J@7N8AOP J@7N8AOX 5@7N8AOX5I P@7N8I P@7N8IP/ 6@7N8IP/60 .@7N8IP9 A@7N8IP9AB O@7N8IP9AW O@7N8IPA O@7N8IPAO. /@7N8IPAO/ 5@7N8IPAO/5= 6@7N8IPAO/5V Q@7N8IPAO/5VQW J@7N8IPAO0 9@7N8IPAOQ 9@7N8IPAOQ96 B@"
"7N8IPAOV Q@7N8IPAOX Q@7N8IPW O@7N8IPWOX J@7P 6@7P6- 8@7P6. /@7P6./5 =@7P6./5=8 0@7P6/ 8@7P6/8. 0@7P6/80 (@7P6/80(& O@7P6/80(&O. I@7P6/80(&O.IM A@7P6/85 =@7P6/8= O@7P6/8A '@7P65 =@7P65=- 8@7P65=. O@7P65=/ 8@7P65=8 .@7P65=E N@7P6= 8@7P6=8/ 5@7P6=8/5. E@7P6=8/5.E0 A@7P6=8/5.E0AM 9@7P6=8/5.E0AM9: J@7P6="
"8/5.E0AM9:JL Q@7P6=8/5.E0AM9:JLQR N@7P6=8/5.E0AM9:JLQRND 4@7P6=8/5O (@7P6=8/5O(' E@7P6=8/5O('E) 0@7P6=8E O@7P6=8N O@7P6=8O A@7P6=8OA- N@7P6=8OA-N/ I@7P6=8OA-N/IQ E@7P6=8OA-N/IQE0 V@7P6=8OA-NM I@7P6=8OA. 5@7P6=8OA: E@7P6=8OA:EM N@7P6=8OAB I@7P6E N@7P6EN. O@7P6EN/ 8@7P6EN/85 =@7P6EN/85=< O@7P6EN5 =@7P"
"6EN5=W O@7P6EN= O@7P6EN=O/ -@7P6EN=O/-. 5@7P6EN=O/-0 5@7P6EN=OM 9@7P6EN=OM98 5@7P6EN=OM985: <@7P6EN=OU I@7P6EN=OUI8 V@7P6EN=OX M@7P6ENU O@7P6ENUOV W@7P6ENUOVW_ M@7P6ENW O@7P6ENWOV X@7P6ENWOVXa M"
;
    for (i = 0; i < hw2; ++i)
        char_keys[chars[i]] = i;
    const int first_board[b_idx_num] = {6560, 6560, 6560, 6425, 6326, 6560, 6560, 6560, 6560, 6560, 6560, 6425, 6344, 6506, 6560, 6560, 6560, 6560, 6560, 6560, 6344, 6425, 6398, 6560, 6560, 6560, 6560, 6560, 6560, 6560, 6560, 6479, 6344, 6398, 6074, 6560, 6560, 6560};
    int ln = param_compressed1.size();
    cout << "book len " << ln << endl;
    int coord;
    board fb;
    book_hash_table_init(book);
    int n_book = 0;
    int data_idx = 0;
    int y, x;
    int tmp[16];
    while (data_idx < ln - 1){
        fb.p = 1;
        for (i = 0; i < b_idx_num; ++i)
            fb.b[i] = first_board[i];
        while (true){
            if (param_compressed1[data_idx] == ' '){
                ++data_idx;
                break;
            }
            coord = char_keys[param_compressed1[data_idx++]];
            fb = move(&fb, coord);
        }
        coord = char_keys[param_compressed1[data_idx++]];
        y = coord / hw;
        x = coord % hw;
        register_book(book, fb.b, calc_hash(fb.b) & book_hash_mask, y * hw + x);
        for (i = 0; i < 8; ++i)
            swap(fb.b[i], fb.b[8 + i]);
        register_book(book, fb.b, calc_hash(fb.b) & book_hash_mask, x * hw + y);
        for (i = 0; i < 16; ++i)
            tmp[i] = fb.b[i];
        for (i = 0; i < 8; ++i)
            fb.b[i] = reverse_board[tmp[7 - i]];
        for (i = 0; i < 8; ++i)
            fb.b[8 + i] = reverse_board[tmp[15 - i]];
        register_book(book, fb.b, calc_hash(fb.b) & book_hash_mask, (hw_m1 - x) * hw + (hw_m1 - y));
        for (i = 0; i < 8; ++i)
            swap(fb.b[i], fb.b[8 + i]);
        register_book(book, fb.b, calc_hash(fb.b) & book_hash_mask, (hw_m1 - y) * hw + (hw_m1 - x));
        n_book += 4;
    }
    cout << n_book << " boards in book" << endl;
}

inline double leaky_relu(double x){
    return max(0.01 * x, x);
}

inline double predict(int pattern_size, double in_arr[], double dense0[n_dense0][20], double bias0[n_dense0], double dense1[n_dense1][n_dense0], double bias1[n_dense1], double dense2[n_dense1], double bias2){
    double hidden0[32], hidden1[32];
    int i, j;
    for (i = 0; i < n_dense0; ++i){
        hidden0[i] = bias0[i];
        for (j = 0; j < pattern_size * 2; ++j)
            hidden0[i] += in_arr[j] * dense0[i][j];
        hidden0[i] = leaky_relu(hidden0[i]);
    }
    for (i = 0; i < n_dense1; ++i){
        hidden1[i] = bias1[i];
        for (j = 0; j < n_dense0; ++j)
            hidden1[i] += hidden0[j] * dense1[i][j];
        hidden1[i] = leaky_relu(hidden1[i]);
    }
    double res = bias2;
    for (i = 0; i < n_dense1; ++i)
        res += hidden1[i] * dense2[i];
    return res;
}

inline void pre_evaluation(int phase_idx, int evaluate_idx, int pattern_size, double dense0[n_dense0][20], double bias0[n_dense0], double dense1[n_dense1][n_dense0], double bias1[n_dense1], double dense2[n_dense1], double bias2){
    int idx, i, digit;
    double arr[20];
    for (idx = 0; idx < pow3[pattern_size]; ++idx){
        for (i = 0; i < pattern_size; ++i){
            digit = (idx / pow3[pattern_size - 1 - i]) % 3;
            if (digit == 0){
                arr[i] = 1.0;
                arr[pattern_size + i] = 0.0;
            } else if (digit == 1){
                arr[i] = 0.0;
                arr[pattern_size + i] = 1.0;
            } else{
                arr[i] = 0.0;
                arr[pattern_size + i] = 0.0;
            }
        }
        ev_arr[phase_idx][evaluate_idx][idx] = predict(pattern_size, arr, dense0, bias0, dense1, bias1, dense2, bias2);
    }
}

inline double get_elem(const string *s, int *idx, string *chars, int ln_char){
    double res = 0.0;
    for (int i = *idx + digit_num - 1; i >= *idx; --i){
        res /= (double)ln_char;
        for (int j = 0; j < ln_char; ++j){
            if ((*s)[i] == (*chars)[j]){
                res += (double)j;
                //cout << j << " ";
            }
        }
    }
    *idx += digit_num;
    res -= 4.76725673675537;
    //cout << res << endl;
    return res;
}

inline void init_evaluation(){
    string chars = "!#$&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~";
    string param_compressed1 = 
"(`y(Tu'xq'<)'jz&v{'Cw'l('~['y>&eJ'x?(0<(!p((9'1f'2c)/a&tW&n[&UY(P|&EP'r<'d6'29(L>(-`'fB'px&^^'v#'TR(.s(Ch&fO'Cn(+Z'N((7,'E7'sl(0Z'MX'UM'A,'SF'/c']f'gW&^T'=:'C?(#c';a&Vl'-V(Lb'Z2'wg(,+(#6'E2&tp'92'mv'tB'bL'Pj'9|'UP('Y(LG'Co'Xz&q5(GA'PC'<|'5S(+-'|]'}Q'NR&2c(-u'^t(&L'yk'`S'lg&/W'&|(JK'Y0&bN(<#(bZ(0$&`B"
"&dD'sb&Aj('!(!Z'^.(p:(E2(){(4J':@'7o'j{(;a'ud(0M'>'(>Z'~V'#A'Bj(C1&iL'h=((W'QQ':[(6C'AC'JS'lK'8A(8@(@A'vD'hd'+C'<9'}R'~4&m{(@7'*V(+N(]E&yc&s8'hb(Sr'JK(3=&oP&~M&b&'~/(,:()M&|*(;z'PO'Bj'wC&^&(#s'WB'Qv(?,((-'Kt'5r&Tm'a,&NZ'Q=(-w'hh'R;()n'Z.'I_(/O'{.&vt'T=&i+'^J'5B()^((P&Wp($W'G;'HB'm['qO('q'd]'{|'PB'&e"
"'6U'5z'@-(:{'v<'B7&|!'oJ'S~'yG'L&'(x(-R&2y(?<'_N'e;'VJ'W`';b&N?':>'vG'Hs'uC'/c'4F).}&~H&}1(;M(1z'a:&6R'*B'zv&i,'Y#(/e(+d'u.'=X('6'`;(Dm(&9'[H'<8'h&'Va'wY('5';y'Q>'YM'Kt'3*'>{'Y]'Z}(#{':e'd*'ds'p>('5'X4'd<'X9'i#'Ck&}>'FZ(/3'jR(Wl(Pg'PZ(9G'm]'r('~n'hb(<H(Nx(+d'MY(T='I/'P<'^['*P'#n'Kp'9-'~s'`u'q''Vg'Y-"
"'=b'py'@|(8x'm~'pK'U['o.',s(R2(.g'M&'z&'nI(2H'y@'D5'TR'Tj'v9(*''|?'ek'k1'F((M^('J'lX(/D'r5'Cl've(*J'wy'_3'0x'Zi'QY'}0'f~'5d'76'`:'{A'Vr'w6'D&'tI(3>'v4(+N'dp'|>((b'T.'sh(ZZ&h?'<r'X@'rl'b|(9C'fw'`&'lr(..'f|'x2(!*'^i'tu(3)'rH'dJ'k9'oR'z}'S$'rC(BB(+&']$&p&'u}'k,()x'mP(*E&qn'4B'q2(9+'o4(1f'j='0|()1(1J'QW"
"'al(-i'xr'U:&qB'.H'Z5'wb'uF'm@(,g'ov'/k's{'Za'c_'g/(7s('.'?Q'HK'Td'pw'xB'tb'vA(7n'zs'+F()&'eb'r/'2O($}'sd'^u'8v';S']z'me'V~'j^()Z'm=(*3'Zd'ub(9y'vc'w4'|E'_#'X8'e_'KH'EI'w1'W4'?+'pa(7Y'w4'vg(D!'[x(/9'f5'u8'(!'nn(+k'q.(*@'tR(Mu'f((3q'yH'z9&44(Rc'RG'u,'A)'oW'[}'Wh(5='W:'Vq(6!'za'05'gf'ys'kw'}u'E`'cb(#~"
"'uL'Sj(9B'jO'e/'r?'n^'o((O#(R<'a>'4<'r&(,s(4p()j(,p(H3'k3(2^(H&'xa'a8(?}'y9'c+(G`'y}'P4'LX'?F'U_'Ua'Wr'OJ'Cg'O''My'r!'Fr'q('Bn'EW'F5'et'a+'g['m4'q$'xh'k('o2'i8'jw']_'k.'tj'hS'l_'`.'iI(@r'g#(bn't*'H|'HR(6?(+1(+((!I'|n'eN'r5'gz':b'[M(UV'IK'+R'?s'4r'J_'o='Z6(4P'ZH'RU$t#'iF&:J'^r'a)'oo&tL&[0'xK'hP'~I''M"
"'w1'?`'f{'Vk'H`'Qv'*X'mp'A*':='Bk'wE']i']Y'ha'KX'b'&c8';0(1='cb($g&o!'Gh(/c't_'`;()t'JM&}b&pT'Pb'vE&tL&rz(7a'L_&S0&v2'ai((9'Vi'NH'O?'H7(7.(Wy(!q(3='Jd&l2'Z3(0k'mu's|(1g'Qp(P='j{']!'/e(+>'{5'23'm6'U1('r'Fv'$2(a4'{q&/('$}'e}'K6'`J&x_'UW'SQ'J['=D'^['ea'zy(@l'<+(8|'vJ(E0'nw&fC'z['yT'3)'8K':Z'zS'No'b[(4K"
"'i9'Vd(C;'dx&b8'@c&R['z;'p<'YX'9l'g7(Lo&)+'f<'A:(?x'hZ'Me'G>&fH'WA(=L'La'ok(2[&TD&yt'bK'-m(,E&c='v,'}>&oY'mh&y8'W9'*$'b*'`Y(*X';Z(C:'!m'<X'Rj'mQ(-,(&('[N'RK'Th'u!'|C'c{'rj'9)'0K'>e(,o'-|(0_'s!'@^((k'?.'M#'k''jQ('E&({(ZN'~d&pF(C_'Vv'FD$}&&o,(CR'|$(.H':@'ye'F:'K[(${(5b'ap'{O'NV(Wu&@D'mL&`M(:((8}'}](UC"
"'|8'nP&Ug'YU&|O'($(.|'Gd';&'5$(1R'uu(Z9'dU'`'&je(*p'K('zT']a'q+'V8'o{'wB'CS'&G(>W'rn'hG'H/(2!'VC'gK'_I'u1'RH'nU'bt'Oo(!P(!$'uO'7i(/u'W;'t<(@v'[]'Yx(Kk(#G(;G'm8()z(O#'|j(dF'nb('a'Qt(MX(4)'8Y($j'c=&jI(EE(XG(..'a{(!D(:7(<d'pk(6J(:X(.R'(&'M''rR(@m(&e':0'z:'eE'mC'pW'eX'K{'|K'hR'3l'S_')k'hD'dr(0+'Nq'jx(@1"
"'qe'[U'jQ(FJ'xG'}5(0*'>@'eU'}b'q)'wB(jV'!j(:m'f#'nr'~y'O7'H]'pt'|h'W#(/n'9h'qy(vr'YI(bp(39(Nh'gS'l2'RN'g|'sC(*5'[,'hE'IQ&|b(p+'W_'|-(??(.n'a~'z)'a!']_(!6('('gZ'ZT'qz'Hl'^0'Qs'd,'y@']='li(7-(Qb(0^'yL(+j'|](E1'm0'dX(6)(7;(_?(<z'^i(Cq(aX&u9(9{'t$'ym(@r(0_'Q.'gF('-'G](-$(4a&Ij'pl'1y'[m(BN'9J'{o'cK'f~'s>"
"(*v'}h'>9(BA&tJ&Bf&{6(#n(8r(=)'o2'k$'n=']<'g_'zC'7W'g]'qO'2r'F@'mo'Z`'[X'[|'Z!'x.'+1'kR'Th'ad'V('-T'fg'S>'j#'=g'gc'|N'p[&aG(XX&ne'MW'ZC(=.((r']d'ck'V}'n:'5s(]d(MM&m5(!A'@$(9}'c<'h?'z+'y!'b~'Nt(57'nA'S''V7(G^(FI'<f('V(')'0~'i3'Y!'f_'t1'N^'b2($1'x;'aH'q<(5D'`-'T+'de'zR'aL'0~'<j'I>'F#'B-'Er'<T'In'C]((N"
"'Ka'FU'Tc'Aw'6Q'sN']r'vO'j#'j='jc'jV'b['i}'n('a7'l@'l,'fN'j*'_F'iI'Sh'Zy'$('A;'sx'iP(AC&V2'ys&;_'M)(Bq(@?'9j&a<'fq'[U(@^&7N'a4'Ok'!S(&d'rz'qW'I(&nd'zQ(.s(?q&N{'.''k!(3&&~q'nJ'G3&uA&5n((w'kL'k^(DW(@^'],&tW'K0(2#'p('h<(5''t|(.N'!j'z;'=&'tP'm|(#^'~R'CT'vW'Ry(2B';b'@#(9U'zu'ri'mH'IE'+_(Bh'|f'n7'rq'F,'mv"
"'ra'Ot(IE(@+'fK(AX'UM'Z)'cj'Cr'Fy'SQ'Zt&HG&|8()A(;{&vB'n|(*w'_9'T`(MM'_^'q?'=u'>o'Zr'tM&;f'wU'w8(-P'H/'(1'~B(.k&Y]'!P(5z(M''Z)'Ww'Tc'l'&2h'/J(&L'})'?]'pR'YU'>W(LZ'rm(J9(2a'N?(>4($?'@@(+o(Ix'w('lV'D|(*0'f$(=c'p+'iV&QJ'FP'N9(&:':y$e3'SG'3|'K_'x](jY'vL(&2'YP'Z7'<v'Pm'JP'u('v3'@N&qT(/E'lo&Kf'PT'80'`!'wz"
"(=('BO'li(&W'd/'7='s6&X?'K_'eq'RB&e*'a1'hj'zp'QR(/m'Db'$T(!w&`A'}&'-_'ha'iU($>'|6'p]'W('`/']N(@d'q;(#z'_E'|B'Jg(8F&DI&`/'a>'Ku'0d'ta'6G'Q!&p8'$I'Qx(Ts'Tk'k,'Vb(&t&Iw(B9(6m'@:(*<'[P(O;'ax&e|'/G(f''CE'P@';Z(3A'=~'*c'{O(7<'[*'EA(-k',G($9'V4'P^(+r'Q<']4'l`'EB'9Q'~h'e}'mo(/)'O{'qw'Y*'gs'q6(!,(57'ur'yJ(*B"
"'{C'pV(#J'50'[C(&u('&(#4'rT'?4'w''[u'hR'[1'V8'/f'ft'mK's`'F6'HQ'k8'j9'j8(03'XA'3Z'Rw'ni'oj'[c'}u'k,'oM'E>'PP'R7'XV'K0'ma'Py(*(&e&'S''Q='kO'cU'Y/'ex'f?'z_'8W(;<'^K'v.'ka'Y2(#~'1*(0b(&E(dB'x!(Ra'u0'hC'fi(Y+'Sr'h5'~E'|k'=@'bZ'iq'|''rZ'|>(6d'p&'s('uo(<>'C{(1k'wE(&P'q$'3J(<#(3|(+Y(>`(8.'|X(-!(0g'{@'PC(YB"
"(!f(1r(*Q($X'{n'n^'d5(Ko(Wp'u6('-(.Z'zm((F'xe&qe'{n(M&(1](5W'C!'Vw'|8'{='n1'fR'5I'FR'lt'jr'6*(I3'#1'{s'i:'[?'m#',''hJ(Br'n@(-c'<J'|K'o4'cR'S/(,k'^~(?i'Fd'].'1H'+2(2}'UR'H_'NE(+R(5|'o-'my'dk'[:'L>'oa'p@'p}(+#'$1'#s'|+().'h:'nD'HF'i5'gd'ps'w6'Hp'ud'yI'pO&Zu(;{'_r'_8(,c(Y^'vK(A.(#@'u''w,'E:'bq(6d'rV(51"
"'sO(;e&k@(F0(;?'cj'&P(Ry's4'WV'FN(NL'&D'Y;'kt'iN'Ga''2(90'}E(#g(&~'V!'qt'l4'f['H}'Z('yK'sk'qy'eq&Yu(#L';.'ZC'o['}d'QB'BD'v2'o_(+G'lD(!^'hS'Tv'}]'O?(7{'ia'.Z'#q'@S'NT'=,'I0'L3'F!'js'hA'?L'>9'JY't!'iJ'_F'gK'jx'jk'j3'aB'j$'j+'lK'_+'n['k?'la'j('a-'tz'iI'xB(0X'p&(*n'z9'<v'^a'BR(!K(QV('I'(B&qw(8I'!9'Km&vI"
"'7q'6n'GE&Kp((B'{P'e('uD'5/'VI(MF&_f'wI(-P'a6'xZ']3'Gp(#^'{s'[O'gZ'!3'S#(7.'`n'Nz'i1'r''#_'wc'UA'mH';!'DD'<0';y'^6'8-(Q(&M8&l#()Z'Wo&x!'lf&pf'6_',t'i,'m](Oq(CP'QH'qM&xD'6*'3}'o;($#'ql(>.'Q&'^2'zJ(5-'hj'*r&r;'JB(#E'x('Q`'xB'@$'GE'[.'(E(9y'wP'sW&gz(+a')<'-!(!]'uj'jv'A@'^c&|b(_#&~J'G9'[c(0!'Jn'U)&}@()x"
"'w`'co(/+'nU'zY's2'r{'Bn'1x'EF(75'&Z&y<'t''iv'Xo'h7'l+'Os'Cy'F7(S~'i''mH&_n'@n&s}(H|'I2(<w'{y(0.'iy'[~(>^''M'Rl'BC'{n'sy'uP'Ez(-x'JZ'Om'[@'.d'a{'<a'NY'zm'Kq'R6'k)'`P'Xr'R8'G,'M;'q+'6}(!L(I?'8q'cx'`Q'BO('y'xb(!('jp'u;'nu'o<':g'tH&h&'^K('x'>w'}3'de'm-'r]'m?'Xv'bd&_2'Kj(5T't$'eW(.A'ex(JR'~Z'o-'~Q'M@'nL"
"'Z@(!C'^e'dB'U](*#'/b'a)(77'rU'N9'`C((<(8?'6S'xJ'eY'l[(#f(KC'tH'f<($V()m'K_'bn'w`'~](J!'dT'>5'i}'sD'Um'gL&wj'o/'u>'E&'zT'U`(6h'8D'RZ'h3'M]'}~'hw'cR'b.'tp'2N'rj'h-(0?'Uf'Pj($v'yb(+Y'^Q'mz'cf'{)(FK'a'(7o'Qo(&t(/<(1A'hh'zd'aQ'yy'mo&`G'XJ'as'`)'jZ'`^'of'4w'dX'hY'PW'|,(<I'vF'B;('s'bc'eW(?+(+s(&j'Xs(!.'<2"
"'t_'zl(=z'oZ'hu'q/'F^'x'(L;'u=&w]'ah((.(2x'xL&qn(!Z(9W(IF(*Y'il'r.'|{'tg'<2('''AV(+-'s='iT'o[(2h'z-'v)'9,'^6'a?(!Y'Ou't;'^Z(*t(;B'v)'k3']R'gJ'F,'iV(3a'a8'w9(/^'|Z(Qb'zC'iE'iN&|M'y7'hy'fM'jQ'wM'uo(9*'f|(?Z'vl(3g'Le(!=(@.'G8'`G'~~'[i'M6'n['`^'ed'i^'P.'UV'pf(3='n)'cX(;D'uP(<O'}b(+!(59's-&{x'P9'yI();'l^"
"'?g('-(=t']s'yR'[q'$4(.*'wC(VY'm@'IC'~h'fn'bm'QC'7$'NL'f]'I5(!+'Ld($`'J#'QP'K:'Xg'lb'WS'8q'I`'jg'h2'pM'r''kC'`&'jX']/'iX'iq'n~'jB'oY'j^'jp'iu'iI&W_'1I(M~&L~(6*'0X(+['B@&q<'sK'r0'o@(.['go't)&RH'PZ'ak'EZ'tP'QP()N'cz'Ve&HL'Ii(L1(69';W'I0(27'u['[.'zw'MO'vc'3_(ET'_:'nC'1Z'^]&[b'JD'71(;[&Mj'i3';6'aC'yh'L_"
"'DX'LD'>y'`('G/']*'y&'#w(7D'0K&q3'oA'E|'UF'?`'R~'K{'g>&Hk'0,'h&'lv'oD'Kk'aT'zL'xX(4_'yW(dR'Au'SO'3M(3T'+Y'9I'r*(i~((d(HG'2a'to(=)'-_(G2'Ke(*T(4)(C2&L5'w9'Px'95'Ae'W-'uA(8Q(6$'BJ'BU'c_('y'tc'>b'>v&Rc'c&(Mf'QR'$:'kf'oK&`o'>((4I'n)'5w'nn']l'yY($1'zv'pl'f)(Sl'RU(!E'sK&zc&xo&QR'P~&|J'r6(MV&ED&|C'[2(C^'Rj"
"'XU'^I(8@'r('eU'Ps'xL(7<'YK(ON'NM'sV(?V'Zg(PW(1F'e~'|n'WM'rf'aP&Tn(&v'q=('^(Vf'w;(<y'mB(*='Ig'E2'r9(<H'ZB(QK('#'?v(>H'#('_O'jm'bu':T'T<(&J'o^'H((&K'lB'Nu'?P'`+'m0'MU'm}(#A'IU(+R'iC([)'zT(Kn'F>'g3'{s'ZR'Gz'z-()l'S~(VG'e;'<o'rw'PV'P-'Z#'=@'*G'fL'Eh'^G'sT()*'U}'hg'YY'C/'l:'^J'4('Tb(0l'P]'QV'i;'^*'pY'f4"
"'5k'{d'rF'Wv':g'Jq'V8'`e'~2''6(K;'T8'W9('8(-V'uO(./(<`']/'y5'_J'(o'(h(j!(P#']{(*R(Mh(*h'rO'V6'k)'bC(&7'cU'jR'cZ'hK(@#'^6'w_'._'E*'{Q'U-'vi'zp&d#&RZ'zP'y!'d<'k#&c}'W,($W'UW'-:']e(-E'h8'i#'kD'ue'[b'{u'hG'vW'vu'th'*I(X3'^/'YH'eC(#4'tL(UY(C6(QU(3<'jn'R,'2]'w0'Uf&ek(5I&m_'Xn'-~'`?'pt'd='z.($3'dH'fC'TV'`D"
"'?f'G2(&{'Z~'oD(E'']0'_V(*~(0h'rA'-a'*t'p<(&-'rW'^X'}Y'wO'+I'ch'ZT'sX'=s'h7'i/'h!'LS'71'<f'sB(*n(+E'V>(&'(3d((E&~3'w+'?='c^'x.'xx&V@'Vj'f/'wb'.8'|a(0P'{{'WV(*l(z1'yF';7(?9(/G'nK'Fa'x9'19'x{'Ux'rk'Z-(:Y'AX(Tp(<w(a5&`-'}z(1^'m;($b&H!'vt())'qn's2'/-&`?(G_'`2'4Z'>:(6s(bM(@j'r((qr(b8'EL'ZR'F,'dv'wk'k7&wM"
"'@1&tn'2R'2M'qg'Ez'h{'{?'g^'pd'rI'bl'OA(6D'f}'VX'K~']F'b5'`X'F1']3'PA'[e'cZ'],'P4'M`'kH'Zp'`E't:'dk'a.'bf'k>'m8'jO'hT'mV'n4'ir'i2'fR'iI(.9(p9'pN(/j('M'S['{5't:'t@(2Y(,'(+y&/c':7'Qh(6Y'v6'iz&e)'rP'9e(1p('x'Q#(74():'-v'Y7&8U(+d'N)&kg'OQ'5,&rS&tf'E6(.-'`C':W'dF'{5&Aw'Br'3)'xV(K['6Z'nU'Li'QI'#2'x&'LW&ju"
"'tE'ly'Nz'eE(+L'|M(&.'>&'F2(Jn'=;'iD'|-'{l(gU'05('P(#F'Hc'T`'3k'q`'$t'p7'T~&}V'NG(Ej($[&vB(eL':p'hE(Qb'pk()S(ZS'}$&pc(4i'TT'H.&qH(W''UV(hG(R2&{Z(]x'Do&gq'VQ'Nr'u^&hw']O'}g($@(]F(9k'YP'Gi'm^'Z9&Ru'`W(OR(Cp&p3'1s(7i'?2(`C(,`'76'A`'t8'(K'vx(KR'Y$(AH(kU()$'fF(IU'`s''?'S8((}'Uq'XC&4H&HE'u('RH&]z'q;&h}'~y"
"'3B(;0(!<'dT(.U'f2(3p'q{(=/'_3(>6(5`'Hd(4:'JA'J}'Uq(#0'cF'~y(0$'+#'yh'Y('Lx'wW(Uh',/(+{'@2'pL')l(#L'`e&bj'yA'J*&:<'#:&qD'}j's](#Q(*X(/6'(b'd[([g'/&'ns(B-'6^'1]'_o'v@(_j'u[(2['RW'kr(CN$aS'z.'qG'!8'[4(=9(.7(o_'m>'j8'YP'm['nz'tI'`_'aQ'a='Z]'Ti'Y.'|v'aS'D~'g/(2?(1S&}S(2z'-8'QW'C0'q{'wl'=!(7e'8F'p>'|,($&"
"'r4(?v(+G'}-'L|'W!'t#(N<'aP'd#(+r&x9'kB'kG'CZ'@2'xA(E^'LT(mV'7m(.^(c7(2D'2{((o'Z''8G(2('cb'VL(cq'mN'XN((e(fL(Rp(27(@m(Gc(&V'LP(jI'$/(+O(#6'a6(QD'd!(5O'rK'7<'1x&}6&|>'5B(/^':>(#f(*c'eL'|v(6^'qc'a.'$U'Jk'B-'RN'[>'H/'WH'UX'2_'Na('H'a''cC'g.'F}'P0'By'KE(<t&Wx'@Y(*3'm2().&^e(Ku'`7'O8'Us'vR(UZ'Sd(6E(52&~W"
"'dy'KY'^e'D{'~$(9'(<K(-1'av'g<'v3'e*(#B()<(!R&mG'h@&ls'6X'qL'|@'}Q'u6('g'B0'r~'sd'[9'za&,y(*=(46'`e($3(76'dQ'fo'y<(0k':-(O('dR'cH'rC&uA&mA(C:(nQ'cr(VD(!d(XS'nP'wB'|-'P`'N0'fB(FK'sP']9'wl(Sz'@8(Hi'Wy'f7'IX'N]&fK'GJ(0Y'De'w`'tR(,I(6v'Id'a,'nb&s`(#|'ls'h='YL'i{(0U'B3'Xq'tt'qe'w!'pg'X{'/f'ss(0p'mK'.F(+s"
"'pM(T#(>b'5q'gX'v@(#]()K'E)'u)'*R(Lc'u*(#T(Bk'_('gE'r('FQ(3e'S}'|L(1U(=E'or(a9)$y'i$'l+(R](-B'Oc'RO'c#&_`'g?'lO'wo'Y`']>'z|']m'go'IX's{'hD'9(';A'f0(!c'@V'sZ'W4'Os'E7'GV'WS'`?'_t'rw'aR'u@'ux'n9'lG'TL'p9'^q'me'j$'jq'n&'m4'iI'Xk'rj'9I&ke'T<'gG(|}(1W'_U'WP&h`(Fc(&H(Xi'<]'g,(>o'j0'zf(@b&~e'9d(0u'}S(ki(!]"
"'Z'(&+(O2'qo&}+(7h'rI'2o'Cn(9z'1m'n3(D,'KI'@<(4O'y?&$x(+h'AF'>@(6V'l'(9j'q`'<='~h'uF'Y3'fv'J1(/y'G''2E'nl'wt'TS'q+&y6'Z[()l'J0'S9'a^':{(!9'wE'm='c0((7('v(+a'C)(.1'k*'1q(2T(_z(59'`t&ba'0d&Te';W(,6(9((#i&<l'V&'5s'gy($W().'N3(QM(,B'y0(Z7(z-'K3&m,(){'>P)+f(V*'vs'ci(zd'k#&lh'Os&un(XO'3F()('pU&y{(S4'}I'rw"
"&Ov&Dg'#!'c:(;,(Eq'ZZ't,&f-(yW(2Q&An'rH'/K'>^'{5'{,(IS(O!(0$'Mx'uy&T*(r!(6r(&$'n$(q6'o*'_`&er(>8(3b(:3'>?(10&YT'MN&xK&H}'vR'U,&Xp'TS'jN'9S'$q'tE(/U(,5'R>'PM&io'?''uN'vw'LC'sA'We'du'[n'~8'lx($N'j0'})'yd'Ej'iM'lV((9'K^'yH'|:'~-'Zj'ZJ'W](/E('+'^u&&<';?(;.(BD'JG(I-'ic'WK')l(*`'Q6'9K(A^'._'[o'xz&bE(Ts(5?"
"'Zv';{((x'Eg'pq't1'3E'SM'N&(Sc'z;'k4&rC'A$'9g'd7'~S(sE(Zl'_l(H_(4P(>K'nr)2G'3~(eL()D(4-)6&'Qa'[K'[R'MT'`j'P}'mr'VO'KL'A}'AW'f-'Y*'lF'6l'fT'{!'7U'fn'g~'rl(1:'yo('w'~i'8f'h)(Mr'uW(;#'V>'j;'vE'Du'Sq'uk'j$'Pd'N9($0'S'(;T'dM(&?'tc'ZE'ly'Kb'xP&~]((2'nr'u[&f5(CL'oe&_V'2/'}B&LF'z?'Li'aw'0v(,t'np'z4'd:'Xb'Ng"
"(^*'.R'!V&E/'}b(&3(0y'qW'eD'`w'pU(3O'zS'w0't3(.k''#'c=&ug'E>'az&gH(4_'`{'cr'ou'[{'o/'Ua(@:'}P'zG(KU'^('dI&aa'9>(U3'q.'oZ'X*'}/'sI']L'n4'Y:'o^'N2'MX'`Z(+6'k''t$'vF'*C(!A'j~'vv'^a'p1(/K'z8'Oi(?#(Rf'{<'nN(,a'Z9&wM'd_'9q'mL(C&'h,(3_(-d'iB($q(>7(?c(-((jv'h['jx(Ec(Fj(-H(<#(@|'}W(/j(!]'iQ'a/'~8&pA'cP'n5'3&"
"'ZR(DN'9D'qt'l+'z/'t}&sF'XE(Qo(BQ'<g'}t'r['Vo'@F'l''kM('s'e)'zB'af'g^'X>'PW'zz'u7(;/'HH((9(')'^['uy(?I&RD'e)'u&'Xg'ei'c;'VZ(!U'p`'QL'_]'m&(;A';T'[-'D1(+b'Cf'm.'jo'f^'`6'kn'TA'k;(8T'x8'-x'u8'~!'at'T['sF'e?'O6(+r'bc(H^'}<((3'z-'Yo'/y'V{'~j'xd(2|'AI'7['vt(=[(-$'hk'dW'`t'jJ(#+'FS'`4'N#'-S&sh'ss'cE(H7'{t"
"'gF'9)'h,'<<'N/'RR'WP'b}'7!'Vw'hQ'lp'Tv'Ic'aR'V6'Zp'R)'dZ'pP'nJ'`|'gO'tM'Xd'g^'sI'xh'jl'wb'b('ne'kk'q8'iI(<}(<!'wR(=,(/R(.O(Qx&U|(P0(du(/`'GK(G6(4o(^p'Z{&h'(;|'>H&2='cy'*]([<(o@(4)'@W&TA(xz&Fl'rL'=L(J$'vY'Oa&uW(_G&n|'WQ'lD'=(&hp&Z+'VK'J'(rY(:F'^?(6k'B,'Z8'n#(7A(83(/,'04'#x'3`'NC'eB'|F'z$'8t'Ev(Fg'X2"
"'><'JM'1c'ZP(#&'z+'`F'OW'XU'fE(,Q'fJ()d(F6(#y'7-'ft'a='~V'Gh'xV'D[']W'KT(F/'ed(+H'I;(+T&T~'qh&T,'UB'sR(Pg(./&GH'9n'kC'j''79'Ua(!0'=9's$(@i'u+(hA'2b((g'{4&et'oC'db'<d(fy(_a(0G&Ca&.e'oO(I4(XL'Q9'U*(rI&1W(!0'Pg'Q((e|&la&+V'n,'eY'u''?H&Rg'Og'3@&@B'N?'Lh(.A(Z^'R.(&0&pg()T(^x'T$(fX(91(Q7(1)'54(q~'P_'7U'U*"
"&}&'Yx'T|(+p($w)'t(E=&|b'`}(,|'e&'Z`('Q'x](L'(=#(jr'O8&QU(d?(xt&mt'T{'OI(?7&h?'fz(9|&`A'`d'l|&}3&}5'xB(HZ(91'>Q(:-'SW'sS'|_&pG(w3(5S'lE'6i(79'H>'jh'Dg'QX'y+'[5(+C(@9'ZZ&zT'M4(4C']s'[w'U('v_'2a'R=($&(X4(G*'c^&rC'9('vf'<Q']K'3x(#{'Vl'H3(#`'E6'B0'Lx'Dr'U]'OO(7=&vB'oN'`D'L('41'}+(9/'a0(!*',n'{+'Qx'=p(3E"
"'@&'K=('5't6&jo'_A'22(9g(CA(kX'vE'1z(E5(C8'&|(=$'$e'um(61'l{'@Y&o,'aH&CP$q>'_{(Bs(S)'ae'vO(dU's*'rS&Og$b0(;<(OQ(N/'?+'R^(Zy&[E'Q''6b'y<'&&'_<&jZ&`~(T['WU(a-'wP')o'nO'P.'e;'OZ'D*'a;'ZR'Ez'r('H:'{!'bM'6j'7Q'fZ'~w'O:&A`(DQ&bc'a~(#J'3w'Di'v_(/B'|Q'Lb'Ed(*@'Z['I]()T'YJ'IK']H'z9'@c'bz(3Y'iB&_a'k`']p'2:'(V"
"'O]'h;'F^'?;'bs(S[(W<'mA'H8'/i'XX'c)'Se&hk'K!'t5(h]'`:(V7(*N'@/'{r'a,'[/'@;';,'X$'g/'S5'jG('u'QO'V<'EI'/7'go(-}',#(<q':#(N-'L='Ek(!#'wg'4u'##(M3'kW&yz'<Q'uj((j'g}(]A'Ld(9_'Wg&mj'Xz':-'Db'7C((v'vS'?s'v+'Ah'Vi(a7'j~'Vb')S(Ys'U*(6v'm&'5E'^*&`a(E@(l2(_C'4/'Sh(/p'FB'l-'_v(G''vN&jX'bv'/2'OZ'y9'[i'k;'z='k("
"(1?&Qp'T]'Zr(7z((l&x&'fV'P.(.;'Q!'bO'Cn'~|'&!(6S'T<'II'1T'oL'`<&vC&m~'~&'yG(#L'cJ'tw('r(-b&v,'ai'v8(+f'~-'E>'oL'r*(C?((S'Tz'pG'^.(!R'F-&|l'LK'6i'FQ(/b('d($^'>]'?+'`;'=6'gH'KX'$Y'hO(2^(*y'I`(V>(CN'G)'aq(67'H3(!o'Ql'ys'w8(6.'*D'HK'nn'Vf'd7'~6'_8'q}'L9'S^&p;'P.'e1'RW'YB&]q'e#(Gn'*j'Pj(>3&iM(^o'hz'fj'Ke"
"(,~'M*'CX'[R'bY((t'JJ(,x'EK'b['Q>'Kn'jp(DN&HQ'g$(.4'da'~/';x'ry'n['L^';K&z='b0(&K'xo'bL'SK'mI'Rg'HS'qN'wO'@H'[='J~'df'By'W_'F6'X@'^5'bf'x2'r0'iY'b-'qZ'{Y'b:'j}'bb'=d'em'w@'tx'k`'iI)#e&V&'^.),;(1V(.m(65'&R)&p$`C(Qd'`7((z$k/'C_'rN':h'Y~&K/'1R'Vv))J&ym'Xa'[}((X'[''hZ''*(_K'i&'+o&Kw(7L'D?'e<'D*'VN'A^';e"
"'X}'nI(?V(F1'm6'|w(-W(5a&{/(#Y(8H(:}'S1&sj'HQ'W?(2G'Sf'S{'(W'Rz'vq(:-'Hu'ex'h.(?Y'=9&et'1|'&>'nW&*Z'ue(um'4a'IZ(HH&p*'^g&}(&N4'gx)!H'!=(Ah&xn'5f(.w'A-'Qr'`!&Q{(Fu&o.'M4'KU'_m(:j'dV(<t'g|'O.'T4'o0(F,'w}&n+'gE'5n'S}&F4'zI'~T'P`'~H'2f(F5'60',r(1T'i4'Kp(&H'ra'@T'6o'}e'Eb(3[''('kG($S'8K'Cv'fa'{L(H['mK(5a"
"'nS':['?Y'`H'vS'hn(5$(P&'IS(!W'Y.'dn&V7'Uq'P8'`;'_0(6>&l['|h&Qx$7''s]'4u&X1'q#&u|(Ba'Ev(/[(Wl(Xr'A9'NS'^d'Ak&`9(}6&G0'PY(S''kU(<g(<U'4/'c$(*((5L'.#'q/(EY'9,&*W'U9&&;'EN'4b'aJ'{L&YU'$o'h~'`k((2(IY'x+'u#(H7(]I'D$&q}'kN'nD'f`(3)'Tw'iT'is'Mv'{6(MN'm~'{I'=~(3m(wl(Ac'Ns(LV($r'uX&6`&x|'$Y'XX'-;'x/&kM'C}(Lz"
"'C<()-'tI'zj(-#(*6']e(vw'YA'TL'oW'[S'pV&_]'Mn'Qi'uM'eK'Q,($t'=}'i0'#'(6+'@3'`i(4I'QI'8b'S5'SW(Nr(/2'7K'v''Bj'f{']a',k'>U'Z{'km(Tq'Xg(?l'2O'pN'J4'n6(60'{?'nB'6I('F'F`'k+$zj'9x&b|&~7'x`'r8(Qw'JT'Zh'*>'vG'!f',r($E(Gp(CP'`5'8q'f$')E'4m'xK'Q:'{8'sl'KU'RK'Yz'Z>'3x'iK'ji'S}'it'O&'CJ'^y'Hq'MV'_C(;J'qU'L;'Mn"
"(!h'r5'(o'pT'eM'bH'l:'lu'SR&{g'*D'|?(<+'m@'Q&'4@'b+']='A?'U9'M2'Jw'yd'qs'FW'KW'mM'Sa'HB'gC'^r'ON'y+'yX'i2'tV'{v(!j'N1'vG(-N'hU'Y6'iQ'BB'fH't?'[:'s1'`}'Q{'P)'gr(8G(4~'X^&@|($K'xN'S/(kd'u-'N6''D(&u(&s'JL('^'pw'T('p#(._&y)'ip'Vg($~'Hw'iP't('B-'[2'X$&v4'kh'{.'#c(AM']Q(Vg'N5'z,'i6(.d'lF'T4'To(&+'w<'^W(+x"
"'wP'v}'0Z'y]'*H&xE'js'|P(E='[_'tO'4_'qV'sU'zE'o4'gh'Y?&4Y(#&'[b'Vb'y['ej'cH'v`(Ct'a/(26'|Z'lf'ZJ'ry(5t&~l((E']](D5('|(,!'6?'fJ'fe(#A'yi'x5'j7'fo'pu'CK'HU'p)(!0(*K'P.'z~'g{'j3'|J&vv'w6'V;'U0'Sp'os'Fg(O`'XC(E$'mX'@+'Z:'E/'fP(1/'l1(9Z(,U'}T'W@(:K&SD'D|'xT'F+(Y`'P^'=S'|j'Us'yf(>v'O#'i`&XL'hk'gy&gX(({'ea"
"&rU'h_'Fp'B5'wE'k5'_y(1D'W#'f#'G+'lL'MQ(!A&or's9'dt(KP(C6(+f(8?'|3'ro')e((G'yi(5{(1x'y!'Hm'wh'vW'EN'@='[Q'V^(2R'Wy':z'qX'~r(!&(E.(+e'wA']['|>'pd']?'/S'~A(&`'ti'L^'4)'@N'7]'FA'V='G_'D}'fk'E{'F!'f#'ok'IJ'?H'q6'h&'fv'`F'jU'jq'nk'hT'j7'e@'w_'m?'^x'_-'bs'fY'iI(cQ(]4(RM'GN(DS'sF)*F'I@(YT'+=&Ly(H''|{'1q(j/"
"'w_'eE'Zd'm&'{>'-L'_1&_9&5d(6f&PJ'n+'eG'r5'm8'Ix'ZS($)':{'Q6($W(Bt&kV'r['ob(4q'8~(n@(:v(!z(&b'&>&|p'],'EV(*$(Et'iY(0;'lf'Y<'!D(2G&sy(<((U.'XK&f_'UM(U*'W~&z-'cx(Q?(HA'|8(1Z'Y}'~f';I'B7(&{(@<'p^(?b(A5(R!(FZ(Dd'hL'gq(|a'LX'l}&B{(8|)<](dm'E+(^6'|G'g~(8Y'^U'v^'>d(8.'2l&Gv'Qz'Z8(6v'X='T['TE'p5(0f'ix&zv'?j"
"('u'A='SU'&n'q`&qg(@f'H1'Vw'A*'pj'd)&tP'~:'lC'(Z'|h($L&rq'aQ('(&^x(:|(6I'-r'Xt'ZN&u~'-X'E8(#V'D-'BE's=(Ap'd0())&e0'xj'rg&ut(7R'Q='tM(*n&9v'du'Au(.f&]j(iO&o1(K1(3&&Ng(7<'y8'dD(<d'@r'~w(Hx'Xn'av'}2(Od(Q[&:2&qK&l1'OJ'{^(9|($L'i7(A_',q'~Q(AB'0*'rc&~9('Q(7X'=w&A]&qF(/t'Z+')W&tj(2''_)&t5'[L(U='hK(B.(2t'd;"
"(6R'If'Y@'L_(!|(kS'Q+&s`(]/'sF(0p'.B'L]'.j'jL'~c'7='Tq';$'ei')|(8w(5b&_c(6;(2D()M&on)9.'uh(Fq&lE(Te'X8((/'dS'n~'j)'GW&mJ&~2(Sc'v3'{}'Kr'fJ'cb'?r'9T(.D'^T'nE'^<'q/(1x'yu'Eb'GL&P&'BB''^(&?'z~(Sv'B!'YO'Lc'wB'yd'i(('H(&*'-R'l0'0L(F{'Sy'py'U$'U0'XF(#R'X/'G5(8Y&|>&H+'kI'2^'|@(3G'i6'N*()~'@O'o0&Ub(3g((X'.d"
"&dd(0t(1h'ta'g.'U$'gM'om'g2'U4'su'sj's3'~}'dy'dT'I7'DL'X-'tD'ee'I/'jr']d'Aa'Iv&_+'z5'Oj($V(<c'[>'hx'ng'^C'I`'a=(GR(C$']4'(b(1E'i^($Y(0Y(&W'sR(6X&`_(P*(/!'n_'kI&n('fR'q@'BS'b/'h)'~S'(-'~F'-Y(/D(JL'pS'o7&ch&pn()C'^s'ds'kG'`3(*>'Tf(,^(<i'Y2'r+(@|'MI'YI(3J'2?'[^'CX'a]'j''Sf'>K'wA&k~'Q0'#H']:'cU'gB'bq'&3"
"'7V(>S'Fv'T.'GE'g}'D$'Ef'hB'}]'`s'lH'qB'^Z'u9&}7(4D'Og(3+'Z*(=q'q.(M~'*~'u~(IX''e'rd(#:'-)'c/'P[&x0(Sd'_&'tF'Ed'bf&~x'[!'ZH&tn'9(($V'|5&}G($?'~a(!-'.P'Wf'WH'Kt'_['lg'h$'^R(A?'SL'c8&jW(5Z'pM'Vd'h4(+r't4'^A(6x'^l'3`'li'`@'2#&b0'uk&wF(+F'gO'iB'T('pS'yX'aU()^'hj'V2&eM&x3'<E&uN'u-'Xr&|D'dE(#4'x4'~e'^C'Vv"
"']~'xn(<C$h<'Z,(2='P9'WG(;<(V&'lY'u=(0S'N|'m}'{g'P}'?;'Y((0>(1}'io'~{'L=(I?&qF'[,&Fp'A]'j7'v)'x[((y(#K'?j(=-'l|(IV'pQ'|x'qk'`:'{Z'+5(]p'1.'Uh((^'pP'cW'gE'rw(3K(3b'mB'|d'a`'U]'v#(5O&BT(3_'?V'e~'Ya'X]'1O'vB(Ik&mB&d4'O*(jI'6+'I+':3',-'s$'Q<'T]'=2'J3'^h'W<'B5'S+'Zc'16'E$'eY'K''l7'VC'ae'qr'iH'ln'o1'pO'lB"
"'Zg'q@'x''jl's~'sU'l!'a3']C'iI'v<'|b'/L&Te&jZ)-@'Vg'4W&rW'oK(n/'i.'([$:B'T`'*q&?}(jc'BU'Qf&4|(!t&u=&m:',Y'uT'j5'25(X*(es'?!&TS(!{&_6'X9's5(?s'~|&m{&s?'s|&Y/'!$(!.(.U'l9'iU'Uh'DU'J<((r'cy'wx'KA&vp'EJ(3>'XF(3('f3(,_'VW(3,&vy(Vm(Xa'Mv(,/(6P(5T'y.(Li&?u'IL(&N(2F'6V'H>'5`'Z&'vF'jY(CB'=D'S]('C&J!'q?'3B'ry"
"'O{'BJ(+)'j@''A'3.'rw'L4'kd(2A(4-&yA'.X&}y'H('*9()u(&0&<H']`'SS')R'r5(d~'!F(/*&q](2;')R(4''yc'.c(J1'Y<'#M'<O&^G($8'x$'tt&7>(*9'M4'Ut&VF'Ja'u)(.E'5r(*>'5^'YS&H*&s*&n~($~'UA(,t'<K'c6':R'rN'|~&p9'AH(&J'pb'V-'Az&Hk&{t(R6(3S$47&9!'gw'ka('&&cW'F.(/j(M7((6'w6(1h'>>(S7&Gj(Z:(X:'IV'pQ'?6(@_'}q(6f'MT'4|(?y(ci"
"'H.&|6'ij'u3'sI'bf'${'a|'t|'n#'Z:&Y@&pA&gi'b`'34'W=(+T'X]'Qm'aG'Q}'_/(>b'fx'x](5G'tN')H'}`'Q|';b'l3(+Z'Jh(8e(O<(8^(KO'g;'o9(;Z'#W(.>'k/&xp(,h(SP'}|(!#'@P'|;(HT'd'(0o(/j'=a((G(=c'Q*'_3&o:(!k(*H'rS'r{(06'Xy&y6(*:'P/&Ub'SD'vU'U1(/?'F('&I'.3'Vm(0j'es'kQ(:l'5x'`c(14(80(9^'uF'i2($T'9t&a2(3|(bB'XS'l_(9T(!x"
"'bG)#T(+H'Z:(=b&pd'va'u6'rT'W.'do'Vj'Uj'yy&fU'|*'tr'Oh'~='b,'c7'.L'RK'jK'Ho'Eo']$(#<(=^'=H'_*'C''Zz'XU'Pt'o3'as'?H'vZ'gY's*'_T'ki'Dg'fB'yf(Qc&3](4.'o]'eZ(O?'fS'=m(Z$(1Y'q;'rl'em(->'qR'z4'L3'?#'#1'N$'IB'Kl'{h(1W'se(-&'bq'N)'r/',`(JG(1w('/&e6']f($I(=Y'^$&u8'D=(*W'}~'qj(2](&g(.V'wU'{t(0#($g(8o'mw&dX'dR"
"'R@(#d'RT'I<'dv'K1'gL'H*&}X'LI'hH'&G(:3'x#($R'vA&x('r~'kN'y='lh'SP'71'R}&w`'(N'z/(J5'Y1'i<(GC'nw'|B&bX(+P'r}'l[(+I&pt'Xs(;~'R.(5x(5#(?w'kA(*x(9b'U]((M(V2'1A'yD(Hd'c_(H8(4h'5o'ua'~}(O;'lw'sK(n{(9>(Bx(+1'tA'o<'bo(1B(94'vU(Vq(,W']-'{('oo'r2'2T&cM'[`('|'rb'ki'po(?>'{V'dS&gS(Bw'B0'Rx'n6(IK('P'1~($G'Yk(Cd"
"'jE'}F'r:'t6'vq'5f'8.'3g&s*'l@'Fs'9/'jT(@H'=2'49'l$($j(^B'[P(V]'z6'FG'-g'[o'l*'/P'QZ'x1'RN'y8'i-'u;'+O(Bw(--'q!'mX'Vb'zE':*']>&ZB'e<'F<'5@'XT&ZP'm+(Rn(,-(5c't.(3O(B1':7(,U'tc'_C'h;'x~'_-'NK'zn'YG(M-(A8(3X'vk(|7'|](&i&ii'mv(+e'd1'PQ&y;'d@'nQ'pU'Rc'mp'`:'dp'cK(TH'FX(!&'y&((t(Br'!0'wo'~p'fv'p#'?8'VC'O2"
"(#z'0j'l='^:'AM'Ws'im';U(&,'Ao'4:'Ob'TG'BF'98'2_'6^'On'r^'l$'qa'li'qc'x['`C'_V'qw'k;'in'_#']z'bt('a'av'iI'eC(,t(#k'{m'S7'hs()T'LE(4{'D7'Yu&8R'+k'E!'hy'v^'+H(>`'}_&ui(NU(/>'NK&r-&zw$vq(6r'^`(:J'9A(8J(]t'bs(DP'}|'Q0(.0'2~'~r'RR'|Y'Ba(&-(HJ&^U'Q{'hb$Gz'Uf'NE'eH'Tz'#W(`H(&h(1_'<a'nU(&?(U;'y5'^.'X5&~W&vA"
"'n>(-l'No'Fi'k8']f'E}(Ln'tj(.*(U_'o?&yS'fZ'e,'6_((Q'{}'1K(2!'v$(@g'd5'M]'Id&F/'bn($@&X](KA'{g&Du([N(#R&Tk(5j'Pp'@C'e.(q>$WL'yZ'6/'fV'=p'@3&w,'nT(#w(4@'yT'fN'|w'rl'{c&i-'Fi(=['XU'il&G`(B&'aq'hd(Dh'q|&;f'v2']5'Rf(O<&hT(:@(1j';u(T_'vc'K!'vC'h6'uL(mm(]4(3v'h8&N6&RA&t!(*|'t>'YR&8:'YQ&Q+(P;'z>('?(*5(&n&U("
"'(m':`(+q&tN($h'LW'm`&v}'|?'.n'wy'De'`[&Su'O/'Wn'8f'y1(,(&v#(#9't3'bv(8R(.e'm^&+(([h'T<&{S'N7'}3'BY'in&{N'Sn'za'(e'c^'sJ'h<'oy'q8&YB'mD(G8'U}'0N'Cq'Z4'uL'dN(3:'!-'q_'rT'nm'8d(BB'|O'qh(Ql({W&i4'pR&p](D{&xP&x>'`p&z^'ud$vM'bV&aA'uS(G'(*A&wx'qD']8'@='W}(.u(]g'z|(3Z'NV(PQ'X+'Ed'g>'4i&~8'b8'=H'qI']P'au'^g"
"'n&'b$'cF'e@'{0'G>'K-(,y(.H(#`'EG(*`'B>'sj'7l'n^())(.{($f(:N(5r(4B']F'P7'v7'ii'l''Lk(K:'Z9'D&'?{'k7'aG'ap'sU((_&},']+(,?'rc'(A((2'kA(GF'[2'Y8&l^&f+'j&'Ch'}q(H5'Bh'U;'t3'wE'1k(@_'bQ'Y9'Y#'P)'+9'T:(#''mb(5l(&5(CW(1F'h0'J3(#!()W'kj(Fb(78')F(#8(I!(5,'S0'_}&M3(6v(IO']k(-M(EB&RF'sY&hC((I(BE(0L(/+'br&{I'1)"
"(;U';7'g&'Rr'y:'T<((;((^(,8'rF(={'n@'kl(3j'O6'c<';;'/.'wI'>f&sU$~{'qa'h|'*4'W.'b((.e('7'rd(E='uK(-8'xx'kH'W/&[v(Y/(-.((w'dy($v('L'3V(3M'~#(1}&ou'hf&AK'Jw'c&'4)&x)&LM'Wu'o#'YB(B.(!*'^3'nO'#W(7e']_'^|'Y*'SG'o$(;V'H''~P(6e'iO'z*(K<'`,'}L&rP'@2(+i':('Zz'os'bb'a1'7Q'`D&vf'Us&sI'_('e+'q''G:'mZ'<t'?)'o#'g("
"$|](M&'9?'{}'vP(!)&Pj'61'qF(#h'Pd'sI&p!(Mk(6>(=v(5`'~F'jp($e(PA'uj'0X'DA'MX'r*'}}(.S&U<(?V'i4(,m'7r'h4(6:'e;'[8(1*'jX(PY'o#'o('c^'4@(F!(!+'gi'dK)-Y(5;'xe(B|'gX'/C'pv'lE'T#(+v'8j'wQ'#f(bS'qs'0P'2#'R]'9/'=l'>@'Ed'Ne'Z8'1V'F/'UV($)'4g'yg':b'jh'r^'b-'bo'g['c&'dm'el'dF'jR'hf'nS'v='sp'o-'iy'hk&q-'QP'wB'lw"
"'P`'m8'~F(C&'r{'mB'y#'yM(3u'M5'VS&JC'r/&i*((8'iD)4`&?f($n'W|&~('v:'uJ'D1'o6'q3('<'/+()<('v'c^&9p'gQ'?Q'{v'v^&{e(BP'rP(Tx'RY(+'(*b&C9(/<'xg(/:'~h'eh('S'e&'$D'#:(3D'Nr&~o()6'x7'KJ&Hk'yL(,9(*c(&8'V]'js&q@&ve&~z'}h(-G(?q(15'@e&$e'SF'ik'{z'#c'T5'Z<'Tx&)z'>H'H4'by(/!'G+(`4($u'S}'vY(?n'2P':w(=y'Vh&vN'r)'90"
"(]?(H9'_|((:(hN'TN)$o(3?(AR'mg'ZN'us('Q'~c&m?'eC(BI(+>'Or'f!($#'U>'Ag(@W&wV&pk'y;'dd'^m'`q'J,'uc'R,'B|']|(!l'sp(H='W['qb(B]&L^(#-'eo'*#'gC'^<'<c'XF$xI&gx'6j'h?'<*'bf'WK'rr&ws'_d&YN(72(9&((R'sL()_'11'r8'5!'_g'@@'kO'*v'SV'iy'k*&y['ww'sE'u@'Yj&~''S^'S''bg()9'{k';w(C~'W}'xF&@z'^k'PP'v@(1K&Sy'N6'lW&zt'b|"
"'Xe'kp'a1'Y6'/Y'UA&2f'T@'5+'t(&cD'n['?V'Z&',w'K7'N'&&.(,5'Qw(.f'5N';-'aV'RQ'Oa&}s'AA(G['g/(NQ&uF$j&&fY'Uc'{g(1|(0/'8G(5S(34(+='kb'Yk',w'h1'5,'}+'5D'M}'dh'az'BQ(#)(//'s!'?q'cP(,1(03'^a'lm'4~'Sh'WV'k.';I(,&'J''z0'yt'l}'/q('k'x+&^$'fS'n}'pP'iV'/t(0<'W8's?(N7'9r'u('OF(/+'g-('0's3(3M(,m'iG'^R(q;'{C&vA'j:"
"(B.']-'}_(!i'qI'zP'c''*|'Sx'ZU'jq'z>(!^(+2'O|'iP(]i'Xo'jS(*v'r<'bo(V?'Y-'z_'e7'kq(#L'_+'dI(F.'oi(5u'+e't&'u<'8Z'pU'TF'rI'ds'zU'sv's#'SB'pF&ic'e&'QN'y&(<#(Bq'Ll'+A'YH'd#'~T(=C(&5't!&`s(2;&~7(.v'@a((y'ey'aA'y((@2':z(3a(!s(7~'z['}5(7s'h+'`0'lb&^#(OW'vN(-!(A8(OJ(SD(CS'j3(*j'}''G2'[b(7I(?:(#p'5#'Sb(#b'~F"
"(|W(m|(6C'GA'g*(<i'yG'Ge';&'w8)*4'bF'}D'6G'k^'_s(2[&eK'ke'[!'c?'kr'^H'U5'GR'Zp'I3'cX'@a'@8'}O(/d(B$'FZ(/Q'Ds(3Y'~>'sZ(,S'^Z'Cl((E(:T((K'>O'}0'|-&Jn'm[('B'pU(&W'dS'nX'sH'a'(+Q'cB'xe'vq'@y'jD'n^'DM'lY'je'5''YK'lg'k='kb'kP'UK&l#'mS(A@'V7'wy()G(35'-.'uK']'(,-(!g's|'jT(>M'*+(km'po(5['7M('9(,j(8[(Q~'Zd&mB"
"'x,(87(.)'yI(4W'b6(>@($='?U(=m'vf'|Y(K0'g;'G''Tr'Yv'm;'z-'a@'Sx(&>'QE'~2'l*()q'!)&u;'FB'.P'6b(17':h')p'GU'G>'2B'9M'>@'66'bz'vR'jH'lc'p('m*'rO'q.'eZ'mH'jR'iL'c$'nw'tT'jD'hk'qV&T~(Eq&eI'Oy(:e'Dr'dP'd*'h:(@t'pJ's6(4h(,#'~d'j|(/@'vW&;P'l@(7~'0u'~''6B#_;'`a(!u'vz&w_'ct'vU(&G&~}'I7'l,(+T(Gk'}=(ui'-C&8B()X"
"&~^($a&~J']`'X6(59&R_'Iy'XM'&u'h^(!k(*P'p]'kM'kD'0-'`p'fv(/;'RO(2_&sd'b''y7'mr'I4'zv'OL'b''t9'|I'NO'fe'IY(3T'nE'pb&7m(T^'hg'F@&Dq'a3(#:'DW((>(.A&|M(3v':1'r3'ZI'=s&mk(/&'ls(T{&ZF(!u(9Q($x(.$&SH(!s($i'Ed'`A'iD&|U'cP()y'{5&|H'Sh'kN'wD(,D'}0&L>'2R'hZ(_9'jo'9g'bl'3-'B4&m/'S/(#Z'2(&u@(4I'a_(Jf'6O&{#(&~(WK"
"(&u']u'CI'oP(tE'^G'wD&/{&xk(5R&X5'fX'$e'Al'n2'}?'o+(/W'jF'}j'~{'^g'g#&Gu'|V'5X'bF(#X'}m'PK'-H'Op'wH(1z'nT'BP'LH'd_'hl'5('m4'&((C_'HZ'aE&Y3(*i'ex'tx'w4'C('Y)'gF'z+(;&'{o'Tj'97'jJ'Zy'[U$X8(Hw'D^'V[&MZ'GH'V>'jX'e-'3Z'`_'iC'i.'nG(/k'wN'2#'Q#&4X'c>&A7'tz'0Y'uK'2G''x(1:'RS'DH'p{(J!'Ym'Fk'Ub&^j'`6&xN'8*&z5"
"(8B&fl']J(&$'W4'nV'Wu&~P'I7(Ie(Oh'Mw'UI'k$'9Z'TU'RW'^t'l''m8'[Q';B'~&'p=':n'Ld'x6'd&'ER'2$'P0(V''|O(3b'uR'd2'`k'a7'qF'lk'7_(&&(3D(A&'0I'VD(As(SY(Hu(<Y(G~(T^'tn($.(,W's:(!A(*i(.=(1E'^K'R+(2G'S&'Pb&T='Py'w`'vn'zc'u3'yG']-'zM's7'nT'^1'hg'j>'YT(8.('4($D'z`'vU'p:($x'dg'ex'l9'{(&E>(<L'@d'r&(Gb'bx'jH($`(&!"
"(1c('!(3m'TT'Q6'T(((h(5N';P(A_(<u(ZH(,J'&1'I3'od('1'{j((V'ls'kx(>_'U^'0W'i!'23(3|'7x&xg(H~(O6'{R(1;'}T'fc'~v'kI'@7(&e'jd&Ke'Li'ob'Lj'g=(!e'?9'q3'dU'iv'u|'S['c_'Dm'>+'qO'Wx'N.']<'XE(*b'z`(S{('$()}'}T'y*'Qn'kf(2:(?L&~&'pt'~Q'hx($`'0d(0>(:]'uV'}/'{<(*I'Ah'o!((a'jP'96(K|(>b'_/'*3'fs'6Q't@'m$($P'n3'je'b+"
"'k}'ti'^i'Bb'U='gc($N(;Y'j0'd/(Ml'd)(G7(0<(1g'mF(5+'j!(Jd(/='X5'kw'u4'}n&u<'a`($w'yg'm/'`='v/'lz'b('^Z'c#&wt'if'j,&|:(G~&<N((/'<c'|/'al'tB'zY'h#'*A'^*'j_&iE'}!'/!'Vf'/j&ve'q:'_#(#v'{5'm)'{;'zP'|n(O{'eg'S,'b/'e2'o7'Xn'p['dK((e'y?'r8'Yq'{e(LG(&c'q!(,`'gI'h@'u9'(o'6<'[U'NW'6<'@)'+p'D7'+{'Nj'T<'<m':$(&`"
"'xf't^'kg't&'rO'bu'`k'jh'jo'f_'i*'i.'iX'cA'in'b#'qo'hk'Ue'Z3'cQ&|E&sc'Yz(2q'X1&^u'#;&}?(.P(&A(F)'Qn(.e'b/'=m'pN&bG'QK()Y'QJ&}b'iN(G^'!p'X.'0:'^O'0z's+&A/&ap'f8'Vc'xG&|X'[Z'B!(+j'5.&wF'qW'2>'x''w4(;T'Hf&tm'g$'fU'tv(&c&X['rT'Aa(FO'M7(PX'RB'wW'+q'!L'dN'rL'5l'|5'k4(6G&+/'Bb(_!'D-'K{&_h'<Q'3/&x*'Vs(Nh(0Y"
"(PK'H~&kV()^($C(']&^R&|9'8|(!!(1w(.}'`,&lf'A@&Kj';+'}b(j`'K0'HE'D=&g''?x'V]'eA(G]'M.(-L'p$(/F&nu'51(B['B5&Xa(!^'|W'U&'VF(7G'q-&Ue(+w&a/'J}&~H'zg(>k'm>'}!'8P(=#'tY'p<($`'=p'+u';`&dg(Eq&@Y'go(2k'Tu(2E'vd'@N''_'d`(9+'Mk(5#&ef(?('0>&]R(,d'Cs'}v'_N'`g'At'o@'vR'FY'GQ'`i'jK'M_'I#'gI'Nn'U9'n}(+/'bU'rc(oo('_"
"(0/'~U'RK']Y(*[(9D'js'nj(&A'.f'aX'fG'st(b9'7|']u(!['fD(Zq&^;'{U'b8(9B(#}'=|(=v'rk((1't|&by'oW'w`'wa((8'sy'A:(<4'aA'P5'o='k3(?8'|2'kB'lT'>j(6+'v'(#3(&.'_<'Cl'(1'j#'`p'|L'h+'p2'vh'gr'wO(!m'V{'{u'pZ'Q?'<M'6P&rm(/6&p8'np'wX(H((&$'ya'lH'wJ&X6'}[(<w'Zi&wi(,3(>X(*7(Ed'wx(22&Y8'~Y'|n'zi'h-(-F(*2'lj'U~(-r'&^"
"&um(06'M8'hd'i|'h*'mA(Cv(+X'v:(#/(2G(#c'J5((K'ir'd#(Ih'u8'w^(Ml'Yt(63(F9(@l(.j'=E(2N(4|((i'x](Jo'$h(En'A+(+}'t>&zB'rd(5y'iL&=H&g#(>V'i&'d5'.:(9@'ft'nM(2X(8X'Kh'wr'i$(1U'z7'f;(9S(/L'x>'fr&~d'fn'qz(;{'eA'l](:B'PJ(AX'{^((T'gB&p/'{y($<'k#(5N(5K&Z6(*c'sk'ti(;J'`?'js(+8'mv(L.(FA(2F'O.(?}(5|&_7'*)'ln'Mg(#'"
"'6w'Dq'Ld'Zj'o3'@g'[7'b)'bE'{6&h~'s{'[p'zV'F1'j*'>l'#6's+((A'iO(Kt'RL((;(*8(36&S.'<?'B!'vH'nl('}'o((]2($g'f-'~E&Zb'O?'m_(#W'ye'J'(7T'{]'yj(>D(&,'|A'A9'<7'C['A*'HS'tp';y';R'DZ'XB'y+'O,'*('0F'8$'Fo'b='jZ'p&'lR'l3'rk'o_'kQ'j3'n''^t'bx'iw'gF'l*'ky't:'hk(8u(-6('V'cj'?a'q$'Oh(2~'th&Gx(<e'3:'V+'Z5'za&2o(0I"
"$]D'o<&Zx'ig(8*':Q(T=(07(TM'k;(#c'c6'CL(!/&}x't{&}P'Au'1f'v0'&B'uy'Sn'J9(00'UB'n|'fL(UG'18'K|&rV'tO&jc'^{'FM&?N'=L'IF'Eu'W6'a.'k['dz&Nu'i?'=R&gJ'YE&Sm'}R&rP'Jz'Cy'~!(.f'pL&w;'^x'~n($2(-7'l('$h(4j'M8'oc(1i'|u's`'uU(_l'P|'3I&]<'.6'xW&wK(MK'@<'1f('''YR(0T'Er(W1&sJ(=_&'o(Al'Yy'hW'Ge(2s'7r'v!'?&'s(&n^'zF"
"'{u't`(4P(>S&gq&e8(E-(UA'Ta('](OW'hl&iD'Uc(El'`O(07'o+(-b'qN'~u'ix'al'if'Z$'pd((*']F'Tz(U;'~@(J7&:S&S,(,T'^B'y9'RY(#l'?z'Uw'UV'HU''t(76'F='{D&>Z'sE'p*'^[(!!(4*(Z1(eZ($L'kd&5L'`1'Yu'sb'M3'WM(2R'W$&P}&}C(D6'Fh'Lb&|g(I0$yv&u+(Dg'k6'T)(#4'l['I]'qF'ix'S&'?&'ep'Vm'ML&tv'Ph'lM'W}'pj'bH'rb'IQ'}{(2*']F(,d(O}"
"(Jg((Z'[v&y:'*k's!'@R'w@'Oj'4>'D((3n'zL(DN(5*(=Z(v@(Vf'_N(9w'YH(m]'x:'Ge'_C'_|'sD'd:'i3'T2'h{'YH&RV'm?'hZ&nB',K'e)($*'X>(#)':f'wI'u'(5C&Jz('C'M>&{4'|/'rK'{['LC'}!(#a&q-'0O';l(8F('['n8((M(,h'Lp']N((V'V#&Mb'{)'v:(H>(I$(SG'.T'en(]Z't,(7G($=(mS'jf'nc'ym'iD'@N(+x(55(.v'tX(7_'rY'gU(7((Nw'|!(pY(A@(#((Ta&u`"
"'xW'|M'!e'fO'R7'i+'|!']k'^]'MF'P('xL&w-'ns'{1&b?(8='ps'Kh'GO')s(3Q'c$'hI']t'7['qq'O<'0i'uR'^D'o]'H,'lO'M@&x/'R>('/'j2'U~'Z4'_&'w@'`?'Ok'wC(#+'JM(AO(&F&hC'gl'Zm'i,'xh(26(&N'pe(6F(24'oT'cO'[3((N'.f'l|(71(/i(94';w'f>'la'ZZ'?/'VB&w&'23'sv'cb'Wi'Mt'eS(4v(0d'N*'@f'js'j,(Lf'ZI(7x'mV']<(5J(5M(F8'Zw't$'C}'O9"
"'V1(=((2F'k](Qs(&!(0]'_T'7u($S(3A&s4'}I(=B(]L&^-&si'rN'NQ'xm'eh'qq'p+(G&'9j'r1'Gb&_}(/j'yz'Y{(@2'kB'A)'[7'cd'oX';}'u7(2<'M*'it'QT'oQ'<K'ym'tl'^Q'^F'p='H0'&Q'=T(8}'/g'*#'{e'76'6H'X#(/>'1Z'Sa'nn't?'ba'g!'mJ'jg'`l'j*'kO'Y;'iW'j+'oe'`K'hb'ns'hk(9&'u&'W|(37('+'`]&O{(03'*R&o3&uO&H<')$'1v(_L'Zz']?(@<(72'AP"
"'z2(-[&G_(#=&m@'v6([j(Fs'@F(.;'pJ']C(A>&}j'Vi&Z8(7|&H''F/'x;(*c(Nb'uf'^b&~`'x@'Ls'x-'/2'FQ'O('ji(=O'(p'zT'>w'}&'#o(#8&gL'Y('m='TB'}Y(1]'am(Of'Tc']>'pt(25&R!'hH($('&m'dB's*'[](62(MX(?v(46'B0'OS&#f'w;'|G&Gw'i='Sf(#6'?8'[g'&M'Zc'Ur'ac'(t'{+()o&yf(&n'p7'iA'c''|Y(';'QK'PN)+4'[A&CC&UX(!m&w5(tF(/$':P'P*'|2"
"'CW'B*&<9(ny(>0'bz((D'ri'-Z(=2&dK&0j(R+(B_(.D'u)&:x&BR(-O'VT(1)'`3(0X']h'^~(A{'{N(0~'Xd&H{(4''Xt&M>'L]'tJ'Y7(/X'WA(8P((,'|W'u*'z='q4'n^(':'a['ho'C#'y,'K)'D1',V'V;'DN'eg(6Y'g`&`5'q}((t'Z)'q.(B$(1j'y+'jf'HB',E'l0'#?&t!&/c')Y'Jv'_4'AY'lj(=Z'&A(&v'hu&d6'w[(5t(.B&h}&:<'@L$QP&Q1':p'C,((:('Z(;?'~d'M9$lo'64"
"(A*&_P((x&J{'C^'Vj'U|'R!'i}'a:'{#'xW(#-'Df'M7'AX']X'FV'Vy's2(#I(!^'j!(Cu(>/'s~'x{(6<(&(&YQ'xy(!#'tQ'7y'}!'~+'f='R3':I'~a'^q'iX'}-'bS'vP&k0(!w'rb'm''/N(:z'Xr'yK'ym'p2()r'6R'p0'|.'z;'P='1V'kt't0(&g&j'(Gd(!C(-x'..(>0'k+'4~'qW'|u(-I'oI(3('y?'nL'nq'Bu'H='F4'w$&p2(?}&o=&`i'lY'ho'`t'^p&yq'rn'j^'m'(4.'zX'h^"
"(XE(!)(OM'Tr')&(FF(5W(C6(/J'}U(.;'yu(#-'p=&kP'9j'f;(2G(/0&p#'KN'}['zN's5'pa'tu't.('-'}X&V`''^'uA'|<&w['yU']x'zR'ls'm=(48'|U(7@'vi'vr'uZ'K~$yE'#+'~:&g{'+y'8((Xz's6('W't)'[f'{[(&S'rN'ws'kN'AV')S'u3&~&''J'5#'|1'_5'~='XO'[>'h4('t'fr'fP(1x(#I'}f('P($K'4X'?W(1p((T'}3'v!'g('7k(,t($,'~g&2(($~'dg'sL'px(Y,'k9"
"'C['aC'k7'}h('H';P((_'dp't<&5W'cO'kU'zu'`k'~i'JQ'*&'}u'gU(3,'2E&}V'p3'r_'k`(7N(A+'gm(&V$_|'_''Yd(?k'zC'g5'xg(-:(FS()d(1w'v3&{M('g'V$'`H(5e':#('N'wF'Tv'ZT'dc'De&8A'xW'yb'ag(?m(2U'3w'vG&yH&O9'CU'xU'je'u1(/`'xr&_E'gl'{1's?'WU'zD']j'#,'HY'i5'=Q'hj'=J',+'0v'1-'Eh&{+'2d'5f'I2'fw'J;'ki'q`'tL'a@'`l'ki'jq'k!"
"'jG'c('jA'lj'k''dw'aW'i6'hk&x5(4r&cW)'$'#_(<x(?d'=['LS'Q_)2c'zn(]a([M'<s'v6'o?(&M'hX'j1()o'{](C,'@_(7/'[U'Jd'O!'Rl(&F)#&(b{'a1'IS'lF'$a&Bl'}b'{C'H~'uL'+i'|[(H*'yQ&}&'+{'r((-F'Y,($:'Au'K)((A'j3(/t'r'(=w'/3(&e'j<'y`'V[&Mm'xT';Y((t()6($2'Tb&}A(-`(&c'bR'm.$TM'KT'Xh'Ml'_P'o`'mm&&J()v'Wt&dJ&=H'C6()9'gf'jy"
"&q_'@v'uw'rO'pz'm$'JJ(R[&s?(9k&<L'K_';c',?(ck(!R'y;(19(+<(@?(4?(#~(-N'di&j9(d3)9L&|6(S'&qo$wL(VD'^D(Pv'u>&w:'?h(2X&vH'ol'_3(3I(@}(&i$xO(Ay(V^&y(&o3&j?'z*&b;'Sr&i`(ov',P'E{'}N&?u(4W(=8'{j'Z2&p-(q:'rE'S$'x?'2(&[N'i7'j='HZ&Va(->(U@(J_'ug&k,(9c'D/'VB&zS(1''0!'(D&-i(1:'gX(>p'Sb'~0(!1'X$'~4'UP(!'&r!'8W'[U"
"'^0&dz'JV&Yo'vy'y[(+y'm3(.7'lX't='k4'a=':1(0e'O;'|p'r_'rN&nl'FI'k0'??'B<(.5&j>(/I((v&xQ(:)'[f('I&_s(*b(L:(2J(sl'<0'C`'w{'c$&{x&G>'m`(+S$^4&'p&|v'&k'o2#`m'|U&g_(=Y'e^(;s&q')jr&8G'w5'Z-$Gp$tU#wn(0g'[_'>1'h['k6'q0'Ul'`U'gF'cR'_8'@g'c}'<z'SM'P>'D-'FF(,''u#(:F(B:'lR'rw'ba'_;'zC&6:'&,'qJ'PY'(F('~(5I'ep'RS"
"'uA'Wp'}K($N(Q)&fl([h'Ee(Vo(#}'R.(@d'u6'[P(vn'Y((X.'Lo'v.'m>'A;'ZY(U/'oL'HZ'lH'#K&em'o`((v'na(!+'{Q($h'h-'k3(Zh&|V'b{'xu((-'v:(*k'5n'>!&m('*#'f&'B}'JM'Z}'kT(Dy'ts'Y,'(R'ZD'o|(53(;d'xq'_K'jG'k5(6A'oO(,0'i:']h'2P'WH'Kq'lx'n~(Br(>}'7:(!s':d'`*';G'U!'[H'g#&b:'fW'{^&GM'&6'fC(+^'f9'^q'0q'}@'}c'W.'r1'kc'oq"
"'Z&&=M'8|'7x(4t's[((K'm{'sk'dn(!F'x-&{:'l@'z)'mN';;'r9&SC'x8&c_'ra'Nh'Y.'p>'~^(``'wH'hL'{@'n&'mZ'Yo'_h(53(*B&is'e|'D-&}>'`C)&o'@`'^R((t'oQ'[='SF'!;'U>'ln'J/(5#'lf'_5'I]'xS'd;'[x'xw'MR'xO(#N'id&{.'`I(R~'/q'&V'g?(X/(Xs'z-(:1'VX(#Z'0/'fb'sY'gy&n<&7/(mg'{N'_e'b6'7t'^A'x((Cf(2r'h2'~B'Z4'2_'t|'Sg'-X&wS(>j"
"(.('k='[k(1s'{Q'U=([]'yN(Sb'{='jP'o8'^V&sW(Ce(cE'^-'d~'IC&j+'}K'M:'mM'|e(A>(+Q'tt(#*&|/(,D'5J()r&j}'|0']x&yf(3N'L&'={'7r'.2'6p'Tm'Oc'i(($q'@Q'ze'Rg'G('i:'K&'GD'qk'ao'j?'__'oN'n{'jg'w)'^+'bX'^0'r('j-'r4'mh'hk'hk'Z&'}j'wg(37&8Q(7/'~K(^v(Fn(Iy'y+((#'WX(4A(2Q&U[(O}'{m(ec(w/&,1'~o(#W&:o&-='K$(e[({}(.S'Q`"
"'@v(CD(m#'fk$'u&x'&6E&r0'@G'Pi(#i'j0'41'?/'5C'^6(O_'4a'Is'mN(@~(Cy&)^'F*'HU'ri(Km(,b'e*'C@'ek'k8&Bl'bJ'lO'iX('{'v^&}/'v`&}O(T/&}d'JZ'T='Ei'al'zH'RQ'K7'Bz(7]&i_(&P'ot'pn(7D&]Q(6B(;9'p(&YY(8B'rz(,O(*4'Z1'V7(*n'q<'ZX'IL(O*(30'k5'*X'Hp&`i'Qv(Sx'Mk'[p'S:'gQ'XR'T$&fI(P&&}z'Pv$PJ'r>&mV&UP(YF(1-(m1'q+&T)(v="
"(70'md(tB&K_'C~$|H(t4&6`()5'lP&6S'2e&eI(-_'NK'=l'a0'y('TR'Q<(#b(TY(,R($i'ph'H_'X((:''BF'o#$Wh(Ml'sY'dW(CO'tZ(R['@J'FO&o~(e7(;f'k](W+('`'Eq'_z'Vc'@l(3y$=}(Mx(>J(NZ'g`&w)&xt',.)QC'VR'O|$RI'j2&tJ&jr'pr(#l'8t'(>'Oo'ub&pZ'XZ&yu'F;(J`'Xf)0/(8t'LE'8Q(Rl'_<(0X'<Z'c((,.'w}(.{'oH'ME(0h'_g'M}'DX(={'qP&rC(?,(*b"
"'>$'g,(.s'5.'3<'we'sS(J<(G('T?(>j'F)'j*(=,'|I(IQ'i]'dg(I''Az($7'mN'?y'co'k|(DQ(BA'lP&zx'R@'.9'b+'j#'oa'[9'@#(1a'x.':a'uC'r{'V-':|(sY'~7(*>'Wi(T>&Db$tV(7o&mt(?N&rg&3^(4T(3{&T<&u}$U!(s8'Pv&l:&,g'.S&z9'ik'qk'Fu'n:'|1&rE'p['@V''G(4u((>(()(L{&{@'2W'0k(?,'ra')l'zE&D3'bQ'qa(1V'V7'q''Ub'8~']H(&i'}j'mE'_$'v("
"'nm(*l('d'6~'Y4'b4((o'_B'Nk'YA'vl&QD(5D(1F(<w&h:'KU'Sh'*A'VZ'C.'n}'Sf'w&',+'t{'Wk(4z'G_'yW(Tn&Xa'4x&AB($P't)(6.'hu&zu&)='e1'a8(Sd'n['U''hJ($K(ve(-H&m-(;|'v8'gF($F'ri'g_()s&gY'7N&v;'h}(6j'PC(<>'VH(?p'K@'vS(5t'RW&mX&v3'Ay'N/',#(C]'&7'dT(?V'$g&D#'GU'F<'~T(1B'm5&TC'3='I!'s+('h'Q[&8,'oo'$u(:9'Jg&{.(5J&UN"
"'v$'kl&Dr(M)'E='Nn'_Z&q}'nx'~$'u)&`y(7+&RJ(&H'iM':/'~r&xf'.8'o5'H!(/I']_'QN&Wn'^G':#(?1'|F&xW&Mg'Lf'v1&[:(F,'dR&7^&`f'Yr'`#'X^'n8'tJ(>l'^,'IX'07'dB'tK'|o(2t'Jx(=e(1G'-W'Wb'F&'OA(1D(JC'Wi'iU'}9'0G'nd'VT&u6(@[&c^'7S'wa&xs(?J&vB'fF($>'!,(&k'6&'Dq'p<(0L''4(:O&tc&S_(:J'2l()W'qc'WT('Y'|Y':m'.F(=/'r5(6G()>"
"'nx(&~'tw&vW'll'0M&t9'm:(#!&Uh&rf'[U&9B'}<'xl'AI'Qn&Wj'b1'|D&ft'o!&jU(8Q(=C'r7&y<(8A'F#'qc'hd&Xv(.L'[0(K6()2(;0$rC'Z)'a1(#E'R?'f1'hs'Lb'T6&wU'w]&};'t1&^4(=6'Yo'Y$(+=(,j&S_(LD'Z3'!a()?'`r'T?'EP'6!'n|'jc('.'qn'l1'a_'&r'Cj'g2'd+'cW()z'Z6'y''Y@'[3'[-'a{'t>'[b'ra'cf'`)'^V'{J'ta'qu'oO'f='hk'Zi(Mb(JT'$V(g&"
"&i8'?5$E~$UB(4n&k?&rK(Ly'p0'HO&ql'{V&?f'Rx&_x'`7&T`'M''dr)'w((u&kO(:i'>V'3s'!!'ZT'|l'{+'pF&wC(SZ'?I'kN'x.&]K'ik'LX(:g(_u'wd'8q'e0'n^'w;'jt'<$'S&'E[')H'#Q(=_'lD',V';J'uX'Jz&}}'`{(F)&`|&gi'4#(HB&{W($i(8h(5y'1,&Q*(BF&`t(;#']Z'/](+I(GY&xH(0K'KG&Ip(FN(7h'n&'_a'j0(?9(#H'17'a`${z'hU('c'S;'<v'uQ'{;(?Y(Zu'I["
"'}3('T'tp'NP(`c(.,(=c'V9'zC&]6'z@(8o'l}'T-'w+'WD'oz([k'2V(Gr&K-'l2(.j&z('p7(!A'5<'ri'S<'.m'b_()['fd'8/()^':o'u((A.&e^(E`'}r'BZ'Dl'}r'bk'fP'z4'h$'BB(*f'8I(*'(;K'm?'pR'RV&Zc$>g(/X'i{&(W'`n(qc&oa'I^$}C&c>'eR(Gt'e)$f6'ZE&nW(-v'sC&f/(,M({R';W&_R).Z'P['vB(f.&gJ'q8'aL'kT'Qs'E`(>X'0H&Fk'KN'@o'BO&|2']1'Sh'DS"
"'.~(6+(M,'W@'zN'b~'R2(*o(l[(#Q('['a6'bG'm6(3L(DR'ny('W'`H(<7&X+(T8()''~h((2(lU'73'`{'Er(Pt&kP(*@'`j&iM(JD(0U'~4'^y'rS(1O$~f'NW&c[&-='lP&s6'F)(2l'AI'R]'uy($<&B*'nS'|{'|o'{!'=w&zo'{(';:'j/('q'M/'KD'QP()4(9?'4|&a;'Bc&Sk((!'d*&ya'J3'p5'8U'(z'k:'wR(JW'gJ(<c'E,(#B&j?'2Z'al'X^(,R'_z(Ul(+8'7b(62'Q8&;F'xm'!J"
"'VX(4)(9`'yq'H_'ii'{2'Ns'$I$Bj'9C'E*'r#':k'T~'~8'}(('{'r0'/O'zR'X.(0$(8E'f|'Ee'F8&v{'w#'DN'e;'pV('j(-R'?<(4v(,s&mQ'WE'kt'pW((W'N''ez'F*&^l'&9'k''-k(Sc'JE(!6'}v'R*'oc(@='bm'8m'A<'jd(.O(>?'[b'Kn'rF'VF(<*'oM&qp$ta'l=']K(&O(,}'i>(1D'#a'Ou'*~&rV(&T'-j(NN'RZ(I2'01'z@'H(&xr'#/(Ao(,/'We($r'IP'$E(CG'I($pW'7|"
"&h&'_4((V&wV'EN((f(5F(&N'gn(.9'Pv'u}'R&'9#(Kj&(V')q')?'i_'Mj'S1'Vd'Bw(5H'7|'tX'QN(9H'1d(8g'uO'oj(HN'dA'l$(-E(#H'`M&l;'DT'ND'>(&49'M1'#`'i>'|x'X1'?I'As'g('.@'I*&RX'uM'02(3~'34(/A'9]'AH(8:'8^(*+'OO'`1'b&(!J'qL&8y&hw&do'kF$}8(7n&s_'Nt'M&'y;'2z'GS'P_'jU'$6(N((+q'M''5((->'B?($:'oH(>]'gw'jy('E&T+'&:(.P(9t"
"(cv&z@(63&s1'm+'p[':F&5B':T(U5'|^'`w'{i&ku'~d'Ws'|n'6?'R@'kb'Q>(#M';1':x'3e&~H':#$du'n)':#'el')=(9g'/Q'18(Wu'2.(*1'_''cn&xT(H6'd<'7P'jM(>h'mP(BN&21'G4&33'3}'JB(&3'hO(X*'bT&el',r&[Q'{H'aB'wG'u['z+'b^&=;&?g(7j(;?(<A&|D'k$'t;'<['^z&a`'c3'}S'&!'}G&p('_0'6F(!q'7e'~T')6(,U'JH'mg'<b'6+'qt'?''@b'UP(+6'Fx'39"
"'H4'pb'~9'r|'o]'a.'qD'bI'Vj'Ou'h9'rx'qb'sL'th']B'e)'`_'a9'hk&n4'|:'8C(;+(F7'A!'Y$'5('Qk'nr(iM(E^(_{'4@&Hd(.F&1Q'CH's](90'@1(8@(.X'8T&yI'}1&hq(.1'_m(xK'nM(>!'D4'xh'IQ&'n(@'$dY'-6'5U($G(2w'<O'UK(.g$A((F)(Hg'`O(<P'HT&fA'fX(JE':A('p(?K(J@(>z'dq&)^(&i'5F(S1'kr(Tz(KM&]m'1{'Y|(/S&y-'|$(KB'9,(3y(FP(&$'+j'd<"
"'&=(`B&1)(/K$yC']O(;8&C@(,b'Kn'(J(Ok([S()O&bB(3_'21'39'x.'@W'5O()>(:x(/d'0E'hy(Ig&}Z'ea'|H't['f7(g,'ti'GY'27'IC(7X'tL&Vx'Rj't+'QB'[6&yE'oa'db(89(3q's#'G^']B'pT':T(/_'[b's9(3W';:'j'&y,&sO(J''HZ(*U(;E(g}'Wz'I-'f](;<(69(*Z(!-(88&~['fb'R$'j&',H(/O'Kh'c-'vn(2h(;A'J#(^b'p'(vZ&z8&lM'}!(b=(FI'z?'Lm'Dv(/R'>m"
"(~>(Eb$s('pm&[^'aS'a9'[1'l7'[{&X1(5g&zu'Gv'R&',.'g?&2W'm.&yf(]8(2{(&''BI(Sr(S#(?|(/6'n.'qN'h((Ud',n'Wh&|6'|J((D(Rt';B&z:&&c'm6&#['hP''>(XE(MW'zp(&z'dE(!C'Ax'w,(D8(^$'v}(;T'D}'lr'W7(R('l.(IC'=M'SM(+f&<R'}H(#}'e|'RK$dz';U(3_('C'ph'&}'m~'Np'_2'P='VT'cD&]D&r@'^L'xY(0:'hk(10'Vv'h>'J<(!b'8B(,5'-'&pG'[N'T8"
"(;S'hP's+'cN'hZ'U,'Y?'05'xA&{j(,*(&a']W'PI(4l'Xh(2*'$<'b+(96'6A'4n'E?(3](*F(Km(+7($5(,f'bm(5D'SK'r)'gH'uH(!6(7='W9'.$'va(!-'eZ'Y['r/'U|']/'Qh'qp'd^'SE'8e($Z'eb'GD(&{'H_&_r&tt(.J'kX(.d'<+$yZ'&=&U/'h.&L4&av'dK&T-'WX'kM(A|(7s'A$'^r'Y!'se&YS(<k&3='jL'Cs'zb'|<&rP'vv'SP'i+&yE'Yy'zB(2D'(o$an'{O'4{(B0'tg'0@"
"(?W'3}',5(!v(5O'V;'1U((d'.<'fd',B'aW'xK'}V&X7(I*'jr((A'*!'uc'Nm'@e'ud'6u'*o']$(Rg'_K'#e&jw&;C((^&w_(.0(+M'gU'ZM(S[(0!'.1(Cy't~'r<&5h(.Q&N;'YY'BL(AQ&ma'2y&zz'M*&TH(93'fb'b$'Uj'l6'v_&Iv&YS'FQ'g2'!~'k-'Hb(:e'=3'@#'TO'$z()V(&$'qF'.u'p{'qw'ED'UL(CF'~^'*;(*/'t!(&p'Qa'ri(!c(P4'(}&sa'Is&n$'f?$y@'F^'Zj'(U&lF"
"(16'83'v^&c<'xA'ND'q2'j&'4Z'!('|u'F{'Jr'?O'f8(!g(#d'7r(V='B:'L,'mK&``'{#'P$&y|(lF'Fk'y>(])(C['i`'BG'l.&Tm(..&S''fU'[y'Uj'N@'v8'iS'XD(,m'E''v@'[H'Cp'W='As(0@&{c'hj''q(3;'yi'G3'-]'dt(5d(@|(03(D8&v9(B_'1q&~<'b1&t!'cq&k1'Q8(5H(a4(PH((P'fJ'aF'xD'TO(1@'p]&d[(5N(&U($/(1''^$$}'&8E((_'p;&s&&N4'13'n+'d8'dL'a'"
"'UX'#T'd~&P+(#Z'm4'S~'TS('['T[(=i('D',5(#d'&:(6Y'yd'OV'o~'Bp()Z'XO'Py'/!&j-'J2(&c'OA'pD'o)'rH'rP'_l'nG']]'f8'w6'|Y'`#'sP'ZN'p}'on'nA'hk'lZ&_Y':l(l$(!F&D{'Yw'Q!#5R'GE&+u&yq(BT)+F($$&cJ(Zr'I7'q4'Gu(F0'w6&_H'|L(tp'[`$i2(0!'U:'_/(X6&!-'}a'Yi((,'fp((@'}6'}~('c'Uc'-='>R'&Z(#~'d9'$G(3+'ei'j_'Z)'p|(8B'm]'u#"
"'s2'<_([$'m='Jh'bc'WC'6a(#i(s?'(!$~m&a5'O*(.&&Wx&:A(e~(7+&4S'ui']3'No'r8'_w'C-(n>(:`'Jz'x?(g(&nt'R(('L''h'fH'3d'zE'e*'cx((s'V$'@:'IP's*'@g'n_'x=(8W'c?'k['Qk&zn'iA'Io(K;'Nc(O+(!c)$o&U{'F3'bh'S?'iP(>r'u:#kp'^1$rz'dz&|#(_{'pZ(<#'[S&4['Io'_v'[:(_M(Ir'TS'sZ(*O'Lk(3_&~n'nL'1F'Wh'u*'TR&}Y()s'z$'M;'i/'rv'eo"
"&~T(DO'4K&X`'`X&P)(:/'>*'Fi(9f'r*&~P'|X&,<'e})28(MK(2q)U}(:8&cG(kx&J9$IX(1d(*n'ip)'7'mS$cQ'y2&?i'gH'u-(Ou'o,)-r('e(=H($9'qv'Og'vQ(6b'|n(3x'3&(Dw'U1(.+'Uz(0W'C5'eC'I9']2(#*'_T'yb'q&'v+'^d(,g'LW(MD'sY'fp(,L'[r(.~&eU&{-']x(&|&6!&=&&lf't#'Y_&,V'~('c/(7(&ky(+S'I9'VV&Y](,Q'[:'gS'j{&Ez(/H'TG'=(&w5(+Z'j3&bs"
"'d2'Y;'rp'MX'Y_')f(/*'z<'Sz'pr&B4'2~'g3(.+'c#'XP'Cz'zA'|O't^&}A(J!&?M'c)'0V'Ti'lU$f/'.2&w((:{(?G(_/'IN'Yp(;1'U8'Dh&oJ'Hs'j7'}}'W''bP(8V&?]'{o'JN',7'.^'wz'm-(b(&gr'yJ(-2(-L(>h&0''x#&~c(*E'Ps':U'l0'Sg'No'd*'ZD(#|'l:'rW'n['LE(/N']4(.Z'uS'F''UO'BI(Bv's#(8`&>m'HY'dB'dl('>'5J'/='$s((0(+<'|/'nA(>G&;|'yV($["
"&zy'C3'/2'~H&9A'[('#R(R8(!3'g>(3z'~`'b+';S'h$'`j'R5(2b'ah']O't0'RU(0y'VM'zY(O?&Y7'u3'Y]'3H'a;'Oc&Zb'tl(1F'K;(,{',f(,:'_;'3U&t$'yM',b'-f'en'WH)$6(8X(3W(,9(.k(/k'Y;'}+'|X(N+'t&'G-(3w'P<'y]'hQ'M;(0s'c8'MV'GA'?7(.7'P3&vI(+~&4n'Or'Zh'|p')''='&[2'vg'p.'AZ'lj'Ub'2I'v~&9B'_I'#F()Q&vZ&[p'LE'Tt&;I'ft'=J'c5'BM"
"(AB&|+(Ji'tM(*D&s`(&x'FY't>':g&v@(@E&ub'FW'iu'iD'hV'h,'9N&Ys($F(0+'d('kb'29'K('xn(i^'bp'}6'jA(I='T`'rv'W}(.-(W6':*&J/(L@(.R$[4'I0'':'<3(B0'?.(AW((n'NC']@(,*'g}&|H'^O(P6'k8(JZ&sg&wF'u+(Ne(61'/t'7c($2&hd'1!'3*'.N'jK&z9'T9'-$&SR'_F'~`'e6'o,'c]'Q$'pr'GK(/S'}f(#v'.t'[.'Z''Al(=g&}$&bG(,^'=c'Mu&Ji(?;'[L(Gs"
"'~N(-E'!`&D7'-O(&,((](*2',n'e@'=x'lZ(m|&l~(hA'Cb&`X'y,&|&(.J(u!(#M(&~'f+(9U('D'Vu'1<'w['e3('R'Ux'RM'>0(d?&h_&o3('@'G#'Bh'Y2'-i'[<'SC'RR'+z'NQ'9g'fr'[e'*!'9;'7j'}1'W&'^<'bj'oF'oS'b7'b4'gT'`I'a}'`('aN'uc'fS'ov'hk'r]'l['ol'Y,'qm(w:'-s(/P'o(']F'Cw'f.&pg(^Z&xE((Z&KP(d<'x)'22(9`$rv)5u'S.'4d'p;':'(A](PF'^m"
"(`L'bU&{]'md'S='W,'U8($p(F{'>J&aL'DA(*l'3S'wt(_`'k('5U'n*'oi'oL'_Q'<u'_j'<|(px'|t'iM'cM'/u'dU'eT'SX$du'o]']m&uh'n.'RW'q]'?R'NE(9m'O*'m#'[)(C<'w'(F&'t}'oo&!E$uY'4+&]T'xu(1D'@K'I,(.g(*7'T?'-](5=&q#'M~(dL&VZ&;j&L0'+Z'Bu'}y&x9'xh'6a']')--$fD(-.(k7'Ym&os$Q0&Rt'Mm'N/(KS'Yi(&9'ac'em'Wq(e7'qc(/k&7U(8-'Q-(95"
"&{h(U4&BN&uD'mQ'_U(`)'95&zX'/G'wX'SX'nX(+K$lq&pt'Sj';M)(w'&a(SZ'h`&tt'y.$W]'}F'kO'pK'x@$O[$Qs'!''gI()n'nH'k]'hc'<D(+|(&f&C)'DR'aE&hO'Im&Zj&fU'nY(/t(1c'7h(#@'~/'!M'GB'h((3Q'KE'Y`'hH';a'vz'M/'h_($c((['a4'_s(5:'[s'S6'_D(/L'3.&{.'lJ'Pz(8F'Tc(Ao&oX'r_'t^'0u'`4'.q(Ip'E1(-J'1.$>5(+x'XE'~;'z,'$Z&w*$8-'@J'Tv"
"(+('?D(#&)Y6'LA(8R''o(-c&M.(du(G|'DD(BM(w<'Km&yG();&><'um&7Q'X]'a{'Yq($/'J9'ts(+5'uq(,C'['(3U'`4'#6't|'6|'C`(E,'X:(<='=G'|k'QJ(G('gx&v}'5;(.Y(d4'87$ws(,^'QE((_'zg'z4(#0&xQ'+i':2(&G'uD(7]&AZ'7^'ca'j~'y''x:($0'bA'p1'vi't-(9[&:9(/A')D(DT&ze(@P'p5(!Y((b(*}'WO'q8(&?'}.'7)(OM(11'z#'sS(/X'5q'F1(?]'W|'s9'=]"
"'w]'l|(!#'p0'eY'Y~&LB(2g'S/(@t'!X&o7(*#'pF(,C(?1('~(#>'u7(2=&:[()u'x/(/5&i2&)!(E^'V^'nn'6V'|}($+'qB(2d't|(,,&zk(D5&_n'g+'vV'YF(5K']t'Ao':N'tk(#c'tG(#H'u@'u)(;~&OX(=I'Cy'}$'B5(6:'j.&Ku(lY'v*']U'uZ'wr(.0'x^(*5&M&&uB(JP&_y'|='aT'Od'po(=<'z)()1'|W'sl'u-(,A(&2'c6'm>'r6&_C'jP&Ly(7e(/d(,*(!q(3C'yA'c='|8('s"
"&hw'ej'Mh'_#'>q'TE()S'SU(2d'?Z'zA$l6'n9'lJ'nN'r`&fL&Iq'72(4+(hc'Q3',{'j`(]S'j;'yT(,:'eq'~*(*7'ux(&((7@(5#'pi(go'E;'t7'`~'?q$Sy'o`&Hj'm''q='[8'x['Yf(HP'b_'dS'S^'jj'fa'nr&iO'Wf']r(Hm'o8'PV'lM'bD&w4&MZ&r(($7(-$'LN(4P&{m&RY'@T'D4'bE'KI'HU'g2'XN(/l'^F'jw'K~(7a'F:'$i(!h(9}(@-'dT&bd'aD'bi'_X'hJ(F~((u'K]&dG"
"'0I'p7'Q@&|e'nm'rA&w3',f&x;'&]&x(&o7'BJ'(B'd0'qt'dV'pb'c.'c|'nB'qq'k&'si'jl'ie'jb'l8'fs'x.'iI'l='pe'ki(F`(>E'l^'e~(9m'U^(GR'Fx'!9(Fx'O6'CA'fW&|_'G4'~f'z''w&(@D'Sf'rt'lS(@d(|U$[9'Pv#}]'4g'*]'q+$nT(9@'8M(8S'vD$d-'r{(I2(+,'(?'uJ'jU'(J'bk(QF&p#&>|&W?$wK(6f(('(6G(#a(Dm'gj'Fs(6p'~!'3e'o@(6t(Hk'g{'kU&lQ'/a"
"(7E(+L'|3(JQ'h)'[i(0h()n'ht(8g'UK(Ad(;8(/4&vG#sN(F3'w@&Na&HV'X0'EK'?<'Tv'TJ'pT'r*'wC(!L(T'(=V&4#&ax'Xe'#,'BO&Js'UA'@D#mB'w7',S(H>']K()x'/c'm''[7'kB'Ox(VP'O-&Zu'rK(L2&pF(Sd&]s'dj(4)'g](0B'f#'rq'{3(9N'j3'T4&|#'Zo&sM(Q8(3W&of'C8'K_'cI'<J(&X'v<(#$'wh'g?'kA'PH&ME(':('!&v9'Jn$E@')+&#;'i~(Ez']@'u+$j.(,I(,9"
"'9^'or(5](.a(P['q/&iW&0C'jO's/&Q<(<C(!#&JH(-b(,t'j7'1$('>'_>'tu(66'gA'0O(+G'~M&lg(B~'pI's}'d6&vA&YT&um'vk'X$($a&y<'|G(!O(*z'P?'w@'+n&48(4N$iY$nz'Sc'Ig'Y_(9X'@c(I_(1E(?~(+R$jD'~k&vq$J|'dv'Jy'Qo(30'l`(>g(3*'pO'/B(#|'I3'kj'Aq(*x'uI'<J'h~(.5(AF&SB'cb'~T'+}(?B&@8(.:'hU'oP'0g'iY(/<'mW'k2'p('l+'W='+b'rv'jd"
"'i`'U:'XG'QX(]n(1/'[c'U6'p&'v]'}V'9D(Q3'jI's[(e<'H{'RD'Sr(PC'`u('U(*W(,^'lg'JL(#>)#<(!j(*O'ET'_>'-0(2['dm'89'Ty(#x'to()&(#''d]'q{'W-'v9((k'sW(9O'h4'8~'k.)R?'fh'{j'p+(p[(9y'dx(L;(@H'P`(-l(~f'Pm'LT)F}'>2(=;'v)'zG(.E((3'zf']W'W@(?~')I'p^(6E'<N'T>'Cv'i-&j,'oX('O's*($)'qI'?O'p_'2[(A>'wy'{!'D#(/L'C)'Tc'cJ"
"(&d'ug(0)(Il's>(+Z'~*'@G(Fy(!n(5T(fa(:[(Q`'#L(i^'U<'#,'[`&L8'^8&nf(,X(X7(f4'?+'d;'q`'/}'&D'AM&dg'~Z'RZ'qF(/5'}u(@3't;&v@(Y4'uS'wK'|j'y5'.d'K)'Td'n!'zT'zN's.'ls'ki(1H(&j'VR'g1'zj&YN'M;'Ze'8R(JC(.c'xT'~/'US(9W'm|($,'q+'AR'l;'vz']P'ek'b<&-C'k3'xQ'yQ'gN'N1'i+'z;'rq'Yu'e6'YH'zw'oN&bk'91(8~'9I'ib's/'YR'r_"
"'d!($P'^h(4.'v^'k>'n[&wz'ZS's$'rE(;2'xP'|o'm@(<h('W'p#'yZ'K+'Y6(,D'~d&JB&^+(.2(-(($b'v4'/*'eQ'ch'hn'qR()k(WE(UX(,D'c'(Fb'4t'zI(/*''X'zi'we'fw'vJ'~S'QJ'rR'MH'p:'gn'xn(;I'Gf(02';9'Np&xd&od'#]&q5')r'E?&q/'pa'-7',6&o+'mI('N'#v'np'j0'im'k9'l4'jA'k('pE'dd'qI'dy'k_'j2'dR'`J'r!'wa'iI&|*'e9&p[&=g'tN'|c(78(VS"
"&tV&Mh&=,(3X(#3'jJ'O2't`'g$'v+'$c&w)&,F(1)(c+(/,'FF&_j(tB()u'^@'}f$]@'?,'`f'IJ(=m'KZ'Vk'jM(#}(!G('0'f2'm-(G*'1i'[.$Uy$e.'dS'yF'sL'mZ(?m'US($D'g>(#@'iA'Y_'}z'~I(),'bC$~`'r;'1F'bw'b_(@5'j0'zI(4E'N4(3u'G#'kN&:L(<h'P}'-.(!P'*Z'L$(2d'd-'Wz'_C'~$'jO(*>'uy'UG($g(Ih'zW'KA(QL&(t'xa${S'tA',M&p.'fN&pH(0*('R#c."
"'Ed&[C(D{(4V&od&ZD(;n&bQ&rv'5T'2J((j(5p(Wa($U&]X'7q&b7'Y='NL&l:'c.&Vm&NF(=9(Q@(b.&tb'h@'-p&-((!=&qL'~v'dv&D:'}t'h;$+g(Qd$vt()a).:&]p'#T'j`'(h'u''<*(1P&PZ&l?'~='Cy'Bq(-:&4}&`E'~3&4R&i:'zF'Ks('Z'f;'b]'sn(Xn(+x'h['se'Il'kH'*Q'eh$fV'>/'NY'Fd'v2'x2'w<(2a(3S'uG(60'nD'aX(0z&B$'Es&sS'[L'Xm'8h(+j'ZX(5^()='YH"
"($](#e(1,'{t'lb'3e'Xh&s''/4'~U('c(4i'1u'<f(7*'C5'_f(#N(U0(66'hU'[d'8*&zM'x9(j#(+4&wv(X|$/V(4<$sA&`i(,[(GD'#|'`^(/^'Si',W$|~(zv'w*&f2'h_&zH'U5&,o'aY'Q^'a<'lz'kV')K'nD'No'6L'k]'Pt'|<'KE'z3(#]'-`'re'JI'';'?A'l^(.i'w#(?Y'k;'|b'oD(9r(7r'jk'MF'x3'bA'hd'KT'fJ'k~'Z('t7&T_'^<'gz'j/'4r')2'o7'q6't)&qj(87'W4(/X"
"']3'w''Z{'N<'jG'fQ'OD'1Y'?$'U)(#K'w1(`t'Ii(V2&l4'RU']u'_`&[*'k}'bk'be(Ft'IA'm<(35'b}'7T([5(){''e(#j'eL(,A'mu'n>'rV(d`'aH'+z(*{('0'ep(/[''>(D,'=3(Q;'~W(JY'o]'tB(>V'a~()!'2T(Ey(J1'a/'~0'/8'+E($_'nq'zC'm-'vb'wX'e&(-t'bt&wW'u='m8'y,'is',$()>'p>'o6'ni('!'*/(&S'x4'On&x+(6i'ip(At(3c(H3(BK(R@'VO'NT'zd(MC'kl"
"(*u'mj(<'(!;'))'^V'OX't~(#G'_o&}q'C/'tD'iM'tN'A2'mS'xQ'[7'I;(@X't+'YI'x}'2E(6+($d'<|(*Y'mm(&5'r['{;'p1(Da()i&XK(+n'dV('*(Py'S:(N#(1N't`'hS'hG'BT'x''s)'{$'bs(?b'gw'|P'h8($Q'FV(C1'Sk'i@(#4'vy(=G'w:'kF'|f'|0(JE'c#'UD'uE&p/'lf'^?'0K's''^d'sn(EJ't<'z^'T.'jU'wQ'l+($o'iJ&ml&t?(|}&M/'{a'vy'yR'`o'dw'n*(2;'Jz"
"(>4'{x(1@'r0'D|'h<(Oz(81(1=(1K(Sc(]`'l+'UH(nF(At(a?($W(*7'pC'R~(5d&{f'U#')2'>8',6'uP'31';-&{R'+Q'LW'9e')r'>M'rO'a.'v)'mr'k]'nZ'i}'a7'l$'kd'gE'n]'tC'jE'nZ'n1'iI(+_'Qt(>-&JA('T'Jz(/s'WF(dx'm0$t+&W]'ax(B`&d=(dI'J8'gQ&_H('n'j''},&uc(Rk(0]'dR&EG(@<'f2'F<&s^(3$&qR'NL(?R'iz'x-'k[&^-&4b'w/'6w'Rd'~3(77(L@'QO"
"'Hq')W'tQ'g;'tJ(Vh&~k'$x(Qa'WH&~?'gS&oL'e#&Ga'_S&m#&Bu')2((]'pe&Fn(('&*N'f3(Q5(Xl(B/(-:'sj'{k'Xa&;3'GA'cV(@`$V+(&W(Dk(X:'<n()6(.y''|&]-&F)'|c(&?(2#(+^'|V'7:'cU(!W'@t&|Y'S}'H+':P(If'4C&HY'bR'Ms'kG&BH'/$&lN'u('g&&NW(=B'71'q#'nY'nv'q?&uT'iI'5=(,m'V*(nU'<I(0A&~^&>>'8w'7;'Gx'CR'c-('0'fj&y]'m9(2k(B3'$['Uf"
"(0>&,P'M0((!'@I(J;'Fk(**&tc(Md(1?(;&':]'G7'q{'[}'pz'K='qp'R0'V&'*B';+'iw'=|'ce'6o'2`'S7'~X'}p(6<(k0(TU(Bc'}F'Yz('E'a8':`'a6(X6'f:(1w'{g(UC&s#(=)(Ci'kk(.V(Am'w4()j(G)'vh',&(O((Bf()a(.U(5('vy'KC',Y'rP'yE'{4(&('tT(3Q(;e'kh(JO(+1'qH'Zr'}P(Cq'Hb(!('RY'hK'vQ'lm'o,'W#'Qr'[Y'?8'v((!5'Rg($^(9L'kd'?z'v9'|,'RV"
"(:o(-j'rE(Li($#'g<'xi(+V'v^()$(X+&fL&bo'pU(.*(<5'~O'z6(8U'Tq'f4'i[(7,'v7'Xe(&-'#v(y9(1F'q>(En(+)'xh(.p(<n(4('n4'|G(/>(5>'!|'s;(WY&GZ&_D(+<(#_'|9(0((#{(6l(D[(3j(L>(0b'y['(q'x)'f!'Pf'=*'`4'|@'vD(,`'|V'vc'1*'Nq'g}'q1'uY'[{('x'W6'E0'r4'xv'Aj(PD'mD(5.'~z'v6(bs(D=(*l(+X'y[(?m&2!'K<(;G(A<(4`'xN(?b(/Z(><'t'"
"(&!'u](,O(1K(0+'wf(J1&$Z'-?(<X((y'hM'vu(-i(--'iu(*~(@v']5'n*(:g'u1(;.&gy'+c(*)()3'C0(7_'nx(3d&>c(-Z(2B'ky'~a'{h'C^(I*'i.'^G'Xl(-_(6:'q#(5;'rh'`7(<X(Bj'Yf'Z?(Eg'q['?M'OX'}|($U't:'d7(^E(Sa(t?'d7(01(Kl(GW(-Z(#E'Wq'L{(76&q/'xL'd6'YT'u3'{q'{I'Q7(/~(9N'kc(!:'r4'/P'r{'iV'q*':-'7<'?_',_'+B'*]'R1'K}&zQ',y'=a"
"':x'g5'cY'tV'u:'gN'ie'q0'fI'jv'lY'cI'lG'eM'm>'is'j/'iI(7$&_-'8`&*#'0y(+c&r)'Nh(N#&O}(+b(jU';q(#U(JI('l&2t'=c'b4$jZ$OZ(YC(=h'[5(>p(+;()`&A*'~9'e4'P#(0~'gE'TJ'y#(5m&Ht'<$'MK&^A'm7('A'^]'gu(9=&c2&z.'q)(=b'VX(Hq(2O(/A'kW'|&&r<&N7'ef(93&_x'gr&ku'tQ'Rq'`!'=s(&c'WX(.b(h{(-:(1u$j+'LE''C&j<'4a'`:&RR&IA(=}'KV"
"&IC()T('*'L&&XZ((<']=($_'LW(->''('vD',m$v?(#[&YR'Nk&l*'5f'7Y'u@'P>(6k&,<(-*$Mw'Py(S6(X9((N$W,(1$(L*(,M&lw'sQ'.H'w1((5&s3(>7'RH(-I(&s't6(,-&:Q(@N()M(.](8}'rR'`8&+*'sg'^x($u'R8&@x(0`&DO'hm(,r&su'pG'c:((Y(CB(#V(,m'vl(0P&Gb'c~'t=&r1'n9'}]'xb(7O'pt'M-'y<'j{(/n(&d&ID'~p$o*'rP(4>'RD'ho(I2(Ji'#}&]O(XY'nm&aQ"
"&a-(23's3(79'4?'*}'+@'N~(0d&e<'^L'DF'a/'uO'yI'zT'R3'b!'dn'X!'}6'3l'Yi'`)'G''qD'Z.(*!(p]&|E'qi(*:(7B'.t'x$&gs'n]'r((#~'vB'IG(!0(7I'!S'c?'$L'^Q'yo'z((Ug(4u(^`'V,(Bn(0$'YV(hV'}k'*G'WK&Sc'a*'cY(6a'^)&bX'[|'t?'`Z&e`'X2'<x'($(#t'qd&?6(4_'X4(+8':c'OA'xp'oZ(KO'w_&uQ'tI&9b'~`(&w'd((jB(*^&3z'Y+'G<($3&T?'}'(7M"
"'w?'/U'ja&H$'T+'W2(-P';d&ei&l.'x-(+`'vv'Rw'nB'oO'q)'IW'vN'Qi'qZ'zQ'd_'7~'p+'[&(+L(P)'ch(Z}(NS(9p'b)'Td'd3'SS(a|(=T'xG)6~'gV'Nb(@q'6z((e(Nu's}(:K'h.'7B'o^'H+(g8'Ts'Xe($7(Dy'hI'aF'XT'v=)4$'Ya(11(&s'~i'fG&3&'~]'cH()=(20'bs'm5'r6(Oy'Qs&A:'}}'6('hX'x('eF(,l'l4'w('X@&}$(;((bD(&u(+['gE'|](1R&]5'}v(H^'uq(A6"
"'}p'Z'(!&(#0(nt'Yq(>0(T/'k='v@'w|'w7'nO(FM'e<'@I)Py(7n't`'VY'[h'Ev'rH'tV(4+'.;(Ld(<&'vE&}@'rR()-'yI($](>w'(:&|#'LP(=_'Jr(#9(a~(5@&`M($d'G$'fB',*(}2(4-(j8)Bk'n5(QZ(Og'Ep(h?'~b(fF(/?'sC(>8'Vq',<'vB'~*'X$'Vv(Lc'g*(,g&|:'i4'0['d#(0P'u}'W`'lV$YK(/F'r['!E'LY'?`']X&z&'mP'00'C,')*'Ym')i'O4'?8'R~&z6'72'm0'm:"
"'a}'b4'l]'_N'p/'v]'o''rs'k.'e`'l$'_2'o7'en'iI&6V(WW'J;'?n'CN'aA(9)'k{'0b'I*'z^(EW(J5'W|(8Q(rw'dg(,;$EQ&s{(Ze'_I(?S'6~)E7't|'n4'm1(aX'=}(=q&m`&QY'MD'e2(kK'E&'hH'y$'zQ'[L'~c'l^(-!&um'VA(H~'Iy&/S'/B(&j'wJ'it'.O(')(4R'5I(#['so'y8'n;'r^']G'W/'pC&V7'`L'[D#jd')_'Nj(0|'sp(+T'Za&Fd'XV'(?(!a'ft&mP'i}'.|(;s(<'"
"(.r$zm'x('cX$~3'kR$^7(=B'L4('a'6y'*j&6Q'8$'q-(@?'os&#g('j'tG(J7(;f'z3&x>'e>&uR'^~(92&2F'Pb'VG'CM&BA(Ke&q,&`K'sb&,F'qK(_$'r;'x=(Bt'|K'Ny(#f$yh#c3'Q<(B+&~h'/5'oT&oB'zh(/a'k[(2s(eF(4i'!i(5N&<7'dj'42';*'/Y'wq'w-'/'($L($F'XI(7'($;(*I&{V&g,'g['Fw'MM'wg&YW'k|'Os'0@'~p'<l'Za'[6'Oo(5v'ha'{f'tu'[m'gg'VK$y='xu"
"&az'Ji(Fv'0](e:()~'sk'(_'&X(#o'~p'ne'Oz('S&Hq'hz&M,(B`$XW'dr(3n$Ro'@_';n'=7'ii'u)'L.'^7&p#&@S'nn(*A(UC'mt']M(,(&Wd'+['m#&hC'F+'f6'E&'&M'b_'Ya'D5'hB'Qy'[:';S'f+'m.'KL'u#'^w(+5't!(3~((R&M>'tk'yG'E7()F(9C(6.(=E'uZ'z]'v5(k~()[(>`()^(MR(;E(I,'mO'G'(@@(9E)7_'@/(^i(U9'r)(3`'t6(/0(4k($&'Hl(#k(H-(54()D(;E(&<"
"'Fn'u/'z('lP'l''TO'n!'kK'Vd(Jy'd2'vS&x'';2'h9'wM'M_'at'|K'uv'i#(5g(/h(2#(=N(Hc'xG(LL'7Z'Of(0S'mr'{3(G&(#&'Bo(J<([((<k(K5(_g(&5(aX'`B'c$'dT(W>(MN'_8(A!(8c(*&'ou'mF'n'($6()F'&N'{O('$(&w(={('9'zM&t3'od's_(Bc((t(Hk($q(2}(-U'!`(:+'ea(3S'3!(7>'r2'q[(Dm(!`'xt'{h'k&'~6'|T'lg(-X(,,$ee&}X()c(-n(/4(3(((3(#?(QA"
"(3<(C_'l#(;e'Rw'~3(1H(.<&gG(B$'f.'cX(=)(9{'|V'pS'{<(-!'l:'pX(73'm_(2b(1-'?i&og(0h't:&ic(#f($*'v.'U>'x@'YA'kg'jP'+;'d-'s1(Qt&~O'bB(('&`J'Sv'j['{L'fY(#8('F'qS'm3(#r(/.'U6',f(58'i~'vv'm*(Af(-.'|D'd0((c't4'uf'Rl((['{@']$'O:&cZ'|B'RH'^{'ry'}}'hP'u>'aJ'kQ'YA'd|(&S'[y'xi'.I(.6'uX'm*'{K'dD'm{'FJ'rQ'`('rd']H"
"'w#'&^'wl'_c'OM(4L(!d'G[(-^'}}'s)'!7'/*'!8''z&|@'&P'JL&yE'!T(SQ'47&ik&yB'N:&{`&~R'n)'km'lK'fr'j3'jr'^|'jq'ij'uq'b#'g2'ls'eC'hs'hR'iI'MZ$an'Sz'F5(_i&UQ'9[(N[&{v'{<)P0'c0'JF'O='p-&NL&JI&sV']''(p'eI'Nl'FI'l!(7$'OG'fP(N/$1N'^]'95)Fo&sa(Q#(c['XB'kL(Xr&~<&FG'yp()6'H|($P(V{(FE(&y()''e+';F's]'aT'wu'I-(/V'7L"
"'pc(.3'mi'W4'Z5'r='zh'Q`'zE'!5'wJ(($'b2'WU(&g('$'cQ'8+'qF'c}'dx'Nv'rz'NN(?~&m|'{|&eq&[f'K?'Je'oR',c'?D(9|'PW'DB)3f'4,'@`'/z(JA'TG(<U$:s's&(k('h,&o['`{(!5(s#(Jt'Y^&;3'pK&Eb(&J'He'$|&Sa'l6&96)SE(y#'wH'Lh&Am&M2&QB(GF'v~&[4'bK&1L(]E'7J'<h'(r'6^'KX&f)($_(#](j8(Jj(]p$NP'h5'aW'S#'cf(1')D4'Lj(&1'p/(mE&mJ();"
"'h>'gD'pS&+t(9y(,(&RQ'V='t~&iW&:x'90'u.(>K'#d(<D'S''br'==(Ho'vm(6P&;''u:'o='k#'8T(2+'U='b,(7k'ut'}8(#b'o?'b9'`'(,v&/P'HJ'bw'ph'~V'w=(6Y($z'|z&xf(#''SZ'VT'aa(#&'bX&X-(T`(f6'Ok(7='n''u#&m]'<-'*f(,+(-v'#e'X7(TZ'c?&q$'u`'jr'pO'h-'9[&D.(,M(?u'?h'jC$WN'nJ'ku&{O'vh&.='}9',3'h$'^c(2L&W<($3'nR&+p&C@'Rb**~'aW"
"'[R'OE'[x'eU'oC'5#'fJ'h>'b{'Bc'sM'_/'Wm'ay'c*';=(;7(&+'vb&Fg&{''?('c$'G|'v]'^3(.3(9m&q&(+c(4h'uQ'~m'Ff($X'+h'z,'3R&4)(CD'f['P,&xk(/](U4'u)'p{'!.'k5'l#'sG&l)&0s(?S(E=(/x(39'8T't8'Ks'?='|('z]'mZ'P/('Y(&?'d.'o:'5C'QS'8/'f]'KU(**&n{&(G'sE'h/'gL'x9'fV'ur'o7&iH'~k'z0&Hl'|P'G^'k8'>T'vk'qn'yj(-o'{w&Uf($A&U<"
"&hw'!d(b4'tW'S-&@n(NQ(W=(.~'qc(+7))?'~9((1(H{(!+(=v(bw'_Y&J3()i'P.'*?'WI&]b((#'lC&ey'vq(B''~y'Qu'Vz':i&fY(=!'yF'bE$wh'L0'ua'fD'vP'r-'te'G1(+g(#m'!+(:i(^a(F^'n^(2w(:E';*'*g(4O's:'u6'Kb'mU'j+'^2'R['0.'[a'ig'c*(^8(,Y(;[(-L'sC'w+'d2'm['vl'WL'-v'b;'4/'<-&aI'VS$T@'}0'Lm'i@'iY'dW(AX($z&sJ'LR(F!(#<(J{'q?(AK"
"'qx'W>'W3(`!&~T'rS'e-'?w(#h'GI'hk'bE((='rG'*V&N8'Ot'v7'P,'K5(,^'ly'nt'Vs(,l'I9(,''Su(-F&q='~E'7v(*X'DA&=@&do',V'x|(#!';H(86&(.('b'sF(;]'m7(!r('[()8'U,&YF'.r'Ai'rb($/&wK'su'lu(5)(Ll(Br'WN'=K'-T'JZ'H''N#'AD'G,'wF(*1'8o&P^'_t&n-'dp'tC'RW&{I'Vu&i?'|L'km'C4'TW&ug&g&'{A'fR'rI'm&'dE']J'jJ'pp'sk'i$'u-'s3'sF"
"'kb'k1'k*'uX'iI&bC()Z(5)(5:'Za'DX't-'wE&V='WX'l]'Xl'T<&X.&mf'eq$as'lD'=m'}M(#;&]'$dr!!!$NS(15'7!(!A)J0#ag&<0(?P(Xd(/S'I2&rS'P!);!(:`()}(m4'x.(Y0'27(&A'=7(xx&59&x0(/j(<7(B|']L(0[),('tl(Qd(!P&K['Fz'}#(+b(7-'>n'3n(P>(:h'lK'oa'ia'a_'rN'8P'h!(6]&u_'WE(6d(PZ'vd(!,(J&'bp'j}'<e')F&fw'lx'p7';H'd>'/e'k*'d3'[#"
"&{5'z?(W0'sT(a5'cj(P3'ab'i2(#3'Xz(@p(@3(*T(#9'zm(LT(jK(Qo'Hq)0O(^G(f['wZ&AE'_,$$T'3o('|'[O(bu'Z-'ox&4N'x''=r#RJ(-O'cH&o3'[3(+v).f'O2(:n$m?(*v(NP'lW(?V(=b(5M'zT'k9'tj'}6(U*'t$'WB'Mv(~L'&H'3''|v'Eh&ni'l[(9`'l~)),&C>'@!&BR&~9(7N((q'R1'mX(_W'_S(H9&uW(fm(#?(!g'|w&w@(ML&vr(Wb'oH)!p'f$'Il$Tb&$1(_3'`s'fl&-r"
"'-y(1k$<V(&M(Y3(cH'NQ'SK(2>'U0(5K(P`#r0&mp(:S&B:'zv'l.'SN'p7'uM&W-'S{(IB(1^'bR(Gr(A0(bV'AZ'x>'<X'pT(4o(!Q(89(![']j(Yy'R;'h7'QG'cD'A#'g-(6*'y^$jK'ot'-&'O>'Eq']?((l'qJ'~}'gE'=@'w('g<&hJ'eY'Y-'=L'i$'vl'co(Qk'Ta'MJ'&>'*:(H{(QI'h|'rC($P(G3(2f&nc$~f(#e'rZ'lo'|I&N4(Fs$Qb'6^(EN(!F&RC'HG([f(.#([Q#xw(T~(kA(yw"
"&^W';($Q^'5d'K0$&#$4X(!p'EC'>!'J~'rT&CE(`V'eq'PA(T0'2H&5e'Na(-R(BU'e]('.()-'Yl'D!(i9'>o'~/'hS'CO'JY'dn()E'@u'fs't3'Q+'W='>y$xz(<#'~i(M@(1>'uX&aM'Y`(&]&:[&V8(IT')G'^`'g!'zA&r9(Ib(;)'Yj']T'ep(@F&XF(!Q'pF'zo'.w&jV'd}'=T'l3'G2'Cw'dh(d='/b'r`(-p&kS'Qf#w?&ua'i?'p-'@_'oV'fn(2F'9{&y3'X7&jq'^p'2Y&mZ&OH'vj([Q"
"'R3'(R'7R'+5'v?'z&':5'sD'!H'R2'sL'EA'Zn';['BN(;v'[v'qF$xS'VL'U.&ty(-]&5m'CK'g#'ix(9;(.}&KG(A>'Lg'2P'ri(YK'AO(Oq'Fd'EP(2{'i1$JF($e&`0'2t(C)'m1'x0(-u'Z*'_E'#o(-i(J6'9[(,5'ER(8.'bH'+~(Z-(/R(1('|l$Oe'Cv(B<'Xa(Np))j'>G'Or(J|'dJ'df'_t'Nr'C:'e-(;W'f5'a0'jE(,s&G{'_C(4('63'P~'eH'on&+I'u?(`o'K^'gW&*2'wI(4L'N5"
"&kZ(=u&V9'!='~d'z@'n5'+q&=i'Ef'~s(!_&&J&us'xT'Xt'+a'ee'mi'F)(3I(6?'mn(.0'UH$4!&X>'u*(M](1V(*s(29'?n')h'z>(,g'l|'lf'uV(;C'XV'iI'v;'=~'8K(F-'?](0,&;e'a='`C'||&l,(CG(!;(4I'P=()f'`*'IM'3_'w+'Kv'6@'n#(Bx&R}(<.'fi';i'tT'az&g$'nJ(DU$Qq(Ap',''ia'3['SI(BW)D7'f+(X&'w6'e''z;'HN(9L'Ml&Pi'<t'L^(c)(<t(Rs(3L'ae&an"
"'y!(*L''Q&Y@(*t':b&_a'&H']B(GZ&d('?((7p'qV'rE'r{'e:'u.'i>'qt'r0'w5']I'_|'r>'qQ']F'bC'd#'iI(KP)M*&xX&|6'2B(JH'0C(?U(!h'w6)$n'G~&|j's2))w$/J(MF#j^(=k'mw(*>#f;&q3(+y'WF'{F&[l(]k(d2&.2(($(Ii'9v(Sg&h)'9x'g!'EA&WV'#N(CE'V7(AH'LF'c)(V$(*r'Jm&[0'rq'}t(Yg'wX'EH&X7'Cf($x(1;'Q:'p^'aV'M@(:H'^A'?2(f#'J0'x<(<O(V4"
"$H''vU&qm&;p'!z(VI'oI'`:(~;)63&~('&C'vv'2-(BY'iR(Cx'v''5V(eB'9H)#Q'{=(7|'iu'[l(5_'TA&Z-(!}'~k&6]'eR(/V&gb'>?'}_'N1(A,'hl()&(,;'Rr&u`(AX'TJ&>n'D|'?O'g^&L=(0R(9='7~'jG(E_'Ee(4](:/'So'v2(11'qz'<L':E'yc&W7(<R'yo'Nx'a2'!4'Iv'Bf'KP'Cw'QL']i(Hs'|G(#`(UP&Z8'^Y'V5'3C'VV(<~'_O'.g(Lw)J?(1$(m$'?M&kS'qQ&;@(?I'X|"
"(OI)=L'&>(!^(Yk#Bc'vn']O'3>&Y6'Yx&px'}y''p'~r(V,&gF(On*$I&rW(-F'G>'0((UN(.[#Dh'Xb(-P&`b'!<'}<'Ie(Pn'_s'&Y(6A'z('rU&_h'pU'80'y|'uG'm|(v;'/@'Oq(Xs'a*'oe't)'F[(0R('E'n3(kf'fu'$H'GB).Z'sp$#=&zF&tK&oG)(u(P:(H@(zt'Js'k.(8:(#H'YC(/)(R''=Z(SW&l)(CD'9$'2C(a,(9Y(g8&Pr(2$'kG(Dk'Hc'es(@o'l.'cq'fa'v7&!u'Si'J4'vz"
"'YT'e`'iE&uL'wM'Cg$rW',^'yw$w((<j'pA(36'GI'Rx(9K'A](ln'=B';{(!X'*j'py'8_(/<'jX(9<'Tx&Z3'ZR()(((o'_l'v~($G'F,()v'~1&J+(#f'N&'qf'eE'k<'Q''N9'~z(9s'Kx()_'9{&`v'z0'cN(&#'HB($Y'iI']u'.Q'|M'M6']^'/Z&b-'Qv&jL'As'EC&kN'<<'mM$gE(5w&~*'8y'ga(B=(z('v-(m((7R#pd&@<'HA)]g([b'q~&Y|(k($lY$`,$~W&[Z'qm)3q&kt'sX'67'Bu"
"&`o'WE'l((Oy$PX(/,&nY'pq()6$NQ&j|'U8'Ld(/8(DQ'MQ'nr(;;(F}(;D(BB(&K$+3'9>&y5'nI&y_(A^(Zp've'?i'pg(CX'kQ(?z'a'$Ll&|D(UF'j*&fH'{!(0x&tB&*O(=2${W(:3'(V'Wn';U&x|&LE'vf'2K()#'[_'_S'sb&GH(Eg'b~'/B&<k('b'=c')x&SR'lL(G[&Gm']I&ue'd<'}R'`W(5!'iT'rN(>S'M9'hk't2'E,'uy('q&!/'zw(=g$o5(Qd(?=&@*'/7'dw(2+(&8(;J'-V'J."
"(L''uh'bf'v8&v}'sV&Ad'G@(;n'FG(Z['Dt$[9($Y(AA'_5(M/'r0':r'ku'T}'~e&[*(;K'Oa'XN'2]&Sj(:z'r$(K~'Ay&P$&x;'75'<P'|c'r:$JG(2D(2>'|P&'e(GT(6{(&G'jd'6Y'l.&&#(3K&lG&}9&Jd(-/(Bs&=d(9S')='(P&M3&zy'y^'T0'oe'J/'(D(TG'mg'(y'F+'wL'5g'eG'tD&`r(W3(XZ'[X&mC'1G(&{(4K'4<'by(/Z'24(;'':_(5b(I3(N8(1^'@1&kl'>K$I}'5U'LR((?"
"(1h(,?'Y8'4-'qV(bu',?$ws'7M&Zo'~v(2J(0M'>X(.q'$u(A&(:9'Qa'[H&MC(5d'8-'b0&k9'NF((.'I&'(B'bs'12'F7&xm'q,'S-';w'(3&fC']B'y0']k'q='_^'e{'u1'qe'nD'sl'dv'_='a''c|'zJ'_j'iI'9+'ym&<L'Sr(aE(ts&Ef'Y@'vT)4U${c&Zv&}I(8|'CO$pk(UG(3#'#O'vT(hc'q$(Un'l|((4(${'ZV(?X&NV(/;'qZ'GJ(L4'13'jM$31'qM(>v&mT('2'YW(gd(<a(RC'`="
"&ql(@E(LJ&<Y(wa'kA(/['i7$^W'pg(,6(-=($U(*{(3i'|=(AI'TM'6#(^d(e.(I&'|D&Mt(`T'c6$M$'}h'e|'g>(;,(,g'z3&t1'_H'>e(Ab#YT'>1$VQ(DV'z5(}s'lb'm.'B='HB(.))$s&<G&L#'}x'a$'l{(+#'|g'og([h(Bl'^d'[J'zH'V8&ZS't<'x(([O'bc'bu'j,(4&'x^'[z'4M'zp'v('C?(=4'2*$t-'yf'}!'!,'t;'Q_(;*'oO'S1'2+(8('rR'j['(E'oT&nX'_Y'Z)(mF&t3(;|"
"'r'(0{(&j&B/'Dg(Ew'pf'Pb(>g's.'e:(&E'7S'#}(s{(z;'J+)@d(8y'FQ'oU'.1&*!'5[(V}${T'oy&x^&dc'cn'XA'ZL(-<'y@(Mv(>Q(&l'aM'ep(nm'mK&Ed'gf'#9(xX(':()/'y+&QA':J$.`'gx(*i'0('z}(_d(iD(V_(c@'O|&]7(Tp(3)'eJ(|l(Bp('['nj(2r$iW'3,$fk(3n'{M(zT'oi&zF'Uu([m(N?(TX&X:'_c'[*'v#(/f(Sl(5U'Qk']c(p^(Dp'6b'#[(*k(#X(+f(RB(D:&@?"
"'3v(J|'HU'>f'u$(:^)9g'G7'Wk'Fu'h~(:5($6'>r'[c(Aw(0V'[J'Yn(PG'_Y'd*'5f'F0'qo'Rm'V('UM'Di($&&h8'^Y'z4'i#'eV'g='vF(4b':;(Xm'g^'h~'kZ(.J&XR(GO'3m(/l'{o(4O'hl'W(&sg'G<'CW'{k('1&i$'eB'bp'|4'e''Z^'XH(!u($R&u<'ny(3b(3/'x$($b'x_&B?'<l'3<&z;'CW(5((>z'D&'XI'.c'p?';:'Ky&~y(B8('1('V'!w'E|(<U'$>'`>'>C'oi'lq'Xj'hy"
"(>$'`b(4n'7.'S@&r4(-v&>5(S_'@,'kZ'Qa'x<(?.'GK'e5&,X(6a'Ge'`O(=n(3t'r^(;$&&t'Qg'a/'$B'Ip&ot&3,'Wr'wj'T~',g(4F'|v'{#(')'Ab$_O(E7&Ij$}t'EQ'q^(NB'&E'$N&nR(LT'eN&$I'nX(I['|R($:'(v&g((!c'f@'73'ki'lh'aC(30'?m'1^&kh'A@(-L&m_'YL'7L(<M']4&eu'hb'~t'zb&p+(?9(]D'c=(=]&M;'Cu(:0([Y(^I'|P&y]'??$MA'T5(=-'n+'/N&|*&Us"
"((/'h_'}x'i~&^/&a#'k7'|1'Ez(b:&}+(=a(V[(X/'M`'$h'jt'a.'m1'L?&n)(@G(3w&av'Zv'HP(-Q(;!'xi'xi'+B'.y'FF&tw'l,(1`'^P(0=&C`(=6(+d'H>'#h($5'~A'<w'tn'H}(/7'J''8x'P4&dk'4_'1d(*U(;$'Yk$u:'R6(9W'q-'m](Ae(*H(,,':B';R(4i'21(&l'2H&!q':g'DM'dO'KD&0E'dp'*}(Jm(KE&tJ()4'k8(A&'wq(YA'~N'}_(+@'|9&md':|'*W#j*'&n'_('eN'k;"
"'e4'[J'sL'im(Oa&w{#)l'$[';6'e4&ev'{1&}N(?k'Mq(F|&RV';=$th'4c'51'W7(QO'sm'jg&e|'7Z'>5&{N(Jj'q`(!W&[J(9,&R]'QX$z(&xy(Ln'Ga'y?(*a(:#'wW&t*(';&3F&lq'5<'5R'S`'vb'L<'+Z'I4&za'w/($-'Y~($W'.>'i8'_0'de's8'dI'r;'pc'p`(!*'a{'r#'`='qR'aW'_#'c>'bD'dr'iI(GU&S{'C9(<R#OJ($})#P(8*&$*&Qe(MO)dx(&~(fa(4?$9`'~+$hj#=k(A|"
"(`D'*X'6d(M?$py(3.'QI$oZ(F^'an'W&$eT'X1(3]'`5'R.(*w'&S'tO'Ja'-?((C'}>'vf'|}(5q&W('wU'pA(GS'Pi'7,'GY(+('~W'Ty'tv'P'()0'ka'LL';!(GL'L-$kN(*Y'Ts'&1(xP&91$pl()M&[}'o5&r_'H*(dD'l&'hU(h='X8'44&4n'{c'em'nK&,Z&ft'NM(7N(U+&oo'}Y(:c(*_(($(7l(:r'Dl'du'EU'bo'C4'NH'qJ'^D(#>'^9'C|'O&'oi'W)'<k(ci'4x)#0(kp'Ee'[Y&hB"
"&<g(&m&<G'g1)'B&qF'K|(pb(RD&~E'KL$e<'[Q'G_'ls'YB'R?(:~(D-']`($@&EU&?[(R=(2U(!O'w]'=A'*Q'z/'c6'b$'va'Y*'n:'rE'7L'X(&(H'O@'Qi&ci$i0(#T$?O)?''`2'Ok$mT!Wr(.a'pk$}4(;.'sZ':?(3W$Bf'Xn'U$$Q3&vU(=[(@J&;U(A1'4+#gU(Q~'oy'd9(|r'uv&TB'Rj'f>(L&&~Y'mX(!g(Yw'`V'};(Tk'j('R?&mx'Py'ox(&C'[3'C<'Wp'|k(-#'UR'}''2I(01'c'"
"'Cw'Sn(QA'G5&Uq&wR(F:'wI(EM'~#&BW(ww(G|(>>&sq(#&'hX(?o'n<(ZX$>Q'9](#Q'h_()K&mJ(BQ(^D'tO(q:(4h'Re('<&8h'vI'F3(7u(&$(No'pZ'r~'hl'yg'VJ'?y'aS(Q6('G'K|'YE'xD'Z=(8!'WQ(8$)Z~(V.(Ln'vZ'~=(o6)2o&8m'v*(+8&FS'TM(Ue(@.'J`'aS(?t'_f&ok(!,&zF'{0(v0'-u(!e(-Z(o7'-X'e9(<~'b>'y~'TV'FP(Aj'c9($('Q`'S_(,1'RD'@@&qe(L.']V"
"'vW($@'gJ'H@'=$'db(!.';d'lj'Bb(*w'c&'+Q':C(>b'0n&(?&MQ(AV&><&{u'Q`'x8'H{(>S'if'cB(85'6p'1$&sO'd>'Sd(Sm(2g'&`'xs&^S'xh&q0'>R'K_'Fs(0L&2K';T'H$'Q$'8{(I=&nm']'((a'rD'cv&S-(0n'Lx'SK&Tn'px(;''UR'M)'F/(13(X7'+#'i2'n9'^0'En'QG(EJ&ap'1;'g{(W6(Kn&19'4K'~N(H.'we()Q(^G'k9&|f(<m';p'.A(-Y(nj(?D&/L&4k&;7&jb'aM(Cc"
")Lc'Wk((+'>F'rc'Eo'vQ(1q&0L$xo'V_&^u'xc'S}&sK(#G'T,(0_'l;'x3(;F'Q4(+W'{,'<7&#9&LQ((1'4X(!j(4|'LR(<z(#I(/Y'P}(8j(;t'7j'X&(!5'$@(/@'i0'#X$y3'XB'te'Eu&j@'hW'6<(@Q'lf'yO&6v'K3'$P(&J('w'Fe'W.'4k($J'I)($U'g1'/o'63#sg(Pp'bS$xx'F7(+I'Ez'^K(+t'y`&@_(/V'/`'fO'DY'8$'N]'al&LI'L1&DM'[z'`K(B/'/$&kf&KB'Ea'`*'pu&j4"
"'mQ'*3'mm(*m(<I'vW(/M$n|(W#'wm'KY(U0'M='yj'xW(2h'UZ'Lx'xA&K;'/T'nt(4!(D,(5W'H^&j~(=I'kt(G6'}A's1'lY&xp(>v'^m'vJ'~D'lN(Ip(!7'^^'Fc'ti$X|(D''s]'UW(Vg'_w'hY$@l'$E(i5'<>(HU'I`',g'j$':P'T3'Ka'l['<M'VY(Kh(ZM'^i&ys&Br(6O(Lk'ER&~H(!''TG'ec'4.&w$(;b'[U'J}'BR'Jx'VF':y'$A'T$'rh'cc'_['cr'a7'aO'j&'sP'nB'[d'o{']R"
"'y6'ud'tD'om'iI(nc'XK'-8(tD'lk(!&'7#(5D'{=(P0(>k'R.'{2'1<(L7(){$y+(1?&XT&iw(C('v}(@N'cZ(=!(+<'fs&b1'db&f9'L(']l'dP'rr'yt((d(Z)(4N'Vl'Jl($u'Cl's'$p('Q)'p*'t&'*b('N'y='o-'m='gx'yT($0'gm'df'@>'dN$E2'W|'L;'^B'v$(6O(@='v<'vI'n1'Yb'L3(:#'Oc'?Z'nM'dX'm7'p~'m}&hs'4@'k((;n'q''W.'~z'zs'aZ'sp'(H&N<'E:&fw'j3&r6"
"&Ev'TE$k9'_y(DK'Le's?(kk(cM(C3']Z'L_'Oa&su()m&<^(,`'dG(1F(50'^^(*X'a!&}K&!S&Bd&'@(!0(+}(mK'nd'5y'wW'xk'TC'35'r5($3&@T'30(-9(*q'5d'~p'ek'-y$3C(5)'c0'@/(,G'VF$)R&w(&W1'kQ'Of(!9(e{(=m'DI'|e))!&j~'IH'wu'^[&>{'}*'mV&xM'aD'@F'q9'6#'FQ';o'o2'PQ(L6'$&'R<(/u'uH'|c'Pk&6`'ux']W(.$'qC()<(3''k9'l3'eP(Qn(.>'}#((1"
"(!R'p[&-?'yj'ze'}#'qw'b4(=?'X/'yy(1!(Ak#>.&.&(@P'j['4R'K~'[F'v['aQ'TV&r[('&(F`'yO(=0&9W'U[&-:('&'{)'~H&|@(BV'DU&(k'sK(_}(:F&G6(/(&Ei'l='i#'De([a'>Q&tg&x}(>_&It&>q&sB({C'oN'[R'^i'w['o`'lL'v>'}Q(&x'ag'a`'^5'jZ'e='PV'UL'Jm'Gk'j/'g3(&1(:j'qY(B.(:S'MX&yH'A_'WM(*p(*?'Nr'{q'[y'kS(18'r@&WZ&|D(Fv'u;'G3&w&'$0"
"']I'*v(Iy'yC'}X':$'|H(4p'|Y(73&vP(aD'j3(6@$K8'E8'If'i$'d/'oo'{Z&N?'P0(>F'Os'Uh(0g($t&F='|J&xE&mL'a-(:U(BX'nf'cl'8>&Ra'3('R1&l[(2b'u/'uu'#:&BQ'','j*'/,'TJ'ne'db(D?'uG&`0(6z(!^'_@&}N&s!'i''LX'OV(-2(D!(;2'2Z'}0'{:''X'pK'SF($|(,W'e-'0g&,K'yB(*:'{A'su'/Z't3(!.&_d(1v'f?'ou((7'J`((S$;+'HX'd&(*T('{'7j'JS'w$"
"'k_(@I(CR'he(2'')h&tH'ze'AX&q`'3:'J7'gi(4>'Si'no'`U's,(*V'JD']Y&yS(:0((P'bS(Uc'9x&L?(!''xd'r_'o:(J.(Fl&x.(M0(=(&r4',I'UH'l{&q:'Ro',['ZN(06&~m'an(##(49(0B'Zv'c;'!j'$l&kp'}o(^:(?k(F0'w-'vX'`j(#m(2d'&y(9W'6-'l#($z'46((z'm1''g&lz'9@(0p&F^'xN(B)(Hg'A|&mp(!h'N0$~Z'PK'gc'&R'l^'r@(@v'pr(Q<(dC'XB'Pl(2e'E{(=u"
"'oD'$e'rC'@a&XB&H)&ya'l='}4'o@(-D'cN'm/(X*'U-(Sp'79&`='[j&&W'(`&Pn(b|'A{(')'1z(5W'Ix'r:(Xq&JG'sV&y['Ut'f|'QW'z^'vs'cc'-7&q~&on&f~'4-'$-'<D(/^'Oa'#&'Y9'o|'^}'b<'t)'dY'w:'nb'bz'e$'gE'iy'o}'^t'oE'f:'iH&4)&Ey((.'<*(BQ(''&H4(?Y't5'xv'p:''s(kz'gQ$r{'tC(T#&19'yH'R6&,e$:H'wA(Fz(FU'^m'7]&`C'>|&i?&^Z(AP'dH($Y"
"(76(DN'jy&,j'd9((](&{'cm'7m'9w'rk'x&'4f'uP'F6'U@'|@'I-(H!'f''ud'}}(!Z'`)'G{();'fr(:S'x5'S='j='4f(8.'d3(!T'lP'uy'Yi(16'Q|&ke'}L(4X'i#'zK('.(2D'YH'c5'LU'n('u}'oK&P8(7S'Tw'D`&qB(*b'e((D}'8_(P9(8<$fF(Bj($U'xP&v9$>W&Lw'_r(V-&4''L;&vI'[H&-D'Jz'{3&`P&]2'}_(Fj&XM'M:&zv&2Y$vx&~0)#r(&.(;W(]~&n'&D-'Sx'|U'yZ()4"
"'QO(8.(58$tn'G|'Qa'x-'(x&8z'iu$v@'iv(4E('{$C!$}a&nC'yF(5v('m'i5'j<'oV(Ux(OI'{g'wV'{e(Bv&>T$86(*S'}Y(,R(*8(3c'z/(&j(!u'3}'y&(,G'w{'qK'IC'q*'sj(2('mg'lE(3*(-L'`P&t>'*n'Zf'|P'_Z'~&(/1'}P'wD'|T'K8'}?(,y(*V'U5'|='Bj'@$'rn'LK$j^(1j'[Y'k.'_#'Ma(#v&f['yv&ID(+?'p[(E~'a,'x/(.J&Ov(;{'r5(<H&d3('G(0$'zv&Nl$4A'zR"
"'2Q'^#'C}&bL'p?'?3'XV&6R$zO'Ow(G?'m8((L'h=&m.$o>'_P'{U'wD'{c'h*'V9'lB'RZ'nE'`I'n]'dK'V:'ml'nn'`4'db'FV'O''Sx'e|'Xe'kh'yA'yI'(=&t{'P>(6{('n'UH'th'L-'e~'{;'|)(-K(AB'w[(&;(4;(HK'*@(B[(9n'X&(Ts'r_(=t'JE'[F'yC'p0'mX&bn'S8'GK(1>'XW'l&'gA'fN'^C'Q_'H`')#'n?'TK(;a'HB(2r'iw(/$'66'Q>(7#'7E(1z&L_'NG((M'^@(.`'q*"
"((p(:s'1v'{Q'~o&pj&o6'm#'Mq'ao(<t'fx(pG&{;(!b'h~'gg'j<(/J(0['S#(Na'pn'r1'}](49(m?'j7'^''FD'sD'mf'uq&Fc'cq(5f(-#(,,'GQ($`'Xu(X>(`d&tN(3R(<;'~8(!q'w4(8j(R0'~/(?Z'rI(:u(#P'Iq(*&(<R'as'o<'ho(!_($0(+A'/?(=^'qv'Tu&;E&W3'j9'Q/'f,'[E'xK'jy(}5'Cb'h*()f'i6(0g(-Q'wj(#''!~'[Y(!-'V;(3W'yZ'_H'wV'R_'z)(LJ'xY']+(:`"
"'|T&r)'7H'm)&[d(0y(Ld'_''fA()8('G'o!(=p(9d'dB(&E(/4(<g(!u'hv'Uq'eH'^_'.a'Ez(&f(:|(@.'YO'-L'WL'qD(3J'}p($I(!~'WN('m'<G'`T()+'b*'e@'uk&fA&]U(1e'k`'[c(B)'~c'Wb'm*'{Q&sW'a|'m0(4M'vA'gW'Kh'e2'V9'xO'qn'zt'(.'k-'H:'|;'8O'[n',d'H7(#_'zx(Z8'8Z'=I'lz'x;(,l(eb(,0(9G(/B';q'a''D1),)'aK'hK&~}&dA&{t'&9'#-'pi'I|'&q"
"'1C&g?'C_'Iz'2N'=p')P'7h'6J'mX'sK'k/'j;'qw'q+'iq'oe'i>'ou'h&'b.'cH'o9'n_'p='`U'iH()A&0#$;N'|l(5*'~C&o)(:v'qD$`C'6-'}X(F7(Iy'io&xk&1X'g.&HK'jP(!h'Sq(Ri(.H'SM'~.('[(*T'y}(.a'q7'1T$t+(<-'P@';D'|2'~+&lv(Js'PG()u('q'yi'ex(5F'a!'}>'oC(9a'os'az'hd($a'#=(PV'Ds'|]($!'_s'?v()Y'Wx'h`()p'SX'sj'8t(<|(&''KV(TC&hw"
"'`s'`>'t.'Hi'ua'r>((A(0x'=J'y|'Io'Nq'i#'mK(XS&v]'f]')R']u'+>${L'T{(.<'y;&{t'T''op'pU'qP'KH(M>(>i']x$sB&Oz'l+&98'zZ'O[(?&'up']V'xB'ZV${H(;}(-(',`(6s(F-$N~'cH'JI'-J&Et(-''~'$S{'Jj&jw(0E(2l'gw']S$h)(XC'lK'>J'qU')8(Ip&3{'Nt(C}'cq'X9'y,'!r'lY'~P&#e&Y~'p<'#E(#g']a'B#'Yv'$e(Ay'wz&d$(.z'Zm'V>'jR'GS&9#'r3&y|"
"($M'>2'r>(-Z'?^(BZ(#!&E&'|Z'~V'&P(,_'wR&eW(*g(*H'ea'-E'D8'qp'k!(0Q'Qq&p|'qq'xS'9f(>z'pc&du'u!'43'xj')}'7}'qo(1I'w['h3'7='H4'T,'U4'm('sl'Ew(?`&ft(70'@#'9r'xe',]'Pz'^L($V&0r'L]'x`'Zw(18';x(Aa&!V&#-&9$'R*'n$(Mz'i$'w9&Ol$ms'bw'K`()x(Ri&ri$n<'R4'=/'vu(0/'dS'SX'q-'-c'U;'fz'Ir'w}'NS'aX'X:'h8'Kt'ef'M}'Q}'sW"
"(PT(D`)-F'9u(+3(98(/}'^N(08'~)'U*'l-'Ra(h-'(9'{E(?o(7I(B*('s'Fg($B'b0'}b(*a'}p(Lq()F'LM(.4(]v'z7'g3'y)'lh'[o(Y`'v,('['io'T.'vw'[g'm&'Y8'D;'4O(&F(Bh'ud'{A(*j'jX(1Z(!](>:(6k(PQ'z*('#'l1'dc'w&(@$(!T(dH'*G'?S'Mk((j'y#'VV'ls'ny'i7'|i'LA(9^&fB'l['I_'N1(@t(qA&`z'f`'Mo'XH's7'|`')W']q&Z!'!^'Zr(#~(E8'e}'eM'-H"
"'Ks'oR'yH'v*'o[($L'lg($O(8](5S'qR'cu'go'ZV'/e'Wg(0D'g<'lI'qf'pc'l]',{'l;'*m'Ht'Kf'oE(>u&~='W6(1j'Ah'u='{^'f['z5(&a(*q(!v'U$(&>'h1'^v'Ms'H?'hQ(lY&g`'W-'yE'}U'qE'o|(ST'nS';+'d,'.F'w0'jw(/d(g>'u~(!((T-'Y1'w2'`H('E)D;'`1(5W(+v(im'c}(/_(!/&{<'>'(Eh'R8(!V'o#'|5'og'7a'yN'L~&qm(5H(*x(E[(:5'i<(FS'y7(Uo(&)'ns"
"(c;(C*(*H'|f'Au(@a(ep'pk'a2';P(TX&i,'sS'uH'h9'^E'P('_s)!i'^N&mg(dp(q5(BE(<c'VA'L3(LS(#M(.R'~0(#V(E('h,(0t'Rf&lz()O(Wk'vG'r'''e(8'('a'K.(8|'xL'[I'@A'ux']U'zp&[w(v/'TU'AV&_w'F<'h.'U3'a''(a'.t'?e'4U'5h'AP'7!'Pw'@M'=b'iO'o2'iV'rt'd8'v3'k!'g_'k+'h?'h#'vh'iI'dv'n@'n|'iH'}Y&HQ$Ur'I}'#o(M;&rG'l,'k]&s$'q}'Y)"
"'vG'|@(Gn'Xf&FM&_q(M/'tD&8i'9F'd=(8Z'Qa&pP(/J(4Z'z0(,['~w'{b'iR'c9'c)'k2's(&Jd(#Y'yj'f)'>9($F'+v'/2'jM($A($_($~(@k(3D'RT$sL'C''r-',s'tp(Hn(:?&Jj(Mg(?h&Nv&E1(4P'`z(#7(#S'wa(-.'{3'gd&qE(No&0V'gG$d7&U3&Ly(Jj'i[&=4&_R(V-(;u&jG'2/(A`'s<&l/(4~'mN'_g'&S(.5'^J&;U'od&W{(X/'V1&[b'|?(8x'mZ'oc(3s'Ew(4U&q9'l9'+X"
"&W5'Kq'wH'B&&ZU',@'b#&Kc'b.'<q'c3(&A'tF&PS'gX&rc(8w'}X(J&(,Y(6*&f8(<Q&Rm(#H(8-')~'aw'C`($?($~'7b(A0(T{(1h(wT'b$&l3'qZ'$&'q~&c!&V|'*&'Fg'tb&(V'|i'd6'Sg'Zs'aB'Sf&3P':u'dm'7U'cu'V-'MY'Q''Qu'bt'm_(vc()j&F#'e|'m('m^'h.'U6'uK'03'sQ&Bm'js('_'mH'<O'uA($Z'U|'wI'qb'uy'iD's~'hI&i4&lZ)r5'~-'e^'sw(h<(e.'nB'v$'hk"
"'q$'`8's)'yi'o5'cr((s&bE'q?'sj'y1'Gd'U)'cz'c;'ot'pn'm8'l/'da'aN'Qc'|k&5b'qN'q>'s<'X(),N'q*(Ru'a8'w2'uf'Jm'Oo'QI'z''{`'IW(-{']B'rA&p$(#&'q('n0(?H(MD(8D(I](?{(/|'gp'a1(xU(W!(fv(lm'fZ(#l'fi('f'Zr']+'M:'tE'`I'{.'j#'y+(:e'9s'L0()H'X+(`Z'~;&xg'gd(&['v;'tD'}((3{(6h(E$'C$(0b(+)'v>(,w(+U'}''xE'l>('[']5't5(Cd"
"'|`($l'a9&v2'|v'uJ'x('Wm'Rs(1q)2*'~>(!Y'~1'y7(&d(4q'+D'<3(Lm(JY(+!(9s'Rz'5-($~'ZF'oO'h0'aa'hN'nt'oY([H()p'-`'^_'^b'`u&{I(5L(5u(HA(&f(8*(.y'vy'~|(`&&0u(6B&mL(Qh(;/(5Q'&f(:&'zm'xv'~H(2='|U'wz'sC(9$(,L(Ms':}(3@(!k(&1(LG(5C('Q';V'|b'}q((/'{x()W(6)&gH(*N&m>(2_();(3g'&s&T,'Q/(o`'t<'t|'k:'il'{O'XF'pb&~p(P`"
"'kb's3'so'Yb(rP('j'Fa(&~'vQ(*4'yl'z'(C/(Bj'U>(}d'ku'bB'{?'J;'.>'D#'M#'L,'FC'M-'MX'HP'6q'O)'h|&kM'8C'BD'=W'H2'a#'hw'o>'j('jM'j_'jN'j*'h5']2'd+(W2'k8'j@'mF'r6'iH'`j'&5(<B'|s(/}'Sj'j4&ba(4?&:!'r@&HM(*H'bS#Hk&Q,(2{&]]'cn'}0&_1'{B'p<&|n(,8'@5'ph'yI$zg'Oy&S*(-W'H_(Oc'Px'=!'lW(*l(&8(7K'W#'JU&g0'Xh&O/'_`(9y"
"'sB(/w'ih'1X'@!'B|&zy(BX(8!&Vy'S['L.(@n'vd'~$'vN'zf&u$'Xx'IB'{:&Br&UB&8{'EO&Kn(+g(/D'qO(5#(PR'EF'{1&R:(/k'm$&a7'0e'xJ'kT'tp(>c(#p(Fi'YD(.:&7(']@'|z'aV')h(b3'81&Rp(6M(.-&&x($q$hN(Kx&NG(*z(C?(K/&:S'}W&$e(=='8F(&m'i=((m'~W']R(/D'h[&/q$sT'}R$gT&L-'EC()i'V''gD(1r$vo'pm(7+'|H'jX'zz'WX(/[(-e']_'br(B}(9U'nz"
"'WT(Cb'aO',k(:B'L5'Y:(G`'+d'pU(1w'WF(&E(,@(D)'yW'W1'<X&VJ&BZ'E.&&Y&RC'pY'rs'}<&g7'K5'br$'^'df(W/'xJ$W9(:P'zQ'J((T8(H<(Ug(<~'~6$j>(9-(?='bA'o'']}'62'PU'Y~'_L'gy']''K5'Zt'Za'[P'v@'Rv'TC'm8(>G(o!($g's}'~]'{l(RV'p{(@~'x$'w~(??'yi'5U'7j'U3&P~'b?&f['m:'l9'vI'MH'vN'kd':s'ad'/*&|{(Wr(M;'nA&n;'Xa':S'qF'n['ly"
"'U:'*L'>|'su's6'TP(:U'ha&{L'A^'tC'[d(+}'OW'C~'ZJ'cy(IT(Ge(cl'zm'x:(F<(#n'qA(2W(i[(f+'Tf']*(*3'UU(Kw'K}(3J(s.'g[(E,(Qt(@5'BE'vq(2$'No&nV'hA(+|'^Z(NH'uY(Om(Z5()Q(!>)6f'>w'6L'{/'8m'3,('5'vD(!>'v?)LQ&Mn'.;'So'jH'Ur'mO'L:'`|(1S(L|(ZJ'Kv'|q'oG'}m(RC(=((7q'A''z9(O+(*n&bV&1I'[<(:)(T7&eQ'f<'s#'w_(F''B4'{F(qH"
"'ts(,z(_Z'1s'nv'Ek'kx(J^'|U'_''_('b!(4F(+t'YF'}S'`0(&?((e'MQ(6)'j{$y0'R('K?'k/'ku']?'VJ(1D'aL'Zf'hR'el'i6(#]'mT'uT'z7'Oq&Uw'vz'q~']^(.c((>'3M'3P'U'(BR(1E(3b(5q'gi(P)&T5(/+'vW'ko'Vy'I!(PP(ui(sq'ok($2'tE'7o&rn'kE'B!'E<(XO'lD'v9'n1'8F'Yh(D@'C*'qR'3h&xt'9Z'yl'HM(RC'J4'fY'H$'5i(;t(Py(1P(]e(Am(4g(G)(@O(&<"
")oH()3',G'HE(.c'p='y`'~,(.6')J'cT'iT'xs'dO'sv(F6(+_'CX'/|(&i'R2'Gn'B?'Gd'7n'9m'':'/6'Lq'1h'#.'UQ'f;'g`(RZ'r,'VQ'j,'j&'iV'^k'q}'ne'o('k('b+'to'dq'd5'iH&t}'Od')g$rJ'iQ(!@'zs&1g(0$('.'.0'YT$v_'^s(B_&Wm(;l's;&dn&HV(<*$jR(!1'^@(0y'n>'Id(!x'OY(7s'X)&.<'<T'~+&6s'`9'zo&HR(0I(6{'qn'u&'TF(4O'vD(0v'HM'v.'V#'fJ"
"&q'(*=(&f(&M'b](-V(3,(+X(*/(0R(.n'i1'Mx(+u'wy'q*&o?']8(&['jw&f0'bA'AA'xT'`|'y:(80(+|'5A(,h'PJ'6-'$U'^Y'|/'`s$xF(+u&fj(1^(Gk$zH('z're(53$md'De'Qg&zw(=*'k3'hk'H{'kD&C)$|i'B,(BR&]`'lA(Lk'{{&H,&N<(E^$[#'*''fH'}S$v;(-/(;k(,_'|9'eE&`H(iK(Qz'eq(_u(+J(,#'wy&>!($Z$Y-(@K'n0&b('si'3L'+W'[5$e0()J'g['wT(1@&|)(@5"
"(;N'^C'QN('*(-S(3?&B#$`N'^['kb(2-(6T'q,(4k')A(:v'iX(3g'nD'w)(B0'gn':{&_-'t5($C'SP'G@'sp'NZ'Jv(#@(7s'rp(!h'Z<(Dt&iq'_-&;x'Mn'v+';1(,`(.!'Qm'=a(&~&#X(1y'ri&X['bT$a3'N-&`B(3*'TT'C`&JA(&y((m&Pf'Uk'za(2^&$-&*((:N(<_'z9&Y,(lf']n'Vu'ey'N*'n!'DA'r*'dM'pc'k;'fw'og'QK'VN'Uf'Gw'Sm'?#'D|(QQ(Op'uj'qq(J''52':F'-e"
"(#j'xo'K3'p+'L}(#M'tC()?'y_'ng'hB('I'[Q'sf']S'H0'jk'me$~>'z''=<'f$&pO&P#'aF'k-(=E'kV'[q'#r(!D'jG'rC'K|'yB'TN'qD'p0'aC(F<'^A'7D'hV'Vf(:A'rk'[R(=='{r(!{'8C'u9(95(;1'n!(*>'#6'LX'R;'PR(#<'+I'so'2V'd&'b&(9?($Y'NM'v''X<(VI'h,&~t'j,(*.'`u'>z'F-'?l'Qi(#w&p#(78'LQ'C,(lQ(U^'Tt&|h'b+'sE't''e8'~^'t;'X&'dk'0}(]Y"
"'{](I}'I+'+('^L(@s(!t'88'{s'v_(#_'lo'g/'w+'6I'Ki'vt'W''`v(=b'@C'?@'Yu'U-(#k'7p'6`&pF(!j(,I(/_''E(4F(3K'JF&JY'Hm(A@(&R'd~'Uv'^4(+0'iK'{m'yw(1$'7w'{('4d&fJ'rq'cl'p0'G_'oq'k&&ox'}Y&~w'm-'^h'G[(&S'K{'8p''f'[<(Wf)>T'T~'r_'hz(@-&#0(4&'rQ'{F(?9(`,'w5(/;(8p'YL&n+(H6(;f(T3'gn(>$(2h(4w'Z1'iG(ZT'~?'n@(+.(C7'Kt"
"'7&&Km(P2(6q(>b(Oi&2h'ph(F_(95&4I(R^(BC'Ds'{o'DS'o:&w.(.P'b1(BK'-9'~~(2o((A(BN(+m(&](!`(7g&kz'g7'Ts(9/'vp(C#'g)(@;(1/(a`(ET'Pq)aR(^h'r|'tN((u'ol'F~'iw'*;'0e&xf&}n'aw'1L'+/&|&'4A&^1'0P'#T'oW'tA'^R'`G'go'k#'h+'}E'b#'fy'k9'gm'cG'wS'jr'h('iH'^)(l&(Tu'P3'EL'>K'!>(uZ)#S'}{'5W(FR(&=&U<'@Q&E!&96(*{'WF$RR&lp"
"'gN's_'^~'*R'/y'4K(cd'jc(FP(<6('P(b_'o{'xm(/p&wz(H^'Oj'/~'(Z(!#(ZJ(<7(,-'jg'M+(jR'LK'pJ'9c'Vl'Mg'[@(7)'zt&xd'k:'h6'g0'jH'oK'px'(Q'#H'nN'TB'vf'k,'<A(Ek'XK'FA'^V(CT'-S('u'Vz'@P'YW'tW'a0(+(&z$'ls'xb'yM'<-'OV'sS(+?&Qg(-<(>A'=+'E`'z*'TQ'^A&PZ(Kc'Se't&&lo'fl$sS'jR'6}'Al'Ec(yP'Uk'sO&id'Ij(S](|k'O4&h.(+n'7["
"&|C'`,&l1(wo'02)12($V(N^($R)=U&d-&kp(i{'0#(x/'RN'1:&?((4q'ey'y]&}<&Lb'59&^S(yt']_&kR#{}'$Q'Z`'~:(8P&C#(5.'p:';f(6]'Wi',x'o)'e4'dv'v@&c](-l'Xt(*=($?'CU&cM'[B'Ol'L$'SK(!O'`F'IR()<'g|(&^'_3(4T'sS(5A'?N'nR'Jj(=6'eX'i&'tu'XH'YA(A0'w_'b!'{$'ub'i@'p7'}f'fV'a<(8z'B>(3r(0~'fl'Qh(+m'lj'A9(?}(RQ'l?(2@(@P(A['eE"
"&sW'd>(?s(<<(_2'WZ'0X(E}&Ds'oM'U:'Yo(cc'tL'Y{&ia(Ns(V~(0W(OM'9Y(/<&PF(e}([''/5(7|'/L&+~'m?&pI(@c(8Q(k8(cK'bp'N5'T}'o!'H]'Y-'pn'_e'VM'zg(&)'^B']r'n@'a7'^a']h'T](2/'_Q'~['ud&~$'T@'*?'Qi$gq'_b']+'Z=(@b'&u'ov'Yv'p$'y!'Pl'F4'rF'7r&uR(8H'cD'oe(9)'ll'LY'Zm&h@'F3'wD(#Q'[?'t2'<5'~a#eS':k'WS&Wa'y3'Iu'ZB'|Q$uB"
"(]i'nk'o{'pa&{7'A1(Z^'~c'kP'P,'fZ'B&('I'g=$K3'Hh(=Z&I<'y['.c&ph(fF(*g(I<(MB'US'D''{[&81'++'WS'J{(!o((}'sk&pG'fY&lR'~5(AF&Sc'U_'ed'jA'wA(Fv(,T(L.'AP'/8'tg&Q~'6d'x$&n:(/('lE(,~(+;((X&{''px(F*&hm('-'OA'r9'7M(,x(6i&FR&z5'Gf's2&L_'hm(GZ(#7'Jm'3V&pF'gZ'rD(_a'|C(A^''L(AC'z^&q^&~+'zH(Ki'wP'ij'yY'+G'J$'pg(=H"
"'zR(!v'sV'EB(6D(/i'<L'y6'@1&b2'DA'sz'N:'fN'uU(LN(5Z&Oq&nw(#j'i{(C0'n|'bQ&8^(MA&/d'`'']:&cb'i=(A&&t['_}(5s(,s(`'&W=(E~'`-(*~'Ke'aP',^(:{'{Z'xA&mu'L3'F5'Xh'{['}Y()''V&'Uq('x'/i(0+'/Y&XZ'G@'n5'_/'tv(&L(6W(!a't#'5N(/G'ZF$sG'!V'u''y5(4B&_b's}&Z8'tp&hn(Hx(-='r('TC'sQ'l:')M'c{&/4'E{'kw'Mh'n_'p4'vB$W9'qx'|R"
"'59&=^'n.'t_(6~&s>'XH&~3'u/(E]&xv'.!'c^'f('XM'Y`(2V'{d'Zj&gk'8z'@v((g'_e'sY'bu'j6'r:'d_'uc'rY'^I'q5'tg'eI'gc'sD'aZ'^#'iH'B='Y0'jt(2X'(2'[F'xI'^A'XZ'=*'oP'{o'i*&}p(6`'ZH(<m(-*'KO$5X$1y)#|)#F'9j'Qq'X&'3M'|e'Ze(9f&P|&y9)6`(2F'RS(up$O{(Ph$|F'`c'g>&q3'UT'1^'oj'O4'[q&wK'e6(EA'Xa'it$kV'ML(Rv'Jc'DK$=C'j^'y+"
"(83(25(C<'Qe'^w'Cx'FD'h~'TK'BK'^&'Y{'pO&RU(n_'cQ';j'nR'<'(?k('0'=F'uS()J']X(,R'Hm'pA'O0'Es'h^'bz(O[(#d'kL'|l'+h$2p(WR'ny'm!'fz'm*#y5(Ts'UV'Ye(`y(CW'`b'OS'&V'h*'/x$w{'Xo(*J$i?'Le(i[(kE'yY(/A(2('ll(8W('T(h`'*{'cq)(:'7S(Uh'2X'3x'5P(k!()J'+>$u;(XZ'vI&9A'>+'mi'Kt'{9'sq'se'rn'jD(R/'7k'{q'T.'n@'T7'b9'[?(-="
"(CK'_f'Z@&m3($F'c8'k:'R?'r|(@B'z['{W'Xr(3='Q['<U(VD'j;'es$'r$4a'!0(Vo'xp'l/'S#'~J'K0&]~'z^(->'a((kz'ks'Ud'Qt(Tv'ul&mL(FV'i=&bJ'1W(XP'ns'W;&I7&D7(5D(.W'?*'g4(->'WY'(g'YP'>4&cS'JS'mu'D?&w9(0]$tp(A((&Z'Z{(0F'jJ(9a'z+'d](&8(|4'7V'oR',$'lE(,S$o+'s0(>H'qe'u<'k<(7*'F,(2;'kO'X]'1S'si'_m';D'jS'i3'+N'nL'$b']k"
"'`b'g='[0(V$(Kp'JI$E('i^(!5&|q'jm'Z>'AP'9M'&>(2U'_}(.8'Tr'dM(wJ'q;'Ra&{M(}:#zW(L6'2x'OB&l7(`x'dC'e&'NP'i#'kK(py'{A$n+#n9'Yn'Q`(,l(7e'Xq(2L'g8'T)('f'iM'Tx'=G(/8'y7'dt'!V't['Zc'ji(+A'=d((<(-V'{x'2u'MH(.s'3K&:&(A<((s'Wh'B''<#'ms'rb'<D(Bw'v|'?X'Tg'y0(0W(R7'GV'dE':l'IN'Kg'59(*i&XP'rL(0O'7d(!x'DV'~f&k3&}J"
"(!D(9='V4'}d&O/'?J(@A'wg(&N&r:'W&'-x('o(+u(?!'qR'QY(Hs&X_'<t'xa$x|(-P'zZ&jr'fq'{y'9?'F)(DJ'RD(>('w-(K7'f|'VL(Kg(86'pJ(*k'yn&@I'wx'XG'b]&sF&Mc&pf(.M(d7'e|('r'Ba(2D(qo($j'hA'Rb&/>'s(#lJ'Q=()s'mY(,['Iz(7f';1(CW&4@(!N'6N&<E'ka'x&'=j(G@'wI&e6(Og([B($T'sK'R?&M:'E;'pO(Mh(OL'}D(2c'ZT(!B(@2&(v'QD'KV'on'g^'pg"
"'T#'Ul&m2(b#'eo'x>(`i'/N('2(g;'2P(/B'}S#k+(`n'Gq(;z'lt&wU'qL$@k&av'.P(dP'|@$Nr$KV'Ck(-v(5P&fH(<C'|z([e'c*'n=&p@&]>&YW'_C'aI'dZ(<f&}I'vT'rX(OB'j;$|U'Bp'+y(8*&o2&A|('S'Vg'fa'Y5'qB(Lq'dE($k'fS'J[(B}(r,(>c'?)(/F(Y$'Vi(.A(Du'I/&ns'f)'uv'(h&fH((k'7s'Yf'=C'w|$N9'SI'9_'Xx'r-'zI'=I'e*'WI(.T'eM'h<(-i'zN&xV(b)"
"&&v'<d'nK';i'^Q'TZ(2['1J'uM(&5(-z&r-'Xd(A/(DA'^5'KY(Q<(<f'E]'z!'ng&oE(f+()X(9J'X;&CO&w+'Y^(:o'}G'FK'A9&gR(]/'E!'^|'G:'<O'F+'{5'g'(MO&xg'2`'Xx&GS'o$'[q'z$'rF'bE'`5'U5'bD'fq']p'b0'sw'or'^~'u!'yl'iH($C'~T'>((QH(m:(rh'r)&T{(Jh(aC'tY'Y3&xM&dE'Zr(j)'d$'Z-'Ix'|p&8c(8+(8$(OK(9s&Z'''r'HX(5}&}/'Mu(Z3'}q'vL&h1"
"&q:(vz(07')3'*e'u](5](2[(OY'a$(Uc'OZ(<!'Wu(-a'Yk&}I'|N(?R(OU'bO'k4'9K'{b(P!'YF(,T'F''b''Ld'!7'b((S:(Az'cc()L'*{(VS&|O'RD(<>&i`(Wt)Nj'Q$(&0'E}'c)'n,'i''i7(lk'T$'g4&>k'dr(c`&cV'`G(gz'[,'9R'Kq&hq')!(A/(5)'dX'Ka'f@&'i'X/'Tv'{g(/3'e9'lC(+2'pY(Vc'}P&jP&F0&ax(>B(!f(Sd(&('f9'S?&^6'>#()r(*s&S?'OY(?4'8K'+i'o:"
"(5m'Bw(/H'ot'0e(A:(#m((i'NQ'~)(W!(!z'ky'ey'4~&T{'}d'Yr'ok'qA'D](!M'`r(>y'SZ(pK((d'7B()y(p,(Dp't9&lY(HK(bh'uD'_''(l&WG'QS(!-(,o(5Y'(:'d@&zl'C_()W(q((J[&`-(r5'M7'#['wZ&m^(a='Rw()<&nW&Q}(D3([q(#L&]&'a)(CM'u2(g*'yl(Vs(+{(`Z(?&(0e'q-'`](+@(#!(29'hA'UG'7r'd?(AY'R/(6}'8w'9G&I('T5(_S'Dl(1P'^I(6b&{a(ja&mj&Ce"
"(A2)5z(Un(Ny'/C(,-'Ek(ai'Wt&n$(-U(6y(P|(2)&xq&[H(]U'XE'3`(2>'Nb']U'k}(Bv'Yw'{u(4-(3>'8V'`m(J<$M8(1}'a0(C~'Y<'L*(`C(<N(@B'U-'&J&r{&+q(UE'of(b:(@G'_f'P@&sq'xT'jR'T('v)'5d(*4'Q.'l|'07(9m'Mr(+E(Vq'+@(&I(,)'tz'4C(-<(E{(@G&D-'X6'Pk&<?'|R'ZW'l{'UQ';Z':k'n/'sD&sW&xt'qO'>Z'{s&{V'dj'O,'Pc'*6'W:'7L'8@'BI(#}'@`"
"':Z'f#&|L'e1(YL'x]']7'o-'7o(7N(Mh&Ug'w,'9>(7g'{j(8j'1s()7'ed'I:&?&'=[(<>']P'.F('D'-G()V'3_(9k'qu'Se'gt(Ob'6M'`O(AS'^!'|Y(Ho'Fr(F=(/b'Gh&kr(.M'`:#Gt'z+(<(']~'o`(!S'G?(1v'iu'p)'Zw'f:'H@&N?(Dx'ed'?@'Qb'+@'J=',w).7',](-2&22(5.&Vl&AZ)RC'kf([i&31'a1'c$'I~']z'w(&o@'T((k?(8i'{/(#y'Z8'@;(/P&t5'7-(Br'5A'FC&lT"
"'J;(/e(M>(xX'Uu'hg(0='>U(WR&[U(0$'RO'L[(86(4|(F$'s?'1E'rW'rM(#T'I|#eZ'K1&IT(]O'h}(2s'wl(?1(6l(1:(2S(BA(]2'Qn'cc'F5'w7($9(ct(AA(5U(?q(9>'eX(m,(3`(a&('y&+L(6!(V4)'6$om#z*&bw'fE#br'oo'SW((+'kK'e6(4c(I.(&W'j1'24'jz'wq&jO'j('^7((K'p$(Q~';6(&F(!~(70'Ey$?H'@n'ef'K1(.;'r$'ZF(4('R$(!o'NX(+}'r'(,Q(QH('b'W~'C2"
"(B3(4G'vq$Rk'YY'X&(e$'Cr'_P'_D'~i'cK&{s'h>'2z'jn#5a'[h'L5&<d'Tv(E}'6q',D':?('H'f2'[O'n=&_1'}p'!{'o[&}!&|?'nK'n:'[?'@=&y`'qW'aa((M'J['hr'Jp'@.(#!''_(Aa(90'tT'Et'^*'-0((2(5O&Z4'Vh'ML'sJ'h*'uN'B|'bv(2~'Vv(C:'KV&Rk'5&'|O'mk'}#'u<'jR'vw'qy'LJ'rX'bQ'Xk'^9't&'^f('l'u+'d`'tB'iH$rO'].(+X(I|((2)'A'^^'hR&/p'?L"
"'Y-'2_'D7(/$)&x(rq(^r(&Q(4j(Ch(N)'kY'yC(A7'oH))a([9(rC$ge(:5'k!(5<($+'~Q(mY((L&Rb'T^&z:(<N((Y'8Q(.W'Jh(-?(I](5<'qM';r'W2(<3&_R(?0'e=&j)(R^(A9'qf&!e(F1'dL(<@(1g&Z=(:t&AY'pg(pH(On(1,&x=(PK(=^&S#'p:&K](!c()Y'o`'kv&!('X''|X'u#(5H';&'6>'rD&n'(G@'L}'A5'Rk$lc'+j(_Q(E>'dq('~'FV&ee'|9(@z'{f'*s's0'KJ(|=(YW'5}"
"']7'@N'e['zi'4]'N['eM'r>(/d'>G'h*'RA(@,'ms(;h'QL'LP'W?'w4(dd'4@'@K'nA(#z(?/'3M'a~'u<((/($d'US'qV'iP'`r'm*'aN'hu'I<'ay'{.&_g'p|($X'4-'L['eX'U>'rR'5N'Ac'hy(2`(6O'25'^v)$q'6@'c1#zO(D~((m'sa(,,(w4'N/)Ho(KA'wY(GH(PF(`+&OV'r#'g;$hX';2(=H(!7&P=(<$(-W(3P'{W'gZ(_b(?k&U$'[S'Ec(z@$g9()t(HM'j2'|+(LC&42'],'y~'AE"
"(IB'yR'tI'];&}n(R}$Bm(9k'7,'UU't_'y@($T'#V'd~&##('D'Up(@L(7_&z<(Tu(`+'jB(Ex&}?'Y$';V'~''`I'kt(B1'xs(`f(/a&lN'1B'|v(^`&E='e|'f8(9R&2x$|]))G'YI(+&(K3'O(&s3'5/(H;'eP'Cx'26'zf(.;'l@'o4'gU'uo'{3(EY'X`'^:'/)'U?(2c'3i't;'eP'PW'v`'CA'H8'N0'dD'}7'j.'K!'HI&XN'lZ(ki'`)'M?(.A'm['~H(3W(<;'Md'k!'Z9'9.'{;(+>(50'S&"
"'Ya(-o'F5'bb'TE(?h'wt'8[(+P':~'^4'w4&F+&mA'E3'lr(gm&Lf':~'D}'AD'v2(.w'*''Wu'N*'v0(D0(9$'Ml&}P'^v'r~(#)()*'4Y(+a'lX'w/'nn'Lk$:`&L,'[['oY'kc&mZ(;K'`4(HD'>L'U-'}_(S?(L+'OE'_v's|'Oi'v>'*K'zI(4|(3j'qM']D'u?'R@'g_&}A(H,&<Y'Fj(0E(,q(E_'3,(;{(1d'AC'S|'ed(H6(ZM(WX($8'[H'r''oL'c8(:p'pu(S,'!o'y='fK'??'|*(:b'1)"
"(8B'f,&h?)$b'90'xf'6p(6u'{d''2&`o'ew'}?$~h'H$'}<'6w'w0&y|'qQ(6>'kT(L+'ev'aY&~s&q+&~r']C'4K&st&y$'(S'IV'Gf(#C'1#'he&pZ'x<'l?'LF(#I'X0(##'1v(:5$w1'wU&O!(+g&jB'`|'kl(5.(8a(<t(2l&3?$n}(&!'mi'E2($]'P1'0c(a-'Vm($o$d8(ND&eB'O`'yq(^&'D]'xX()W'_c'YR)F[(U@'|i'y/'ip'=Z'cu'jI'Ux&Wu'Tx(*$(.V(.Z'r!(={&WA'[#'^p(MC"
"(TX(3z'Y('=d'|(&.X'c1'kb'rT&}{'}k'?;$=`'}?(#,'}>';S'Un()d'Wo(#'([E&yC'2:'~t'Vx(-.'az'xn'Zk'E&(0{(_T',a(9A'|['^Z(8`'hr(`('YN''L(8S(,&'Mp'kB&ua'<!'Ok'3/'[T'a=''^(6~'_u'U=']0',m&qR(4D',k'hI'X5'?''Z`'8q(Js'#g&6l'aa'sr'uL&}_'x='xd'XE&p8'3['s#(0Q'pf&#W'tr'i*'pi'JY'6@'3f&nP'?1'RH'F*'r1'bg'JZ(*x'fa'ta'qp'db"
"'c2'_~'`='cT't!'d]'ax'dn'qW'cM'bb'`}'cr'z5'iH&l0(;/'wd(aN(ba(&p(u8'}i([8&[5''_'J/&7]$|I'x,'DW&bq&!m&d-(6@'1&'=Y''Z(#.&rL'M|'Bk'WI'!_'CB(!b(*9'sa'8a's`'to'oF((@'rV&DA'[`(A_$FM(Qg'b8'/3'5('cO($B'wC'th'V}'jR'^#'`&(2k(*@'hP'_a'}o'x:'eX'cC'Ko&WT'=j&-Z)!h&+s'j6'td'O='4w'q$(PU'Y~(dh(WM(=n)!;'B''Ue(@>(:^'om"
"(M'(<7&S<(3z(Jz(?U(#p'Q((<n'?,(-A(+l'et'nb'Z@'Us'_,'e2&=N'fY(#)(6.'~g'sA&j:'r5'&I)$R&jB'f/(ik(du'Qs&P.(*f&=W&rM'aX(#E'.p(+F&vn(Nt'R}'Xw(1''}`(&z(<('DK'rR(b?(oN&l((,E(hI',-'#i'cq(!c(8`((H'F1'IS']w'Ce(@6'|I'dY'KY'cg(!C'gJ'k3'-B&UM(4#'{f([;(sk&EC(pG'o@(_*'+j((!$n]'@1((`'C?&Q,(Wc&1,':l(5c'pk&s]'vi'{z&'*"
"(R.(>y$8J'Ho'Ef'gV$`@(3U(Wr(.X'YT'`]'+x'F!&a,'Jn(*='`m&os'na'pH'mo'K*(#l'~K'pq'JQ'rk'*}'ab(2!()?'eD'qy'lL(3.'`r'}x'TY&Uf'N_&I{(<j&l#(z('AN'DV'TN'hH']R&Vu'eH'hZ)_=(*j'Kl'lF&7k'oz(0B(TJ'ux&3^'s<('v(.<(BM'fS&DA()T(*P't.'`X'X3'5c'^B':)'Jq(Lz'fQ't,(*b'FL'^1()l'zf'tw(k'&u+'fC(.Z(}z(`=$s['gi&R-')J'B]&~(&EO"
"'tA(Bj'zK'IT(DT(>['ZJ'v.(@D(h$'lA'G<(oG&JV'$R(/B'sX'b9'kl'vV(5s(Fe'+G'L]',K'O{(Sh(&j'fn(BI'u3()-'PI(/]'u4'd]'uC'~b'E&'H/'OR'I/((M'f&'X;'p}'fw'~m'[N'g{'7k&Df(4W(Fl$d&(L/'ip&BJ'Lr',3$sC'cr&15(JJ(2&'yf'^l&e|'al'*$&Er(2M'uD(2F(T?(W>$z/'}/(?Y(Cc(1='L!'KR&z0((5'yr(G['cO'gK',+'j#$Yh'90&gy'iq'J_(>f'q@&?.&#I"
"'&q&H:'_t'TS'v|(TJ&nh'u+'z@'01'u8'sE'wl(!E(,^&yP&r?'Uo(!)&~W's_'hZ'|9'KO(,,({J'x9'h3')|&Ia(_m'XS'l='d4(@I'0('vq'Xi'Ev(>s'er('|(Mg(R*(Y~&/Y')!(N6'H/(E!&rv(Ka'|2&mm(UW'+r&gH(@h'&L(Bx'XW(/o'H3(FH'2o'y>'OS)sQ('R'g7$t}'?D'9q'yP&}i'aq(^}'8B(Bb'sF'mP$~S&zq'CW'aD&V-(IJ$Ti'rF(5_(Mu&rD'yp'1N$yZ'P+'4{(,l&Kh'Tn"
"'bY'3I't?&cC'k,'2~'Y@(Cz'L>(0o'`S(.w'^a'sM'lM(CU'm;'gu&qS(^y'mr(M&(9Y'bo'O&'h_&Ez(-|(Uj&Oj'f}$us'r8'IC&!N(O:'_*(k(&A4'2l(0/'UI':w'J5($H'p/'~P'oI'kK'A]'+d$ZE'Cr'{4'J1'Sz'9z'9k(03&[((d?'|J'$E'L6'|T'`C(Me'K`'`](W`(#u'Sq(,h'D{'@P'VN'R_'Fr'.)(=*'w#']D'iP'e$(7#'o0'9~'nF&iK'M@&tY()Y(pK'bQ'gM&u`'mA'm/&z~(b6"
"'{;'OD'm]'RL'gr'>:'hR(Q7'L7'V''m|'>}&s}'O#''g'>x('u'LP''t'iF&k('xm'N5't|'g8'ak']A's^'h8'vu'`,'w'']0'ri'c}'u2'cn'p>'rH'iH(UU)Lz)XR(Yg**J)I,(&O)8|&YF*>w&S<&N>&|!'+1(xN'Rf'TY)/F'vd)36)2='04(cd$Tv&/g(-N&0d&>7&E?']s(|W&-I*,d'Aq'f*)ja)L<(K?(>L$p?&,f(&p&{C&Y:$N@'r))T5&MS'6c)O()LX)*a(XP'pi)3.(KC$rf(`~$V3$b2"
"&_5&V=)44&#s'er'G.(YV)IW(_|(VC(b'$A'',l(NN&V0&[N&TH'L;)R]&C@&~0))`'dK(G8)6{$aP(f($e:'OB(Ak'&S'&2&_#'qO(_z&H+)aw()9']4(l!([2(?'(5[&>P&bZ'r*'Rm'Nr#iq'gA(`0&zB)}i(f!'jf)R$(]S&DU)Rd'kI&4/'MN&@G$vA&&~'y3)mH'$D$`3)ss)+f*3:']5'xY)]u)`y$;1'OZ$C[&1f&+D&1l)&F$gh*7R*;,(o;)Zs)fP#(b(XQ(pK$D').U$!Z$3&&1p'JL*=m$0="
"'^f'bs'i=)ZK'S,)Jh(NC&gW$.l&~#&5{&bT$a/'Py)?K&=@'o}'_i(u+'J,'X+'h$'d7'l`'o_&~G(&+(#$'vx&U&'XR(!l'vY'JM'gi'jY'<F';l(&*'Zz'{Q'5!(8y((('ko'`G'x2'{('LW(&+'YO(+{'wt'vi'KL'k1'UB']?'Y3'[f'|f'f.'b<'Wz'^`'L7'[6'|^(1}'i>'ga'^I'<W(+I'w]'rX'e''JS'vz'VU'?#'94(!4(-9(43'<W'oo'M-(b>'~k'ET'P''PR'NL'xb'Ow'xA([z(E;(ya"
"&eR(.)(nL&[H(DA&aX'Co&ar&[V(d((l<&nV($w'J](WJ'Mi'T[(&6&fq'bw('_'}6'q^&f>'f]([e(QH'[C(SD&y5(`-(&L&8')8O&gm$m~(44((.((L&$h''))Q}(vy'<m(aF'q_(xM)K4&dF'P|&iE&>n)*$&Y9'bA'Te$r](;8)6#&ki(+@(,a(U>'x}'];'rr'HQ'8J(on''F(H0'82&w]([E(i?'t^(Lr&t{(EP'gI&O;)5b&f*&'m(c''CN(oG&]Z';b)5})7u'G5'{I'X['|h(M3'-L(il&z5&l{"
"(K&'=Q'm)&/V'Sy(cb(BE&}9(l.(F|(aF(}Q&jJ)iB'y{&$l(0v(IC'R&$a3&#U)1+(Zg$r>&38(FJ)RN)#0&am'9c'Wg$Zq)@z&A4)KK'${&Na(fJ)av&>!'Rg(Lg)S4))u&EF(}~&~w$lt)2D'R/(GS$i]&e4*c_)S)${y$d|'nJ)Z*(D<'#A&:j':W'40)^('Hx)X)'0Z&F:)?~)'z'Y&(xw(k7'il'Y6'yb'on(LM'uP'Wh'9e'u:'n*'nb'X<'eb'ki'ks'bR'Gf'Au'pj'x$'d@'xA(!U'Z6(>Z'_!"
"(?F'tt'wh'lz'#_'nV'o)'wG'w)'8?'vC'x1('`']#'/~(56'Jn'a#(/r'^o'ZE'D^'S[(9Y'6#'ep'yt'c?'/`'g0'r8'VY'rx('B'v8'F^(#I(.B'Bf'z<(RI'9d'RZ'So'(?'VW(;x'By'GI(:@(4r(5J(O`'mX&<L(yQ&//)ra)RN$(M&-F'q(#LT*3U)y5(|D&5x):T)`Q&pN'[+(!,($8(G-$`F(!|'<#(/;)2K)>V'1!(O])9{'za(/}&KC(?C*4=)]m)@k!}s&FM&FN)Us(XW$gS&:f'd3)t4((5"
"$yX#f-'SO)$a'9a)7Y(`B&yB&6h(7)'vC'f'(&d(.5'H]'_,(7_'=X's$)!5(#C(cQ'_[&V>&um)BO&_1&A('[c'aQ'uJ(qE'bw&k:*I,({q)P~(XE$YB'OS&vd(1_'w0&]A'?Q'57(dY(?/&Zf&;b&{?(mC&Fq(tA'zQ&k,'&e(?Y'>}(mf(TD'A}'i7(~L'UJ&c.&,Q)bd(OZ)pd(DA&0m$^m)Q!(?&$kz)9A(5k'Zi)G.(f5$UZ&cb+q5'o<++^)=S$-<$9k)x6$@n'W7'*b'vT'MX+7>(Kr(C5)=T)xM"
"('5)X&'jW&Nc&ZW*h['3K(qI'wb'|v'/U)-?()C$#x#Ui)Gk)gZ)=w&g3$|~&*B'm3(#v&kC(T!(2P(0@(gu'^H'8}'f4'>B'ub'W:(lp'o^'J:'X*'pC'c>';4)L;(|s'X*))S(!2'>h'dA(Oi'kK'XD'[('s3'wl'8+'hh'm&'{G'q='Ob(#.((w'Pz'zv'Qb(#4'>@'be'ms'j+'2j(5^'}Z'Jy'(w'hf'l|(7r(=D'9e')o(=<'e_',o(06'zk(I~')e'b5'@A'JR(<0'rZ'GF'Qd'*)(0:(FA(!k(T:"
"'@0't!'GO'I9'[6(.S'GJ'z''G+'RC'wY)+y(Sp(HW&mz'^b&fn(3#&pP&zg'*a(9'(;H'3!'jO(gN'U3&M?(/v'0+'Aa'mD)op(Kp'^S&xt).w&y.',z'oF'Z-'<A'Iw(+X(OW(l*&:)'.x&r9(hN&Tl&|V&5](bF)!9&@~'J+)W='oE)Mw)2~(,b&9+'>P(Q')!n'K:&}4(!L(R1)1a$MC'S;)7N(+~(5A(YX*0`$p1'N^$`*(_r&]<',b&<?(Mb)5f$]q(=I*MI'yx(h9($#'s+'wf'~r&N>'sB'@v'S9"
"+<!(E8(VR'^q&ta'g<'nE(RR(Qi']3')U(*4(/S(0|':;'7f'<a'bW(61's='Z3(0(&a_'+x(|t($(&16'v9&f*(xh&Pn&F3($n'V](?b'_m'Gi(qi$z[$HO)R|(#g$g7(.3(iG)0x$z#&Rv'20'Xj(T$'Pv)4()@H$m*'V.)'O)+7&<]'bC(&S(_*&uG&X8&}F''f'wf&Y'&?r)JJ'H?):x)0v(la$s)'jZ)$s(zu&Q'&^*(Kt&Wy(j~&&g'xL)T9'QT'[)'m='t>'ei')x'hy(-I'^+'q&'k*&J,&q('cp"
"(O|'i,(-X'Zq(&w'7~'dQ&FW&YM(7[(!V'{Y'Y2'7}'V?'>:'HN']G(6['gZ(#c'+A'eo'&1'CI(/((5V(#B'^o(!y(:n'B6'7=']W'rw'o.'ze(W^'Sf(?p'bn'i3('l'zx($t'U['b((3?'?{'_X'f!'LL(Ab(sl)0E#|W'7O(0^'Bs':L'X('A@(1e')x(Xv&=9'Bp'e]";

    int i, j, phase_idx, pattern_idx;
    double dense0[n_dense0][20];
    double bias0[n_dense0];
    double dense1[n_dense1][n_dense0];
    double bias1[n_dense1];
    double dense2[n_dense1];
    double bias2;
    const int pattern_sizes[n_patterns] = {8, 8, 8, 5, 6, 7, 8, 10, 10, 10, 10};
    int ln_char = (int)(chars.size());
    int data_idx = 0;
    for (phase_idx = 0; phase_idx < n_phases; ++phase_idx){
        for (pattern_idx = 0; pattern_idx < n_patterns; ++pattern_idx){
            for (i = 0; i < pattern_sizes[pattern_idx] * 2; ++i){
                for (j = 0; j < n_dense0; ++j){
                    dense0[j][i] = get_elem(&param_compressed1, &data_idx, &chars, ln_char);
                }
            }
            for (i = 0; i < n_dense0; ++i){
                bias0[i] = get_elem(&param_compressed1, &data_idx, &chars, ln_char);
            }
            for (i = 0; i < n_dense0; ++i){
                for (j = 0; j < n_dense1; ++j){
                    dense1[j][i] = get_elem(&param_compressed1, &data_idx, &chars, ln_char);
                }
            }
            for (i = 0; i < n_dense1; ++i){
                bias1[i] = get_elem(&param_compressed1, &data_idx, &chars, ln_char);
            }
            for (i = 0; i < n_dense1; ++i){
                dense2[i] = get_elem(&param_compressed1, &data_idx, &chars, ln_char);
            }
            bias2 = get_elem(&param_compressed1, &data_idx, &chars, ln_char);
            pre_evaluation(phase_idx, pattern_idx, pattern_sizes[pattern_idx], dense0, bias0, dense1, bias1, dense2, bias2);
        }
    }
    for (phase_idx = 0; phase_idx < n_phases; ++phase_idx){
        for (i = 0; i < n_all_input; ++i){
            for (j = 0; j < n_all_dense0; ++j){
                all_dense0[phase_idx][j][i] = get_elem(&param_compressed1, &data_idx, &chars, ln_char);
            }
        }
        for (i = 0; i < n_all_dense0; ++i){
            all_bias0[phase_idx][i] = get_elem(&param_compressed1, &data_idx, &chars, ln_char);
        }
        for (i = 0; i < n_all_dense0; ++i){
            all_dense1[phase_idx][i] = get_elem(&param_compressed1, &data_idx, &chars, ln_char);
        }
        all_bias1[phase_idx] = get_elem(&param_compressed1, &data_idx, &chars, ln_char);
    }
}

inline void search_hash_table_init(const int table_idx){
    for(int i = 0; i < search_hash_table_size; ++i)
        search_replace_table[table_idx][i] = NULL;
}

inline search_node* search_node_init(const int *key, double l, double u){
    search_node* p_node = NULL;
    p_node = (search_node*)malloc(sizeof(search_node));
    for (int i = 0; i < hw; ++i)
        p_node->k[i] = key[i];
    p_node->v.first = l;
    p_node->v.second = u;
    p_node->p_n_node = NULL;
    return p_node;
}

inline void register_search(const int table_idx, const int *key, int hash, double l, double u){
    if(search_replace_table[table_idx][hash] == NULL){
        search_replace_table[table_idx][hash] = search_node_init(key, l, u);
    } else {
        search_node *p_node = search_replace_table[table_idx][hash];
        search_node *p_pre_node = NULL;
        p_pre_node = p_node;
        while(p_node != NULL){
            if(compare_key(key, p_node->k)){
                p_node->v.first = l;
                p_node->v.second = u;
                return;
            }
            p_pre_node = p_node;
            p_node = p_node->p_n_node;
        }
        p_pre_node->p_n_node = search_node_init(key, l, u);
    }
}

inline pair<double, double> get_search(const int *key, const int hash, const int table_idx){
    search_node *p_node = search_replace_table[table_idx][hash];
    while(p_node != NULL){
        if(compare_key(key, p_node->k)){
            return p_node->v;
        }
        p_node = p_node->p_n_node;
    }
    pair<double, double> empty = {-inf, -inf};
    return empty;
}

inline double end_game(const board *b){
    int count = 0;
    for (int idx = 0; idx < hw; ++idx)
        count += count_arr[b->b[idx]];
    if (b->p == 1)
        count = -count;
    if (count > 0)
        return 1.0;
    else if (count < 0)
        return -1.0;
    return 0.0;
}

inline int calc_canput(const board *b){
    int res = 0;
    for (int i = 0; i < b_idx_num; ++i)
        res += canput_arr[b->p][b->b[i]];
    if (b->p)
        res = -res;
    return res;
}

inline int calc_surround0(const board *b){
    int res = 0;
    for (int i = 0; i < b_idx_num; ++i)
        res += surround_arr[0][b->b[i]];
    return res;
}

inline int calc_surround1(const board *b){
    int res = 0;
    for (int i = 0; i < b_idx_num; ++i)
        res += surround_arr[1][b->b[i]];
    return res;
}

inline int calc_phase_idx(const board *b){
    int turn = -4;
    for (int idx = 0; idx < hw; ++idx)
        turn += count_all_arr[b->b[idx]];
    if (turn < 30)
        return 0;
    else if (turn < 40)
        return 1;
    else if (turn < 50)
        return 2;
    return 3;
}

inline void calc_pattern(const board *b, double arr[]){
    int idx, phase_idx;
    phase_idx = calc_phase_idx(b);
    double line2 = 0.0, line3 = 0.0, line4 = 0.0, diagonal5 = 0.0, diagonal6 = 0.0, diagonal7 = 0.0, diagonal8 = 0.0, edge_2x = 0.0, triangle = 0.0, edge_block = 0.0, cross = 0.0; //corner25 = 0.0;

    line2 += ev_arr[phase_idx][0][b->b[1]];
    line2 += ev_arr[phase_idx][0][b->b[6]];
    line2 += ev_arr[phase_idx][0][b->b[9]];
    line2 += ev_arr[phase_idx][0][b->b[14]];
    line2 += ev_arr[phase_idx][0][reverse_board[b->b[1]]];
    line2 += ev_arr[phase_idx][0][reverse_board[b->b[6]]];
    line2 += ev_arr[phase_idx][0][reverse_board[b->b[9]]];
    line2 += ev_arr[phase_idx][0][reverse_board[b->b[14]]];

    line3 += ev_arr[phase_idx][1][b->b[2]];
    line3 += ev_arr[phase_idx][1][b->b[5]];
    line3 += ev_arr[phase_idx][1][b->b[10]];
    line3 += ev_arr[phase_idx][1][b->b[13]];
    line3 += ev_arr[phase_idx][1][reverse_board[b->b[2]]];
    line3 += ev_arr[phase_idx][1][reverse_board[b->b[5]]];
    line3 += ev_arr[phase_idx][1][reverse_board[b->b[10]]];
    line3 += ev_arr[phase_idx][1][reverse_board[b->b[13]]];

    line4 += ev_arr[phase_idx][2][b->b[3]];
    line4 += ev_arr[phase_idx][2][b->b[4]];
    line4 += ev_arr[phase_idx][2][b->b[11]];
    line4 += ev_arr[phase_idx][2][b->b[12]];
    line4 += ev_arr[phase_idx][2][reverse_board[b->b[3]]];
    line4 += ev_arr[phase_idx][2][reverse_board[b->b[4]]];
    line4 += ev_arr[phase_idx][2][reverse_board[b->b[11]]];
    line4 += ev_arr[phase_idx][2][reverse_board[b->b[12]]];

    diagonal5 += ev_arr[phase_idx][3][b->b[18] / pow3[3]];
    diagonal5 += ev_arr[phase_idx][3][b->b[24] / pow3[3]];
    diagonal5 += ev_arr[phase_idx][3][b->b[29] / pow3[3]];
    diagonal5 += ev_arr[phase_idx][3][b->b[35] / pow3[3]];
    diagonal5 += ev_arr[phase_idx][3][mod3[reverse_board[b->b[18]]][5]];
    diagonal5 += ev_arr[phase_idx][3][mod3[reverse_board[b->b[24]]][5]];
    diagonal5 += ev_arr[phase_idx][3][mod3[reverse_board[b->b[29]]][5]];
    diagonal5 += ev_arr[phase_idx][3][mod3[reverse_board[b->b[35]]][5]];

    diagonal6 += ev_arr[phase_idx][4][b->b[19] / pow3[2]];
    diagonal6 += ev_arr[phase_idx][4][b->b[23] / pow3[2]];
    diagonal6 += ev_arr[phase_idx][4][b->b[30] / pow3[2]];
    diagonal6 += ev_arr[phase_idx][4][b->b[34] / pow3[2]];
    diagonal6 += ev_arr[phase_idx][4][mod3[reverse_board[b->b[19]]][6]];
    diagonal6 += ev_arr[phase_idx][4][mod3[reverse_board[b->b[23]]][6]];
    diagonal6 += ev_arr[phase_idx][4][mod3[reverse_board[b->b[30]]][6]];
    diagonal6 += ev_arr[phase_idx][4][mod3[reverse_board[b->b[34]]][6]];

    diagonal7 += ev_arr[phase_idx][5][b->b[20] / pow3[1]];
    diagonal7 += ev_arr[phase_idx][5][b->b[22] / pow3[1]];
    diagonal7 += ev_arr[phase_idx][5][b->b[31] / pow3[1]];
    diagonal7 += ev_arr[phase_idx][5][b->b[33] / pow3[1]];
    diagonal7 += ev_arr[phase_idx][5][mod3[reverse_board[b->b[20]]][7]];
    diagonal7 += ev_arr[phase_idx][5][mod3[reverse_board[b->b[22]]][7]];
    diagonal7 += ev_arr[phase_idx][5][mod3[reverse_board[b->b[31]]][7]];
    diagonal7 += ev_arr[phase_idx][5][mod3[reverse_board[b->b[33]]][7]];

    diagonal8 += ev_arr[phase_idx][6][b->b[21]];
    diagonal8 += ev_arr[phase_idx][6][b->b[32]];
    diagonal8 += ev_arr[phase_idx][6][reverse_board[b->b[21]]];
    diagonal8 += ev_arr[phase_idx][6][reverse_board[b->b[32]]];

    idx = pop_digit[b->b[1]][6] * pow3[9] + b->b[0] * pow3[1] + pop_digit[b->b[1]][1];
    edge_2x += ev_arr[phase_idx][7][idx];
    idx = pop_digit[b->b[1]][1] * pow3[9] + reverse_board[b->b[0]] * pow3[1] + pop_digit[b->b[1]][6];
    edge_2x += ev_arr[phase_idx][7][idx];
    idx = pop_digit[b->b[6]][6] * pow3[9] + b->b[7] * pow3[1] + pop_digit[b->b[6]][1];
    edge_2x += ev_arr[phase_idx][7][idx];
    idx = pop_digit[b->b[6]][1] * pow3[9] + reverse_board[b->b[7]] * pow3[1] + pop_digit[b->b[6]][6];
    edge_2x += ev_arr[phase_idx][7][idx];
    idx = pop_digit[b->b[9]][1] * pow3[9] + b->b[8] * pow3[1] + pop_digit[b->b[9]][6];
    edge_2x += ev_arr[phase_idx][7][idx];
    idx = pop_digit[b->b[9]][6] * pow3[9] + reverse_board[b->b[8]] * pow3[1] + pop_digit[b->b[9]][1];
    edge_2x += ev_arr[phase_idx][7][idx];
    idx = pop_digit[b->b[14]][1] * pow3[9] + b->b[15] * pow3[1] + pop_digit[b->b[14]][6];
    edge_2x += ev_arr[phase_idx][7][idx];
    idx = pop_digit[b->b[14]][6] * pow3[9] + reverse_board[b->b[15]] * pow3[1] + pop_digit[b->b[14]][1];
    edge_2x += ev_arr[phase_idx][7][idx];

    idx = b->b[0] / pow3[4] * pow3[6] + b->b[1] / pow3[5] * pow3[3] + b->b[2] / pow3[6] * pow3[1] + b->b[3] / pow3[7];
    triangle += ev_arr[phase_idx][8][idx];
    idx = reverse_board[b->b[0]] / pow3[4] * pow3[6] + reverse_board[b->b[1]] / pow3[5] * pow3[3] + reverse_board[b->b[2]] / pow3[6] * pow3[1] + reverse_board[b->b[3]] / pow3[7];
    triangle += ev_arr[phase_idx][8][idx];
    idx = b->b[7] / pow3[4] * pow3[6] + b->b[6] / pow3[5] * pow3[3] + b->b[5] / pow3[6] * pow3[1] + b->b[4] / pow3[7];
    triangle += ev_arr[phase_idx][8][idx];
    idx = reverse_board[b->b[7]] / pow3[4] * pow3[6] + reverse_board[b->b[6]] / pow3[5] * pow3[3] + reverse_board[b->b[5]] / pow3[6] * pow3[1] + reverse_board[b->b[4]] / pow3[7];
    triangle += ev_arr[phase_idx][8][idx];
    idx = b->b[8] / pow3[4] * pow3[6] + b->b[9] / pow3[5] * pow3[3] + b->b[10] / pow3[6] * pow3[1] + b->b[11] / pow3[7];
    triangle += ev_arr[phase_idx][8][idx];
    idx = reverse_board[b->b[8]] / pow3[4] * pow3[6] + reverse_board[b->b[9]] / pow3[5] * pow3[3] + reverse_board[b->b[10]] / pow3[6] * pow3[1] + reverse_board[b->b[11]] / pow3[7];
    triangle += ev_arr[phase_idx][8][idx];
    idx = b->b[15] / pow3[4] * pow3[6] + b->b[14] / pow3[5] * pow3[3] + b->b[13] / pow3[6] * pow3[1] + b->b[12] / pow3[7];
    triangle += ev_arr[phase_idx][8][idx];
    idx = reverse_board[b->b[15]] / pow3[4] * pow3[6] + reverse_board[b->b[14]] / pow3[5] * pow3[3] + reverse_board[b->b[13]] / pow3[6] * pow3[1] + reverse_board[b->b[12]] / pow3[7];
    triangle += ev_arr[phase_idx][8][idx];

    idx = pop_digit[b->b[0]][0] * pow3[9] + pop_mid[b->b[0]][6][2] * pow3[5] + pop_digit[b->b[0]][7] * pow3[4] + pop_mid[b->b[1]][6][2];
    edge_block += ev_arr[phase_idx][9][idx];
    idx = pop_digit[b->b[0]][7] * pow3[9] + pop_mid[reverse_board[b->b[0]]][6][2] * pow3[5] + pop_digit[b->b[0]][0] * pow3[4] + pop_mid[reverse_board[b->b[1]]][6][2];
    edge_block += ev_arr[phase_idx][9][idx];
    idx = pop_digit[b->b[7]][0] * pow3[9] + pop_mid[b->b[7]][6][2] * pow3[5] + pop_digit[b->b[7]][7] * pow3[4] + pop_mid[b->b[6]][6][2];
    edge_block += ev_arr[phase_idx][9][idx];
    idx = pop_digit[b->b[7]][7] * pow3[9] + pop_mid[reverse_board[b->b[7]]][6][2] * pow3[5] + pop_digit[b->b[7]][0] * pow3[4] + pop_mid[reverse_board[b->b[6]]][6][2];
    edge_block += ev_arr[phase_idx][9][idx];
    idx = pop_digit[b->b[8]][0] * pow3[9] + pop_mid[b->b[8]][6][2] * pow3[5] + pop_digit[b->b[8]][7] * pow3[4] + pop_mid[b->b[9]][6][2];
    edge_block += ev_arr[phase_idx][9][idx];
    idx = pop_digit[b->b[8]][7] * pow3[9] + pop_mid[reverse_board[b->b[8]]][6][2] * pow3[5] + pop_digit[b->b[8]][0] * pow3[4] + pop_mid[reverse_board[b->b[9]]][6][2];
    edge_block += ev_arr[phase_idx][9][idx];
    idx = pop_digit[b->b[15]][0] * pow3[9] + pop_mid[b->b[15]][6][2] * pow3[5] + pop_digit[b->b[15]][7] * pow3[4] + pop_mid[b->b[14]][6][2];
    edge_block += ev_arr[phase_idx][9][idx];
    idx = pop_digit[b->b[15]][7] * pow3[9] + pop_mid[reverse_board[b->b[15]]][6][2] * pow3[5] + pop_digit[b->b[15]][0] * pow3[4] + pop_mid[reverse_board[b->b[14]]][6][2];
    edge_block += ev_arr[phase_idx][9][idx];

    idx = b->b[21] / pow3[4] * pow3[6] + b->b[20] / pow3[5] * pow3[3] + b->b[22] / pow3[5];
    cross += ev_arr[phase_idx][10][idx];
    idx = b->b[21] / pow3[4] * pow3[6] + b->b[22] / pow3[5] * pow3[3] + b->b[20] / pow3[5];
    cross += ev_arr[phase_idx][10][idx];
    idx = b->b[32] / pow3[4] * pow3[6] + b->b[31] / pow3[5] * pow3[3] + b->b[33] / pow3[5];
    cross += ev_arr[phase_idx][10][idx];
    idx = b->b[32] / pow3[4] * pow3[6] + b->b[33] / pow3[5] * pow3[3] + b->b[31] / pow3[5];
    cross += ev_arr[phase_idx][10][idx];
    idx = reverse_board[b->b[21]] / pow3[4] * pow3[6] + pop_mid[reverse_board[b->b[20]]][7][4] * pow3[3] + pop_mid[reverse_board[b->b[22]]][7][4];
    cross += ev_arr[phase_idx][10][idx];
    idx = reverse_board[b->b[21]] / pow3[4] * pow3[6] + pop_mid[reverse_board[b->b[22]]][7][4] * pow3[3] + pop_mid[reverse_board[b->b[20]]][7][4];
    cross += ev_arr[phase_idx][10][idx];
    idx = reverse_board[b->b[32]] / pow3[4] * pow3[6] + pop_mid[reverse_board[b->b[31]]][7][4] * pow3[3] + pop_mid[reverse_board[b->b[33]]][7][4];
    cross += ev_arr[phase_idx][10][idx];
    idx = reverse_board[b->b[32]] / pow3[4] * pow3[6] + pop_mid[reverse_board[b->b[33]]][7][4] * pow3[3] + pop_mid[reverse_board[b->b[31]]][7][4];
    cross += ev_arr[phase_idx][10][idx];

    arr[0] = line2 / 8.0;
    arr[1] = line3 / 8.0;
    arr[2] = line4 / 8.0;
    arr[3] = diagonal5 / 8.0;
    arr[4] = diagonal6 / 8.0;
    arr[5] = diagonal7 / 8.0;
    arr[6] = diagonal8 / 4.0;
    arr[7] = edge_2x / 8.0;
    arr[8] = triangle / 8.0;
    arr[9] = edge_block / 8.0;
    arr[10] = cross / 8.0;
}

inline double evaluate(const board *b){
    int phase_idx = calc_phase_idx(b);
    double in_arr[n_all_input];
    calc_pattern(b, in_arr);
    in_arr[11] = (double)calc_canput(b) / 30.0;
    in_arr[12] = (double)calc_surround0(b) / 30.0;
    in_arr[13] = (double)calc_surround1(b) / 30.0;
    double hidden[n_all_dense0];
    int i, j;
    for (i = 0; i < n_all_dense0; ++i){
        hidden[i] = all_bias0[phase_idx][i];
        for (j = 0; j < n_all_input; ++j)
            hidden[i] += in_arr[j] * all_dense0[phase_idx][i][j];
        hidden[i] = leaky_relu(hidden[i]);
    }
    double res = all_bias1[phase_idx];
    for (i = 0; i < n_all_dense0; ++i)
        res += hidden[i] * all_dense1[phase_idx][i];
    if (b->p)
        res = -res;
    return min(0.9999, max(-0.9999, res));
}

double canput_exact_evaluate(board *b){
    int res = 0;
    for (const int &cell: vacant_lst){
        if (pop_digit[b->b[cell / hw]][cell % hw] == 2){
            for (const int &idx: place_included[cell]){
                if (move_arr[b->p][b->b[idx]][local_place[idx][cell]][0] || move_arr[b->p][b->b[idx]][local_place[idx][cell]][1]){
                    ++res;
                    break;
                }
            }
        }
    }
    return (double)res;
}

bool move_ordering(const board p, const board q){
    return p.v > q.v;
}

double nega_alpha_ordering(const board *b, int skip_cnt, int depth, double alpha, double beta);

inline bool mpc_lower(const board *b, int skip_cnt, int depth, double alpha){
    return false;
    double vd = nega_alpha_ordering(b, skip_cnt, depth / 4, (alpha - mpct * mpcsd) / mpca - mpcwindow, (alpha - mpct * mpcsd) / mpca);
    if (vd < (alpha - mpct * mpcsd) / mpca)
        return true;
    return false;
}

inline bool mpc_higher(const board *b, int skip_cnt, int depth, double beta){
    return false;
    double vd = nega_alpha_ordering(b, skip_cnt, depth / 4, (beta + mpct * mpcsd) / mpca, (beta + mpct * mpcsd) / mpca + mpcwindow);
    if (vd > (beta + mpct * mpcsd) / mpca)
        return true;
    return false;
}

double final_move(const board *b, bool skipped){
    ++searched_nodes;
    int before_score = 0;
    for (int i = 0; i < hw; ++i)
        before_score += count_arr[b->b[i]];
    if (b->p == 1)
        before_score = -before_score;
    int score;
    for (const int &cell: vacant_lst){
        if (pop_digit[b->b[cell / hw]][cell % hw] == 2){
            score = before_score + 1;
            for (const int &idx: place_included[cell])
                score += (move_arr[b->p][b->b[idx]][local_place[idx][cell]][0] + move_arr[b->p][b->b[idx]][local_place[idx][cell]][1]) * 2;
            break;
        }
    }
    if (score == before_score + 1){
        if (skipped)
            return before_score;
        board rb;
        for (int i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -final_move(&rb, true);
    }
    //return score;
    if (score > 0)
        return 1.0;
    else if (score == 0)
        return 0.0;
    return -1.0;
}

double nega_alpha_final(const board *b, int skip_cnt, int depth, double alpha, double beta){
    ++searched_nodes;
    if (b->n == hw2_m1)
        return final_move(b, false);
    if (skip_cnt == 2)
        return end_game(b);
    board nb;
    bool passed = true;
    for (const int &cell: vacant_lst){
        if (pop_digit[b->b[cell / hw]][cell % hw] == 2){
            for (const int &idx: place_included[cell]){
                if (move_arr[b->p][b->b[idx]][local_place[idx][cell]][0] || move_arr[b->p][b->b[idx]][local_place[idx][cell]][1]){
                    passed = false;
                    nb = move(b, cell);
                    alpha = max(alpha, -nega_alpha_final(&nb, 0, depth - 1, -beta, -alpha));
                    if (beta <= alpha)
                        return alpha;
                    break;
                }
            }
        }
    }
    if (passed){
        board rb;
        for (int i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -nega_alpha_final(&rb, skip_cnt + 1, depth, -beta, -alpha);
    }
    return alpha;
}

double nega_alpha_ordering_final(const board *b, int skip_cnt, int depth, double alpha, double beta){
    if (skip_cnt == 2)
        return end_game(b);
    if (depth <= 7)
        return nega_alpha_final(b, skip_cnt, depth, alpha, beta);
    /*
    if (mpc_higher(b, skip_cnt, depth, beta))
        return beta + window;
    if (mpc_lower(b, skip_cnt, depth, alpha))
        return alpha - window;
    */
    ++searched_nodes;
    vector<board> nb;
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (pop_digit[b->b[cell / hw]][cell % hw] == 2){
            for (const int &idx: place_included[cell]){
                if (move_arr[b->p][b->b[idx]][local_place[idx][cell]][0] || move_arr[b->p][b->b[idx]][local_place[idx][cell]][1]){
                    nb.push_back(move(b, cell));
                    nb[canput].v = -canput_exact_evaluate(&nb[canput]);
                    ++canput;
                    break;
                }
            }
        }
    }
    if (canput == 0){
        board rb;
        for (int i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -nega_alpha_ordering_final(&rb, skip_cnt + 1, depth, -beta, -alpha);
    }
    if (canput > 2)
        sort(nb.begin(), nb.end(), move_ordering);
    for (int i = 0; i < canput; ++i){
        alpha = max(alpha, -nega_alpha_ordering_final(&nb[i], 0, depth - 1, -beta, -alpha));
        if (beta <= alpha)
            return alpha;
    }
    return alpha;
}

double nega_alpha(const board *b, int skip_cnt, int depth, double alpha, double beta){
    ++searched_nodes;
    if (b->n == hw2_m1)
        return final_move(b, false);
    if (skip_cnt == 2 || b->n == hw2)
        return end_game(b);
    if (depth == 0 && b->n < hw2)
        return evaluate(b);
    double g;
    board nb;
    bool passed = true;
    for (const int &cell: vacant_lst){
        if (pop_digit[b->b[cell / hw]][cell % hw] == 2){
            for (const int &idx: place_included[cell]){
                if (move_arr[b->p][b->b[idx]][local_place[idx][cell]][0] || move_arr[b->p][b->b[idx]][local_place[idx][cell]][1]){
                    passed = false;
                    nb = move(b, cell);
                    g = -nega_alpha(&nb, 0, depth - 1, -beta, -alpha);
                    if (g == inf)
                        return -inf;
                    alpha = max(alpha, g);
                    if (beta <= alpha)
                        return alpha;
                    break;
                }
            }
        }
    }
    if (passed){
        board rb;
        for (int i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -nega_alpha(&rb, skip_cnt + 1, depth, -beta, -alpha);
    }
    return alpha;
}

double nega_alpha_ordering(const board *b, int skip_cnt, int depth, double alpha, double beta){
    ++searched_nodes;
    if (skip_cnt == 2 || b->n == hw2)
        return end_game(b);
    if (depth == 0 && b->n < hw2)
        return evaluate(b);
    if (depth <= 3)
        return nega_alpha(b, skip_cnt, depth, alpha, beta);
    if (mpc_higher(b, skip_cnt, depth, beta))
        return beta + window;
    if (mpc_lower(b, skip_cnt, depth, alpha))
        return alpha - window;
    int hash = (int)(calc_hash(b->b) & search_hash_mask);
    pair<double, double> lu = get_search(b->b, hash, 1 - f_search_table_idx);
    if (lu.first != -inf){
        if (lu.first == lu.second)
            return lu.first;
        alpha = max(alpha, lu.first);
        if (alpha >= beta)
            return alpha;
    }
    if (lu.second != -inf){
        beta = min(beta, lu.second);
        if (alpha >= beta)
            return beta;
    }
    vector<board> nb;
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (pop_digit[b->b[cell / hw]][cell % hw] == 2){
            for (const int &idx: place_included[cell]){
                if (move_arr[b->p][b->b[idx]][local_place[idx][cell]][0] || move_arr[b->p][b->b[idx]][local_place[idx][cell]][1]){
                    nb.push_back(move(b, cell));
                    nb[canput].v = get_search(nb[canput].b, calc_hash(nb[canput].b) & search_hash_mask, f_search_table_idx).second;
                    if (nb[canput].v == -inf)
                        nb[canput].v = -evaluate(&nb[canput]) - 1000.0;
                    else if (b->p == 1)
                        nb[canput].v = -nb[canput].v;
                    nb[canput].v -= 0.02 * (double)nb[canput].op;
                    ++canput;
                    break;
                }
            }
        }
    }
    if (canput == 0){
        board rb;
        for (int i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -nega_alpha_ordering(&rb, skip_cnt + 1, depth, -beta, -alpha);
    }
    if (canput > 2)
        sort(nb.begin(), nb.end(), move_ordering);
    double v = -inf, g;
    for (int i = 0; i < canput; ++i){
        g = -nega_alpha_ordering(&nb[i], 0, depth - 1, -beta, -alpha);
        if (g == inf)
            return -inf;
        if (beta <= g){
            if (lu.first < g)
                register_search(1 - f_search_table_idx, b->b, hash, g, lu.second);
            return g;
        }
        alpha = max(alpha, g);
        v = max(v, g);
    }
    if (v <= alpha)
        register_search(1 - f_search_table_idx, b->b, hash, lu.first, v);
    else
        register_search(1 - f_search_table_idx, b->b, hash, v, v);
    return v;
}

double nega_scout(const board *b, int skip_cnt, int depth, double alpha, double beta){
    ++searched_nodes;
    if (skip_cnt == 2 || b->n == hw2)
        return end_game(b);
    if (depth == 0 && b->n < hw2)
        return evaluate(b);
    if (depth <= 3)
        return nega_alpha(b, skip_cnt, depth, alpha, beta);
    if (mpc_higher(b, skip_cnt, depth, beta))
        return beta + window;
    if (mpc_lower(b, skip_cnt, depth, alpha))
        return alpha - window;
    int hash = (int)(calc_hash(b->b) & search_hash_mask);
    pair<double, double> lu = get_search(b->b, hash, 1 - f_search_table_idx);
    if (lu.first != -inf){
        if (lu.first == lu.second)
            return lu.first;
        alpha = max(alpha, lu.first);
        if (alpha >= beta)
            return alpha;
    }
    if (lu.second != -inf){
        beta = min(beta, lu.second);
        if (alpha >= beta)
            return beta;
    }
    vector<board> nb;
    int canput = 0;
    for (const int &cell: vacant_lst){
        if (pop_digit[b->b[cell / hw]][cell % hw] == 2){
            for (const int &idx: place_included[cell]){
                if (move_arr[b->p][b->b[idx]][local_place[idx][cell]][0] || move_arr[b->p][b->b[idx]][local_place[idx][cell]][1]){
                    nb.push_back(move(b, cell));
                    nb[canput].v = get_search(nb[canput].b, calc_hash(nb[canput].b) & search_hash_mask, f_search_table_idx).second;
                    if (nb[canput].v == -inf)
                        nb[canput].v = -evaluate(&nb[canput]) - 1000.0;
                    else if (b->p == 1)
                        nb[canput].v = -nb[canput].v;
                    nb[canput].v -= 0.02 * (double)nb[canput].op;
                    ++canput;
                    break;
                }
            }
        }
    }
    if (canput == 0){
        board rb;
        for (int i = 0; i < b_idx_num; ++i)
            rb.b[i] = b->b[i];
        rb.p = 1 - b->p;
        rb.n = b->n;
        return -nega_scout(&rb, skip_cnt + 1, depth, -beta, -alpha);
    }
    if (canput > 2)
        sort(nb.begin(), nb.end(), move_ordering);
    double v, g;
    g = -nega_scout(&nb[0], 0, depth - 1, -beta, -alpha);
    if (g == inf)
        return -inf;
    if (beta <= g){
        if (lu.first < g)
            register_search(1 - f_search_table_idx, b->b, hash, g, lu.second);
        return g;
    }
    v = g;
    alpha = max(alpha, g);
    for (int i = 1; i < canput; ++i){
        g = -nega_alpha_ordering(&nb[i], 0, depth - 1, -alpha - window, -alpha);
        if (g == inf)
            return -inf;
        if (beta <= g){
            if (lu.first < g)
                register_search(1 - f_search_table_idx, b->b, hash, g, lu.second);
            return g;
        }
        if (alpha < g){
            alpha = g;
            g = -nega_scout(&nb[i], 0, depth - 1, -beta, -alpha);
            if (g == inf)
                return -inf;
            if (beta <= g){
                if (lu.first < g)
                    register_search(1 - f_search_table_idx, b->b, hash, g, lu.second);
                return g;
            }
            alpha = max(alpha, g);
        }
        v = max(v, g);
    }
    if (v <= alpha)
        register_search(1 - f_search_table_idx, b->b, hash, lu.first, v);
    else
        register_search(1 - f_search_table_idx, b->b, hash, v, v);
    return v;
}

inline search_result search(const board b, int r_depth, int w_r_depth){
    vector<board> nb;
    for (const int &cell: vacant_lst){
        for (const int &idx: place_included[cell]){
            if (move_arr[b.p][b.b[idx]][local_place[idx][cell]][0] || move_arr[b.p][b.b[idx]][local_place[idx][cell]][1]){
                cout << cell << " ";
                nb.push_back(move(&b, cell));
                break;
            }
        }
    }
    cout << endl;
    int canput = nb.size();
    cout << "canput: " << canput << endl;
    int depth;
    int res_depth;
    int policy = -1;
    int tmp_policy, i;
    double alpha, beta, g, value;
    searched_nodes = 0;
    bool break_flag = false;
    bool normal_flag = false;
    if (b.n >= hw2 - w_r_depth){
        cout << "win searching" << endl;
        depth = hw2_m1 - b.n;
        alpha = -1.5;
        beta = 1.5;
        for (i = 0; i < canput; ++i)
            nb[i].v = -canput_exact_evaluate(&nb[i]) - 3.0 * evaluate(&nb[i]);
        if (canput > 1)
            sort(nb.begin(), nb.end(), move_ordering);
        for (i = 0; i < canput; ++i){
            g = -nega_alpha_ordering_final(&nb[i], 0, depth, -beta, -alpha);
            if (alpha < g || i == 0){
                alpha = g;
                tmp_policy = nb[i].policy;
            }
            if (alpha >= 1.0)
                break;
        }
        f_search_table_idx = 1 - f_search_table_idx;
        policy = tmp_policy;
        value = max(-1.0, min(1.0, alpha));
        res_depth = depth + 1;
        cout << "depth: " << res_depth << " policy: " << policy << " value: " << value<< " nodes: " << searched_nodes << endl;
        if (value == -1.0){
            r_depth = 5;
            normal_flag = true;
        }
    }
    if (b.n < hw2 - w_r_depth || normal_flag){
        cout << "normal searching" << endl;
        for (depth = 1; depth <= r_depth; ++depth){
            alpha = -1.5;
            beta = 1.5;
            search_hash_table_init(1 - f_search_table_idx);
            for (i = 0; i < canput; ++i){
                nb[i].v = get_search(nb[i].b, calc_hash(nb[i].b) & search_hash_mask, f_search_table_idx).second;
                if (nb[i].v == -inf)
                    nb[i].v = -evaluate(&nb[i]) - 1000.0;
                else if (b.p == 1)
                    nb[i].v = -nb[i].v;
                nb[i].v -= 0.02 * (double)nb[i].op;
            }
            if (canput > 1)
                sort(nb.begin(), nb.end(), move_ordering);
            g = -nega_scout(&nb[0], 0, depth, -beta, -alpha);
            alpha = max(alpha, g);
            policy = nb[0].policy;
            for (i = 1; i < canput; ++i){
                g = -nega_alpha_ordering(&nb[i], 0, depth, -alpha - window, -alpha);
                if (g == inf){
                    break_flag = true;
                    break;
                }
                if (alpha < g){
                    g = -nega_scout(&nb[i], 0, depth, -beta, -g);
                    if (g == inf){
                        break_flag = true;
                        break;
                    }
                    if (alpha < g){
                        alpha = g;
                        tmp_policy = nb[i].policy;
                    }
                }
            }
            f_search_table_idx = 1 - f_search_table_idx;
            value = alpha;
            res_depth = depth + 1;
            cout << "depth: " << res_depth << " time: " << " policy: " << policy << " value: " << value<< " nodes: " << searched_nodes << endl;
        }
    }
    if (normal_flag)
        value = -1.0;
    cout << "policy: " << policy << " value: " << value << endl;
    search_result res;
    res.policy = policy;
    res.value = value;
    res.depth = res_depth;
    return res;
}

double calc_result_value(double v){
    return 50.0 + v * 50.0;
}

inline double output_coord(int policy, double raw_val){
    return 1000.0 * (double)policy + calc_result_value(raw_val);
}

int cmp_vacant(int p, int q){
    return cell_weight[p] > cell_weight[q];
}

inline int input_board(int (&board)[b_idx_num], const int *arr){
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
    for (i = 0; i < b_idx_num; ++i){
        board[i] = n_line - 1;
        for (j = 0; j < idx_n_cell[i]; ++j){
            if (1 & (b >> global_place[i][j]))
                board[i] -= pow3[hw_m1 - j] * 2;
            else if (1 & (w >> global_place[i][j]))
                board[i] -= pow3[hw_m1 - j];
        }
    }
    return n_stones;
}

extern "C" int main(){
    cout << "initializing AI" << endl;
    init_pow();
    init_mod3();
    init_move();
    init_local_place();
    init_included();
    init_pop_digit();
    init_pop_mid();
    init_book();
    init_evaluation();
    f_search_table_idx = 0;
    search_hash_table_init(f_search_table_idx);
    cout << "iniitialized AI" << endl;
    return 0;
}

extern "C" void init_ai(int a_player, int r_depth, int w_r_depth, int b_depth){
    ai_player = a_player;
    read_depth = r_depth;
    win_read_depth = w_r_depth;
    book_depth = b_depth;
    cout << "AI param " << ai_player << " " << read_depth << " " << win_read_depth << " " << book_depth << endl;
}

extern "C" double ai(int *arr_board){
    cout << "start AI" << endl;
    int i, n_stones, policy;
    board b;
    search_result result;
    cout << endl;
    n_stones = input_board(b.b, arr_board);
    b.n = n_stones;
    b.p = ai_player;
    print_board(b.b);
    cout << evaluate(&b) << endl;
    cout << n_stones - 4 << "moves" << endl;
    if (n_stones < book_depth){
        policy = get_book(b.b);
        cout << "book policy " << policy << endl;
        if (policy != -1){
            b = move(&b, policy);
            ++n_stones;
            result = search(b, read_depth, win_read_depth);
            return output_coord(policy, -result.value);
        }
    }
    result = search(b, read_depth, win_read_depth);
    cout << "policy " << result.policy << endl;
    return output_coord(result.policy, result.value);
}

extern "C" void calc_value(int *arr_board, int e_count, int direction, int *res){
    ai_player = 1 - ai_player;
    int i, n_stones, policy;
    board b;
    search_result result;
    n_stones = input_board(b.b, arr_board);
    print_board(b.b);
    b.n = n_stones;
    b.p = ai_player;
    cout << evaluate(&b) << endl;
    cout << n_stones - 4 << "moves" << endl;
    int tmp_res[hw2];
    vector<int> moves;
    for (const int &cell: vacant_lst){
        for (const int &idx: place_included[cell]){
            if (move_arr[b.p][b.b[idx]][local_place[idx][cell]][0] || move_arr[b.p][b.b[idx]][local_place[idx][cell]][1]){
                cout << cell << " ";
                moves.push_back(cell);
                break;
            }
        }
    }
    cout << endl;
    for (i = 0; i < hw2; ++i)
        tmp_res[i] = -1;
    board nb, rb;
    bool passed;
    for (const int &policy: moves){
        nb = move(&b, policy);
        passed = true;
        for (const int &cell: vacant_lst){
            for (const int &idx: place_included[cell]){
                if (move_arr[nb.p][nb.b[idx]][local_place[idx][cell]][0] || move_arr[nb.p][nb.b[idx]][local_place[idx][cell]][1]){
                    passed = false;
                    break;
                }
            }
        }
        if (passed){
            for (i = 0; i < b_idx_num; ++i)
                rb.b[i] = nb.b[i];
            rb.p = 1 - nb.p;
            rb.n = nb.n;
            for (const int &cell: vacant_lst){
                for (const int &idx: place_included[cell]){
                    if (move_arr[rb.p][rb.b[idx]][local_place[idx][cell]][0] || move_arr[rb.p][rb.b[idx]][local_place[idx][cell]][1]){
                        passed = false;
                        break;
                    }
                }
            }
            if (passed){
                tmp_res[policy] = calc_result_value(-end_game(&nb));
                continue;
            }
        }
        tmp_res[policy] = calc_result_value(-search(nb, 5, 0).value);
    }
    for (i = 0; i < hw2; ++i)
        res[10 + i] = max(0, min(100, tmp_res[i]));
    ai_player = 1 - ai_player;
}
