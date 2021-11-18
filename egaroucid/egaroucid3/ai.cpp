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
REPLACE_BOOK_HERE
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
REPLACE_PARAM_HERE
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
