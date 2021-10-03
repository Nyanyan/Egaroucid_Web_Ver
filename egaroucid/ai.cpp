// Egaroucid2

#include <iostream>
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

#define inf 100000.0
#define board_index_num 38

#define complete_stones 12
#define simple_threshold 3
#define hash_table_size 16384
#define hash_mask (hash_table_size - 1)

#define c_puct 2.0
#define c_end 1.0
#define c_value 0.5
#define mcts_complete_stones 8

#define n_board_input 3
#define kernel_size 3
#define n_kernels 32
#define n_residual 2
#define n_dense1_policy 64
#define n_dense1_value 32
#define n_dense2_value 16
#define conv_size (hw_p1 - kernel_size)
#define conv_padding (kernel_size / 2)
#define conv_padding2 (conv_padding * 2)

#define n_div 1000000
#define tanh_min -5.0
#define tanh_max 5.0
#define exp_min -20.0
#define exp_max 20.0

#define compress_digit 3

struct book_elem{
    int p;
    double v;
};

struct board_param{
    unsigned long long trans[board_index_num][6561][hw];
    unsigned long long neighbor8[board_index_num][6561][hw];
    bool legal[6561][hw];
    int put[hw2][board_index_num];
    int board_translate[board_index_num][8];
    int board_rev_translate[hw2][4][2];
    int pattern_space[board_index_num];
    int reverse[6561];
    int pow3[15];
    int rev_bit3[6561][8];
    int pop_digit[6561][8];
    int digit_pow[3][10];
    int put_idx[hw2][10];
    int put_idx_num[hw2];
    int restore_p[6561][hw], restore_o[6561][hw], restore_vacant[6561][hw];
    int turn_board[4][hw2];
    int direction;
    int board[board_index_num];
    int n_stones;
};

struct eval_param{
    double weight[hw2];
    double avg_canput[hw2];
    int canput[6561];
    int cnt_p[6561], cnt_o[6561];
    double open_eval[40];
    double tanh_arr[n_div];
    double exp_arr[n_div];

    double conv1[n_kernels][n_board_input][kernel_size][kernel_size];
    double conv_residual[n_residual][n_kernels][n_kernels][kernel_size][kernel_size];
    double dense1_policy[n_kernels][n_dense1_policy];
    double bias1_policy[n_dense1_policy];
    double dense2_policy[n_dense1_policy][hw2];
    double bias2_policy[hw2];

    double dense1_value[n_kernels][n_dense1_value];
    double bias1_value[n_dense1_value];
    double dense2_value[n_dense1_value][n_dense2_value];
    double bias2_value[n_dense2_value];
    double dense3_value[n_dense2_value];
    double bias3_value;
};

struct search_param{
    int max_depth;
    long long strt, tl;
    int turn;
    int searched_nodes;
    vector<int> vacant_lst;
    int vacant_cnt;
    int evaluate_count;
};

struct board_priority_move{
    int b[board_index_num];
    double priority;
    int move;
    double open_val;
};

struct board_priority{
    int b[board_index_num];
    double priority;
    double n_open_val;
};

struct open_vals{
    double p_open_val, o_open_val;
    int p_cnt, o_cnt;
};

struct mcts_node{
    int board[board_index_num];
    int children[hw2_p1];
    double p[hw2];
    double w;
    int n;
    bool pass;
    bool expanded;
    bool end;
};

struct mcts_param{
    mcts_node nodes[10000];
    int used_idx;
    double sqrt_arr[10000];
};

struct predictions{
    double policies[hw2];
    double value;
};

board_param board_param;
eval_param eval_param;
search_param search_param;
mcts_param mcts_param;

inline long long tim(){
    //return static_cast<int>(chrono::duration_cast<chrono::milliseconds>(chrono::steady_clock::now().time_since_epoch()).count());
    return chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now().time_since_epoch()).count();
}

inline int map_liner(double x, double mn, double mx){
    return max(0, min(n_div - 1, (int)((x - mn) / (mx - mn) * n_div)));
}

inline double rev_map_liner(int x, double mn, double mx){
    return (double)x / (double)n_div * (mx - mn) + mn;
}

void print_board_line(int tmp){
    int j;
    for (j = 0; j < hw; ++j){
        if (tmp % 3 == 0){
            cout << ". ";
        }else if (tmp % 3 == 1){
            cout << "P ";
        }else{
            cout << "O ";
        }
        tmp /= 3;
    }
}

void print_board(int* board){
    int i, j, idx, tmp;
    for (i = 0; i < hw; ++i){
        tmp = board[i];
        for (j = 0; j < hw; ++j){
            if (tmp % 3 == 0){
                cout << ". ";
            }else if (tmp % 3 == 1){
                cout << "P ";
            }else{
                cout << "O ";
            }
            tmp /= 3;
        }
        cout << endl;
    }
    cout << endl;
}

int reverse_line(int a) {
    int res = 0;
    for (int i = 0; i < hw; ++i) {
        res <<= 1;
        res |= 1 & (a >> i);
    }
    return res;
}

inline int check_mobility(const int p, const int o){
	int p1 = p << 1;
    int res = ~(p1 | o) & (p1 + o);
    int p_rev = reverse_line(p), o_rev = reverse_line(o);
    int p2 = p_rev << 1;
    res |= reverse_line(~(p2 | o_rev) & (p2 + o_rev));
    res &= ~(p | o);
    // cout << bitset<8>(p) << " " << bitset<8>(o) << " " << bitset<8>(res) << endl;
    return res;
}

int trans(int pt, int k) {
    if (k == 0)
        return pt >> 1;
    else
        return pt << 1;
}

int move_line(int p, int o, const int place) {
    int rev = 0;
    int rev2, mask, tmp;
    int pt = 1 << place;
    for (int k = 0; k < 2; ++k) {
        rev2 = 0;
        mask = trans(pt, k);
        while (mask && (mask & o)) {
            rev2 |= mask;
            tmp = mask;
            mask = trans(tmp, k);
            if (mask & p)
                rev |= rev2;
        }
    }
    // cout << bitset<8>(p) << " " << bitset<8>(o) << " " << bitset<8>(rev | pt) << endl;
    return rev | pt;
}

int create_p(int idx){
    int res = 0;
    for (int i = 0; i < hw; ++i){
        if (idx % 3 == 1){
            res |= 1 << i;
        }
        idx /= 3;
    }
    return res;
}

int create_o(int idx){
    int res = 0;
    for (int i = 0; i < hw; ++i){
        if (idx % 3 == 2){
            res |= 1 << i;
        }
        idx /= 3;
    }
    return res;
}

int board_reverse(int idx){
    int p = create_p(idx);
    int o = create_o(idx);
    int res = 0;
    for (int i = hw_m1; i >= 0; --i){
        res *= 3;
        if (1 & (p >> i))
            res += 2;
        else if (1 & (o >> i))
            ++res;
    }
    return res;
}

constexpr int ln_char = 91;
constexpr double compress_bias = 21.32882;
const string chars = "!#$&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~";
double pow_ln_char[3] = {1.0, 0.01098901, 0.00012076};
unordered_map<char, double> char_keys;

inline double unzip_element(char a, char b, char c){
    return char_keys[a] + char_keys[b] * 0.01098901 + char_keys[c] * 0.00012076 - compress_bias;
}

extern "C" int init_ai(){
    cout << "start init" << endl;
    long long strt = tim();
    int i, j, k, l;
    static int translate[hw2] = {
        0, 1, 2, 3, 3, 2, 1, 0,
        1, 4, 5, 6, 6, 5, 4, 1,
        2, 5, 7, 8, 8, 7, 5, 2,
        3, 6, 8, 9, 9, 8, 6, 3,
        3, 6, 8, 9, 9, 8, 6, 3,
        2, 5, 7, 8, 8, 7, 5, 2,
        1, 4, 5, 6, 6, 5, 4, 1,
        0, 1, 2, 3, 3, 2, 1, 0
    };
    const int consts[476] = {
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57, 2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59, 4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61, 6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63, 5, 14, 23, 4, 13, 22, 31, 3, 12, 21, 30, 39, 2, 11, 20, 29, 38, 47, 1, 10, 19, 28, 37, 46, 55, 0, 9, 18, 27, 36, 45, 54, 63, 8,
        17, 26, 35, 44, 53, 62, 16, 25, 34, 43, 52, 61, 24, 33, 42, 51, 60, 32, 41, 50, 59, 40, 49, 58, 2, 9, 16, 3, 10, 17, 24, 4, 11, 18, 25, 32, 5, 12, 19, 26, 33, 40, 6, 13, 20, 27, 34, 41, 48, 7, 14, 21, 28, 35, 42, 49, 56, 15, 22, 29, 36, 43, 50, 57, 23, 30, 37, 44, 51, 58, 31, 38, 45, 52, 59, 39, 46, 53, 60, 47, 54, 61, 10, 8, 8, 8, 8, 4, 4, 8, 2, 4, 54, 63, 62, 61, 60, 59, 58, 57,
        56, 49, 49, 56, 48, 40, 32, 24, 16, 8, 0, 9, 9, 0, 1, 2, 3, 4, 5, 6, 7, 14, 14, 7, 15, 23, 31, 39, 47, 55, 63, 54, 3, 2, 1, 0, 9, 8, 16, 24, 4, 5, 6, 7, 14, 15, 23, 31, 60, 61, 62, 63, 54, 55, 47, 39, 59, 58, 57, 56, 49, 48, 40, 32, 0, 1, 2, 3, 8, 9, 10, 11, 0, 8, 16, 24, 1, 9, 17, 25, 7, 6, 5, 4, 15, 14, 13, 12, 7, 15, 23, 31, 6, 14, 22, 30, 63, 62, 61, 60,
        55, 54, 53, 52, 63, 55, 47, 39, 62, 54, 46, 38, 56, 57, 58, 59, 48, 49, 50, 51, 56, 48, 40, 32, 57, 49, 41, 33, 0, 9, 18, 27, 36, 45, 54, 63, 7, 14, 21, 28, 35, 42, 49, 56, 0, 1, 2, 3, 4, 5, 6, 7, 7, 15, 23, 31, 39, 47, 55, 63, 63, 62, 61, 60, 59, 58, 57, 56, 56, 48, 40, 32, 24, 26, 8, 0
    };
    const string param_compressed = 
REPLACE_PARAM_HERE

    int all_idx = 0;
    for (i = 0; i < board_index_num; ++i)
        board_param.pattern_space[i] = consts[all_idx++];
    for (i = 0; i < board_index_num; ++i){
        for (j = 0; j < board_param.pattern_space[i]; ++j)
            board_param.board_translate[i][j] = consts[all_idx++];
    }
    int idx;
    for (i = 0; i < hw2; ++i){
        idx = 0;
        for (j = 0; j < board_index_num; ++j){
            for (k = 0; k < board_param.pattern_space[j]; ++k){
                if (board_param.board_translate[j][k] == i){
                    board_param.board_rev_translate[i][idx][0] = j;
                    board_param.board_rev_translate[i][idx++][1] = k;
                }
            }
        }
        for (j = idx; j < 4; ++j)
            board_param.board_rev_translate[i][j][0] = -1;
    }
    for (i = 0; i < hw2; ++i){
        for (j = 0; j < board_index_num; ++j){
            board_param.put[i][j] = -1;
            for (k = 0; k < board_param.pattern_space[j]; ++k){
                if (board_param.board_translate[j][k] == i)
                    board_param.put[i][j] = k;
            }
        }
    }
    for (i = 0; i < ln_char; ++i)
        char_keys[chars[i]] = (double)i;
    int compress_idx = 0;
    for (i = 0; i < n_kernels; ++i){
        for (j = 0; j < n_board_input; ++j){
            for (k = 0; k < kernel_size; ++k){
                for (l = 0; l < kernel_size; ++l){
                    eval_param.conv1[i][j][k][l] = unzip_element(param_compressed[compress_idx], param_compressed[compress_idx + 1], param_compressed[compress_idx + 2]);
                    compress_idx += compress_digit;
                }
            }
        }
    }
    int residual_i;
    for (residual_i = 0; residual_i < n_residual; ++residual_i){
        for (i = 0; i < n_kernels; ++i){
            for (j = 0; j < n_kernels; ++j){
                for (k = 0; k < kernel_size; ++k){
                    for (l = 0; l < kernel_size; ++l){
                        eval_param.conv_residual[residual_i][i][j][k][l] = unzip_element(param_compressed[compress_idx], param_compressed[compress_idx + 1], param_compressed[compress_idx + 2]);
                        compress_idx += compress_digit;
                    }
                }
            }
        }
    }
    for (i = 0; i < n_kernels; ++i){
        for (j = 0; j < n_dense1_value; ++j){
            eval_param.dense1_value[i][j] = unzip_element(param_compressed[compress_idx], param_compressed[compress_idx + 1], param_compressed[compress_idx + 2]);
            compress_idx += compress_digit;
        }
    }
    for (i = 0; i < n_dense1_value; ++i){
        eval_param.bias1_value[i] = unzip_element(param_compressed[compress_idx], param_compressed[compress_idx + 1], param_compressed[compress_idx + 2]);
        compress_idx += compress_digit;
    }
    for (i = 0; i < n_kernels; ++i){
        for (j = 0; j < n_dense1_policy; ++j){
            eval_param.dense1_policy[i][j] = unzip_element(param_compressed[compress_idx], param_compressed[compress_idx + 1], param_compressed[compress_idx + 2]);
            compress_idx += compress_digit;
        }
    }
    for (i = 0; i < n_dense1_policy; ++i){
        eval_param.bias1_policy[i] = unzip_element(param_compressed[compress_idx], param_compressed[compress_idx + 1], param_compressed[compress_idx + 2]);
        compress_idx += compress_digit;
    }
    for (i = 0; i < n_dense1_value; ++i){
        for (j = 0; j < n_dense2_value; ++j){
            eval_param.dense2_value[i][j] = unzip_element(param_compressed[compress_idx], param_compressed[compress_idx + 1], param_compressed[compress_idx + 2]);
            compress_idx += compress_digit;
        }
    }
    for (i = 0; i < n_dense2_value; ++i){
        eval_param.bias2_value[i] = unzip_element(param_compressed[compress_idx], param_compressed[compress_idx + 1], param_compressed[compress_idx + 2]);
        compress_idx += compress_digit;
    }
    for (i = 0; i < n_dense1_policy; ++i){
        for (j = 0; j < hw2; ++j){
            eval_param.dense2_policy[i][j] = unzip_element(param_compressed[compress_idx], param_compressed[compress_idx + 1], param_compressed[compress_idx + 2]);
            compress_idx += compress_digit;
        }
    }
    for (i = 0; i < hw2; ++i){
        eval_param.bias2_policy[i] = unzip_element(param_compressed[compress_idx], param_compressed[compress_idx + 1], param_compressed[compress_idx + 2]);
        compress_idx += compress_digit;
    }
    for (i = 0; i < n_dense2_value; ++i){
        eval_param.dense3_value[i] = unzip_element(param_compressed[compress_idx], param_compressed[compress_idx + 1], param_compressed[compress_idx + 2]);
        compress_idx += compress_digit;
    }
    eval_param.bias3_value = unzip_element(param_compressed[compress_idx], param_compressed[compress_idx + 1], param_compressed[compress_idx + 2]);
    compress_idx += compress_digit;

    int p, o, mobility, canput_num, rev;
    for (i = 0; i < 6561; ++i){
        board_param.reverse[i] = board_reverse(i);
        p = reverse_line(create_p(i));
        o = reverse_line(create_o(i));
        eval_param.cnt_p[i] = 0;
        eval_param.cnt_o[i] = 0;
        for (j = 0; j < hw; ++j){
            board_param.restore_p[i][j] = 1 & (p >> (hw_m1 - j));
            board_param.restore_o[i][j] = 1 & (o >> (hw_m1 - j));
            board_param.restore_vacant[i][j] = 1 & ((~(p | o)) >> (hw_m1 - j));
            eval_param.cnt_p[i] += board_param.restore_p[i][j];
            eval_param.cnt_o[i] += board_param.restore_o[i][j];
        }
        mobility = check_mobility(p, o);
        canput_num = 0;
        for (j = 0; j < hw; ++j){
            if (1 & (mobility >> (hw_m1 - j))){
                rev = move_line(p, o, hw_m1 - j);
                ++canput_num;
                board_param.legal[i][j] = true;
                for (k = 0; k < board_index_num; ++k){
                    board_param.trans[k][i][j] = 0;
                    for (l = 0; l < board_param.pattern_space[k]; ++l)
                        board_param.trans[k][i][j] |= (unsigned long long)(1 & (rev >> (7 - l))) << board_param.board_translate[k][l];
                    board_param.neighbor8[k][i][j] = 0;
                    board_param.neighbor8[k][i][j] |= (0b0111111001111110011111100111111001111110011111100111111001111110 & board_param.trans[k][i][j]) << 1;
                    board_param.neighbor8[k][i][j] |= (0b0111111001111110011111100111111001111110011111100111111001111110 & board_param.trans[k][i][j]) >> 1;
                    board_param.neighbor8[k][i][j] |= (0b0000000011111111111111111111111111111111111111111111111100000000 & board_param.trans[k][i][j]) << hw;
                    board_param.neighbor8[k][i][j] |= (0b0000000011111111111111111111111111111111111111111111111100000000 & board_param.trans[k][i][j]) >> hw;
                    board_param.neighbor8[k][i][j] |= (0b0000000001111110011111100111111001111110011111100111111000000000 & board_param.trans[k][i][j]) << hw_m1;
                    board_param.neighbor8[k][i][j] |= (0b0000000001111110011111100111111001111110011111100111111000000000 & board_param.trans[k][i][j]) >> hw_m1;
                    board_param.neighbor8[k][i][j] |= (0b0000000001111110011111100111111001111110011111100111111000000000 & board_param.trans[k][i][j]) << hw_p1;
                    board_param.neighbor8[k][i][j] |= (0b0000000001111110011111100111111001111110011111100111111000000000 & board_param.trans[k][i][j]) >> hw_p1;
                    board_param.neighbor8[k][i][j] &= ~board_param.trans[k][i][j];
                }
            } else
                board_param.legal[i][j] = false;
        }
    }
    for (i = 0; i < hw2; ++i){
        board_param.put_idx_num[i] = 0;
        for (j = 0; j < board_index_num; ++j){
            if (board_param.put[i][j] != -1)
                board_param.put_idx[i][board_param.put_idx_num[i]++] = j;
        }
    }
    for (i = 0; i < 15; ++i)
        board_param.pow3[i] = (int)pow(3, i);
    for (i = 0; i < 6561; ++i){
        for (j = 0; j < 8; ++j){
            board_param.rev_bit3[i][j] = board_param.pow3[j] * (2 - (i / board_param.pow3[j]) % 3);
            board_param.pop_digit[i][j] = i / board_param.pow3[j] % 3;
        }
    }
    for (i = 0; i < 3; ++i){
        for (j = 0; j < 10; ++j)
            board_param.digit_pow[i][j] = i * board_param.pow3[j];
    }
    for (i = 0; i < 40; ++i)
        eval_param.open_eval[i] = min(1.0, pow(2.0, 2.0 - 0.667 * i) - 1.0);
    for (i = 0; i < n_div; ++i){
        eval_param.tanh_arr[i] = tanh(rev_map_liner(i, tanh_min, tanh_max));
        eval_param.exp_arr[i] = exp(rev_map_liner(i, exp_min, exp_max));
    }
    for (i = 0; i < 100; ++i)
        mcts_param.sqrt_arr[i] = sqrt((double)i);
    for (i = 0; i < hw; ++i){
        for (j = 0; j < hw; ++j){
            board_param.turn_board[0][i * hw + j] = i * hw + j;
            board_param.turn_board[1][i * hw + j] = j * hw + i;
            board_param.turn_board[2][i * hw + j] = (hw_m1 - i) * hw + (hw_m1 - j);
            board_param.turn_board[3][i * hw + j] = (hw_m1 - j) * hw + (hw_m1 - i);
        }
    }
    board_param.direction = -1;
    cout << "initialized" << endl;
    return 0;
}

inline double leaky_relu(double x){
    return max(x, 0.01 * x);
}

inline predictions predict(const int *board){
    int i, j, k, sy, sx, y, x, coord1, coord2, residual_i;
    predictions res;
    double input_board[n_board_input][hw + conv_padding2][hw + conv_padding2];
    double hidden_conv1[n_kernels][hw + conv_padding2][hw + conv_padding2];
    double hidden_conv2[n_kernels][hw + conv_padding2][hw + conv_padding2];
    double after_conv[n_kernels];
    double hidden1[64];
    double hidden2[64];
    // reshape input
    for (i = 0; i < hw; ++i){
        for (j = 0; j < hw; ++j){
            input_board[0][i + conv_padding][j + conv_padding] = board_param.restore_p[board[i]][j];
            input_board[1][i + conv_padding][j + conv_padding] = board_param.restore_o[board[i]][j];
            input_board[2][i + conv_padding][j + conv_padding] = board_param.restore_vacant[board[i]][j];
        }
        for (j = 0; j < hw + conv_padding2; ++j){
            for (k = 0; k < n_board_input; ++k){
                input_board[k][0][j] = 0.0;
                input_board[k][hw_m1 + conv_padding2][j] = 0.0;
                input_board[k][j][0] = 0.0;
                input_board[k][j][hw_m1 + conv_padding2] = 0.0;
            }
        }
    }
    // conv and normalization and leaky-relu
    for (i = 0; i < n_kernels; ++i){
        for (y = 0; y < hw + conv_padding2; ++y){
            for (x = 0; x < hw + conv_padding2; ++x)
                hidden_conv1[i][y][x] = 0.0;
        }
        for (j = 0; j < n_board_input; ++j){
            for (sy = 0; sy < hw; ++sy){
                for (sx = 0; sx < hw; ++sx){
                    for (y = 0; y < kernel_size; ++y){
                        for (x = 0; x < kernel_size; ++x)
                            hidden_conv1[i][sy + conv_padding][sx + conv_padding] += eval_param.conv1[i][j][y][x] * input_board[j][sy + y][sx + x];
                    }
                }
            }
        }
        for (y = conv_padding; y < hw + conv_padding; ++y){
            for (x = conv_padding; x < hw + conv_padding; ++x)
                hidden_conv1[i][y][x] = leaky_relu(hidden_conv1[i][y][x]);
        }
    }
    // residual-error-block
    for (residual_i = 0; residual_i < n_residual; ++residual_i){
        for (i = 0; i < n_kernels; ++i){
            for (y = 0; y < hw + conv_padding2; ++y){
                for (x = 0; x < hw + conv_padding2; ++x)
                    hidden_conv2[i][y][x] = 0.0;
            }
            for (j = 0; j < n_kernels; ++j){
                for (sy = 0; sy < hw; ++sy){
                    for (sx = 0; sx < hw; ++sx){
                        for (y = 0; y < kernel_size; ++y){
                            for (x = 0; x < kernel_size; ++x)
                                hidden_conv2[i][sy + conv_padding][sx + conv_padding] += eval_param.conv_residual[residual_i][i][j][y][x] * hidden_conv1[j][sy + y][sx + x];
                        }
                    }
                }
            }
        }
        for (i = 0; i < n_kernels; ++i){
            for (y = conv_padding; y < hw + conv_padding; ++y){
                for (x = conv_padding; x < hw + conv_padding; ++x)
                    hidden_conv1[i][y][x] = leaky_relu(hidden_conv1[i][y][x] + hidden_conv2[i][y][x]);
            }
        }
    }
    // global-average-pooling
    for (i = 0; i < n_kernels; ++i){
        after_conv[i] = 0.0;
        for (y = 0; y < hw; ++y){
            for (x = 0; x < hw; ++x)
                after_conv[i] += hidden_conv1[i][y + conv_padding][x + conv_padding];
        }
        after_conv[i] /= hw2;
    }

    // dense1 for policy
    for (j = 0; j < n_dense1_policy; ++j){
        hidden1[j] = eval_param.bias1_policy[j];
        for (i = 0; i < n_kernels; ++i)
            hidden1[j] += eval_param.dense1_policy[i][j] * after_conv[i];
        hidden1[j] = leaky_relu(hidden1[j]);
    }
    // dense2 for policy
    for (j = 0; j < hw2; ++j){
        res.policies[j] = eval_param.bias2_policy[j];
        for (i = 0; i < n_dense1_policy; ++i)
            res.policies[j] += eval_param.dense2_policy[i][j] * hidden1[i];
    }

    // dense1 for value
    for (j = 0; j < n_dense1_value; ++j){
        hidden2[j] = eval_param.bias1_value[j];
        for (i = 0; i < n_kernels; ++i)
            hidden2[j] += eval_param.dense1_value[i][j] * after_conv[i];
        hidden1[j] = leaky_relu(hidden2[j]);
    }
    // dense2 for value
    for (j = 0; j < n_dense2_value; ++j){
        hidden2[j] = eval_param.bias2_value[j];
        for (i = 0; i < n_dense1_value; ++i)
            hidden2[j] += eval_param.dense2_value[i][j] * hidden1[i];
        hidden2[j] = leaky_relu(hidden2[j]);
    }
    // dense3 for value
    res.value = eval_param.bias3_value;
    for (i = 0; i < n_dense2_value; ++i)
        res.value += eval_param.dense3_value[i] * hidden2[i];
    res.value = eval_param.tanh_arr[map_liner(res.value, tanh_min, tanh_max)];

    // return
    return res;
}

inline void move(int *board, int (&res)[board_index_num], int coord){
    int i, j, tmp;
    unsigned long long rev = 0;
    for (i = 0; i < board_index_num; ++i){
        res[i] = board_param.reverse[board[i]];
        if (board_param.put[coord][i] != -1)
            rev |= board_param.trans[i][board[i]][board_param.put[coord][i]];
    }
    for (i = 0; i < hw2; ++i){
        if (1 & (rev >> i)){
            for (j = 0; j < 4; ++j){
                if (board_param.board_rev_translate[i][j][0] == -1)
                    break;
                res[board_param.board_rev_translate[i][j][0]] += board_param.rev_bit3[res[board_param.board_rev_translate[i][j][0]]][board_param.board_rev_translate[i][j][1]];
            }
        }
    }
}

inline int move_open(int *board, int (&res)[board_index_num], int coord){
    int i, j, tmp;
    unsigned long long rev = 0, neighbor = 0;
    for (i = 0; i < board_index_num; ++i){
        res[i] = board_param.reverse[board[i]];
        if (board_param.put[coord][i] != -1){
            rev |= board_param.trans[i][board[i]][board_param.put[coord][i]];
            neighbor |= board_param.neighbor8[i][board[i]][board_param.put[coord][i]];
        }
    }
    for (i = 0; i < hw2; ++i){
        if (1 & (rev >> i)){
            for (j = 0; j < 4; ++j){
                if (board_param.board_rev_translate[i][j][0] == -1)
                    break;
                res[board_param.board_rev_translate[i][j][0]] += board_param.rev_bit3[res[board_param.board_rev_translate[i][j][0]]][board_param.board_rev_translate[i][j][1]];
            }
        }
    }
    int open_val = 0;
    for (i = 0; i < hw2; ++i){
        if(1 & (neighbor >> i))
            open_val += (int)(board_param.pop_digit[board[i >> 3]][i & 0b111] == 0);
    }
    return open_val;
}

inline double end_game(const int *board){
    int res = 0, i, j, p, o;
    for (i = 0; i < hw; ++i){
        res += eval_param.cnt_p[board[i]];
        res -= eval_param.cnt_o[board[i]];
    }
    if (res > 0)
        return 1.0;
    else if (res < 0)
        return -1.0;
    return 0.0;
}

inline open_vals open_val_forward(int *board, int depth, bool player){
    open_vals res;
    if (depth == 0){
        res.p_open_val = 0.0;
        res.o_open_val = 0.0;
        res.p_cnt = 0;
        res.o_cnt = 0;
        return res;
    }
    --depth;
    int i, j;
    int n_board[board_index_num];
    open_vals tmp;
    res.p_open_val = -inf;
    res.o_open_val = inf;
    double open_val = -inf;
    bool passed = false;
    for (const int& cell : search_param.vacant_lst){
        for (i = 0; i < board_param.put_idx_num[cell]; ++i){
            if (board_param.legal[board[board_param.put_idx[cell][i]]][board_param.put[cell][board_param.put_idx[cell][i]]]){
                passed = false;
                open_val = max(open_val, eval_param.open_eval[move_open(board, n_board, cell)]);
                tmp = open_val_forward(n_board, depth, !player);
                if (res.p_open_val < tmp.p_open_val){
                    res.p_open_val = tmp.p_open_val;
                    res.p_cnt = tmp.p_cnt;
                }
                if (res.o_open_val > tmp.o_open_val){
                    res.o_open_val = tmp.o_open_val;
                    res.o_cnt = tmp.o_cnt;
                }
            }
        }
    }
    if (passed){
        res.p_open_val = 0.0;
        res.o_open_val = 0.0;
        res.p_cnt = 0;
        res.o_cnt = 0;
        return res;
    }
    if (player){
        res.p_open_val += open_val;
        ++res.p_cnt;
    } else {
        res.o_open_val += open_val;
        ++res.o_cnt;
    }
    return res;
}

int cmp(board_priority p, board_priority q){
    return p.priority > q.priority;
}

double nega_alpha_light(int *board, const int depth, double alpha, double beta, const int skip_cnt){
    ++search_param.searched_nodes;
    if (skip_cnt == 2)
        return end_game(board);
    bool is_pass = true;
    int i, j, k;
    double v = -1.5, g;
    int n_board[board_index_num];
    int n_depth = depth - 1;
    for (const int& cell : search_param.vacant_lst){
        for (i = 0; i < board_index_num; ++i){
            if (board_param.put[cell][i] != -1){
                if (board_param.legal[board[i]][board_param.put[cell][i]]){
                    is_pass = false;
                    move_open(board, n_board, cell);
                    g = -nega_alpha_light(n_board, n_depth, -beta, -alpha, 0);
                    if (beta <= g)
                        return g;
                    alpha = max(alpha, g);
                    v = max(v, g);
                    break;
                }
            }
        }
    }
    if (is_pass){
        for (i = 0; i < board_index_num; ++i)
            n_board[i] = board_param.reverse[board[i]];
        return -nega_alpha_light(n_board, depth, -beta, -alpha, skip_cnt + 1);
    }
    return v;
}

double nega_alpha(int *board, const int depth, double alpha, double beta, const int skip_cnt){
    if (depth < simple_threshold)
        return nega_alpha_light(board, depth, alpha, beta, skip_cnt);
    ++search_param.searched_nodes;
    if (skip_cnt == 2)
        return end_game(board);
    int i, j, k, canput = 0;
    double v = -1.5, g;
    board_priority lst[30];
    for (const int& cell : search_param.vacant_lst){
        for (i = 0; i < board_param.put_idx_num[cell]; ++i){
            if (board_param.legal[board[board_param.put_idx[cell][i]]][board_param.put[cell][board_param.put_idx[cell][i]]]){
                lst[canput].n_open_val = eval_param.open_eval[move_open(board, lst[canput].b, cell)];
                lst[canput].priority = lst[canput].n_open_val;
                ++canput;
                break;
            }
        }
    }
    if (canput == 0){
        int n_board[board_index_num];
        for (i = 0; i < board_index_num; ++i)
            n_board[i] = board_param.reverse[board[i]];
        return -nega_alpha(n_board, depth, -beta, -alpha, skip_cnt + 1);
    }
    int n_depth = depth - 1;
    if (canput > 1)
        sort(lst, lst + canput, cmp);
    for (i = 0; i < canput; ++i){
        g = -nega_alpha(lst[i].b, n_depth, -beta, -alpha, 0);
        if (fabs(g) == inf)
            return -inf;
        if (beta < g)
            return g;
        alpha = max(alpha, g);
        v = max(v, g);
    }
    return v;
}

double nega_alpha_heavy(int *board, int depth, double alpha, double beta, int skip_cnt){
    if (depth <= search_param.max_depth - 3)
        return nega_alpha(board, depth, alpha, beta, skip_cnt);
    ++search_param.searched_nodes;
    if (skip_cnt == 2)
        return end_game(board);
    int i, j, canput = 0;
    board_priority lst[30];
    open_vals tmp_open_vals;
    for (const int& cell : search_param.vacant_lst){
        for (i = 0; i < board_param.put_idx_num[cell]; ++i){
            if (board_param.legal[board[board_param.put_idx[cell][i]]][board_param.put[cell][board_param.put_idx[cell][i]]]){
                lst[canput].n_open_val = eval_param.open_eval[move_open(board, lst[canput].b, cell)];
                tmp_open_vals = open_val_forward(lst[canput].b, 1, true);
                if (tmp_open_vals.p_cnt)
                    lst[canput].priority = (lst[canput].n_open_val + tmp_open_vals.o_open_val) / tmp_open_vals.o_cnt - tmp_open_vals.p_open_val / tmp_open_vals.p_cnt;
                else
                    lst[canput].priority = (lst[canput].n_open_val + tmp_open_vals.o_open_val) / tmp_open_vals.o_cnt;
                ++canput;
                break;
            }
        }
    }
    if (canput == 0){
        int n_board[board_index_num];
        for (i = 0; i < board_index_num; ++i)
            n_board[i] = board_param.reverse[board[i]];
        return -nega_alpha_heavy(n_board, depth, -beta, -alpha, skip_cnt + 1);
    }
    if (canput > 2)
        sort(lst, lst + canput, cmp);
    double v = -1.5, g;
    int n_depth = depth - 1;
    for (i = 0; i < canput; ++i){
        g = -nega_alpha_heavy(lst[i].b, n_depth, -beta, -alpha, 0);
        if (fabs(g) == inf)
            return -inf;
        if (beta < g)
            return g;
        alpha = max(alpha, g);
        v = max(v, g);
    }
    return v;
}

int cmp_main(board_priority_move p, board_priority_move q){
    return p.priority > q.priority;
}

int cmp_vacant(int p, int q){
    return eval_param.weight[p] > eval_param.weight[q];
}

inline pair<int, int> find_win(int *board){
    vector<board_priority_move> lst;
    int cell, i;
    int draw_move = -1;
    int canput = 0;
    double score;
    for (cell = 0; cell < hw2; ++cell){
        for (i = 0; i < board_index_num; ++i){
            if (board_param.put[cell][i] != -1){
                if (board_param.legal[board[i]][board_param.put[cell][i]]){
                    ++canput;
                    board_priority_move tmp;
                    tmp.open_val = eval_param.open_eval[move_open(board, tmp.b, cell)];
                    tmp.priority = tmp.open_val;
                    tmp.move = cell;
                    lst.push_back(tmp);
                    break;
                }
            }
        }
    }
    if (canput > 1)
        sort(lst.begin(), lst.end(), cmp_main);
    search_param.searched_nodes = 0;
    for (i = 0; i < canput; ++i){
        score = -nega_alpha_heavy(lst[i].b, search_param.max_depth, -1.1, 0.1, 0);
        if (score > 0.0)
            return make_pair(1, lst[i].move);
        else if (score == 0.0)
            draw_move = lst[i].move;
    }
    if (draw_move != -1)
        return make_pair(0, draw_move);
    else
        return make_pair(-1, lst[0].move);
}

inline double end_game_evaluate(int idx){
    return c_end * end_game(mcts_param.nodes[idx].board);
}

double evaluate(int idx, bool passed, int n_stones){
    double value = 0.0;
    int i, j;
    if (n_stones >= hw2 - mcts_complete_stones){
        //int result = find_win(mcts_param.nodes[idx].board).first;
        double result = c_end * (double)nega_alpha_heavy(mcts_param.nodes[idx].board, search_param.max_depth, -1.1, 1.1, 0);
        mcts_param.nodes[idx].w += result;
        ++mcts_param.nodes[idx].n;
        return result;
    }
    if (!mcts_param.nodes[idx].expanded){
        // when children not expanded
        // expand children
        mcts_param.nodes[idx].expanded = true;
        bool legal[hw2];
        mcts_param.nodes[idx].pass = true;
        for (int cell = 0; cell < hw2; ++cell){
            mcts_param.nodes[idx].children[cell] = -1;
            legal[cell] = false;
            for (i = 0; i < board_index_num; ++i){
                if (board_param.put[cell][i] != -1){
                    if (board_param.legal[mcts_param.nodes[idx].board[i]][board_param.put[cell][i]]){
                        mcts_param.nodes[idx].pass = false;
                        legal[cell] = true;
                        break;
                    }
                }
            }
        }
        mcts_param.nodes[idx].children[hw2] = -1;
        if (!mcts_param.nodes[idx].pass){
            //predict and create policy array
            predictions pred = predict(mcts_param.nodes[idx].board);
            mcts_param.nodes[idx].w += c_value * pred.value;
            ++mcts_param.nodes[idx].n;
            double p_sum = 0.0;
            for (i = 0; i < hw2; ++i){
                if (legal[i]){
                    mcts_param.nodes[idx].p[i] = eval_param.exp_arr[map_liner(pred.policies[i], exp_min, exp_max)];
                    p_sum += mcts_param.nodes[idx].p[i];
                } else{
                    mcts_param.nodes[idx].p[i] = 0.0;
                }
            }
            for (i = 0; i < hw2; ++i)
                mcts_param.nodes[idx].p[i] /= p_sum;
            return c_value * pred.value;
        } else{
            for (i = 0; i < hw2; ++i)
                mcts_param.nodes[idx].p[i] = 0.0;
            value = c_value * predict(mcts_param.nodes[idx].board).value;
            mcts_param.nodes[idx].w += value;
            ++mcts_param.nodes[idx].n;
            return value;
        }
    }
    if ((!mcts_param.nodes[idx].pass) && (!mcts_param.nodes[idx].end)){
        // children already expanded
        // select next move
        int a_cell = -1;
        value = -inf;
        double tmp_value;
        double t_sqrt = mcts_param.sqrt_arr[mcts_param.nodes[idx].n];
        for (const int& cell : search_param.vacant_lst){
            if (mcts_param.nodes[idx].p[cell] != 0.0){
                if (mcts_param.nodes[idx].children[cell] != -1)
                    tmp_value = c_puct * mcts_param.nodes[idx].p[cell] * t_sqrt / (1 + mcts_param.nodes[mcts_param.nodes[idx].children[cell]].n) - mcts_param.nodes[mcts_param.nodes[idx].children[cell]].w / mcts_param.nodes[mcts_param.nodes[idx].children[cell]].n;
                else
                    tmp_value = c_puct * mcts_param.nodes[idx].p[cell] * t_sqrt;
                if (value < tmp_value){
                    value = tmp_value;
                    a_cell = cell;
                }
            }
        }
        if (mcts_param.nodes[idx].children[a_cell] == -1){
            mcts_param.nodes[idx].children[a_cell] = mcts_param.used_idx;
            mcts_param.nodes[mcts_param.used_idx].w = 0.0;
            mcts_param.nodes[mcts_param.used_idx].n = 0;
            mcts_param.nodes[mcts_param.used_idx].pass = true;
            mcts_param.nodes[mcts_param.used_idx].expanded = false;
            mcts_param.nodes[mcts_param.used_idx].end = false;
            move(mcts_param.nodes[idx].board, mcts_param.nodes[mcts_param.used_idx++].board, a_cell);
        }
        value = -evaluate(mcts_param.nodes[idx].children[a_cell], false, n_stones + 1);
        mcts_param.nodes[idx].w += value;
        ++mcts_param.nodes[idx].n;
    } else{
        if (mcts_param.nodes[idx].end){
            value = mcts_param.nodes[idx].w / mcts_param.nodes[idx].n;
        } else{
            // pass
            if (mcts_param.nodes[idx].children[hw2] == -1){
                mcts_param.nodes[idx].children[hw2] = mcts_param.used_idx;
                mcts_param.nodes[mcts_param.used_idx].w = 0.0;
                mcts_param.nodes[mcts_param.used_idx].n = 0;
                mcts_param.nodes[mcts_param.used_idx].pass = true;
                mcts_param.nodes[mcts_param.used_idx].expanded = false;
                mcts_param.nodes[mcts_param.used_idx].end = false;
                for (i = 0; i < board_index_num; ++i)
                    mcts_param.nodes[mcts_param.used_idx].board[i] = board_param.reverse[mcts_param.nodes[idx].board[i]];
                ++mcts_param.used_idx;
            }
            if (passed){
                value = end_game_evaluate(idx);
                mcts_param.nodes[idx].end = true;
            } else{
                value = -evaluate(mcts_param.nodes[idx].children[hw2], true, n_stones);
            }
        }
        mcts_param.nodes[idx].w += value;
        ++mcts_param.nodes[idx].n;
    }
    return value;
}

inline void mcts_init(){
    int i, cell;
    mcts_param.used_idx = 1;
    for (i = 0; i < board_index_num; ++i)
        mcts_param.nodes[0].board[i] = board_param.board[i];
    mcts_param.nodes[0].w = 0.0;
    mcts_param.nodes[0].n = 0;
    mcts_param.nodes[0].pass = true;
    mcts_param.nodes[0].expanded = true;
    mcts_param.nodes[0].end = false;
    // expand children
    bool legal[hw2];
    for (cell = 0; cell < hw2; ++cell){
        mcts_param.nodes[0].children[cell] = -1;
        legal[cell] = false;
        for (i = 0; i < board_index_num; ++i){
            if (board_param.put[cell][i] != -1){
                if (board_param.legal[board_param.board[i]][board_param.put[cell][i]]){
                    mcts_param.nodes[0].pass = false;
                    legal[cell] = true;
                    break;
                }
            }
        }
    }
    //predict and create policy array
    predictions pred = predict(board_param.board);
    mcts_param.nodes[0].w += pred.value;
    ++mcts_param.nodes[0].n;
    double p_sum = 0.0;
    for (i = 0; i < hw2; ++i){
        if (legal[i]){
            mcts_param.nodes[0].p[i] = eval_param.exp_arr[map_liner(pred.policies[i], exp_min, exp_max)];
            p_sum += mcts_param.nodes[0].p[i];
        } else{
            mcts_param.nodes[0].p[i] = 0.0;
        }
    }
    for (i = 0; i < hw2; ++i)
        mcts_param.nodes[0].p[i] /= p_sum;
    board_param.n_stones = 0;
    for (i = 0; i < hw; ++i)
        board_param.n_stones += eval_param.cnt_p[board_param.board[i]] + eval_param.cnt_o[board_param.board[i]];
}

extern "C" void mcts_main(){
    int i;
    for (i = 0; i < search_param.evaluate_count; ++i)
        evaluate(0, false, board_param.n_stones);
}

extern "C" double mcts_end(){
    int i, mx = -inf, policy;
    for (i = 0; i < hw2; ++i){
        if (mcts_param.nodes[0].children[i] != -1){
            //cout << i << " " << mcts_param.nodes[mcts_param.nodes[0].children[i]].n << endl;
            if (mx < mcts_param.nodes[mcts_param.nodes[0].children[i]].n){
                mx = mcts_param.nodes[mcts_param.nodes[0].children[i]].n;
                policy = i;
            }
        }
    }
    cout << "SEARCH " << mcts_param.nodes[mcts_param.nodes[0].children[policy]].n << " " << mcts_param.used_idx << endl;
    cout << board_param.turn_board[board_param.direction][policy] / hw << " " << board_param.turn_board[board_param.direction][policy] % hw << " " << 50.0 - 50.0 * (double)mcts_param.nodes[mcts_param.nodes[0].children[policy]].w / mcts_param.nodes[mcts_param.nodes[0].children[policy]].n << endl;
    return 1000.0 * board_param.turn_board[board_param.direction][policy] + 50.0 - 50.0 * (double)mcts_param.nodes[mcts_param.nodes[0].children[policy]].w / mcts_param.nodes[mcts_param.nodes[0].children[policy]].n;
}

extern "C" double complete(){
    pair<int, int> result = find_win(board_param.board);
    if (result.first == 1)
        cout << "WIN" << endl;
    else if (result.first == 0)
        cout << "DRAW" << endl;
    else
        cout << "LOSE" << endl;
    cout << board_param.turn_board[board_param.direction][result.second] / hw << " " << board_param.turn_board[board_param.direction][result.second] % hw << " " << 50.0 + 50.0 * result.first << endl;
    return 1000.0 * board_param.turn_board[board_param.direction][result.second] + 50.0 + 50.0 * result.first;
}

extern "C" int start_ai(int *arr_board, int evaluate_count){
    int i, j, board_tmp, ai_player, policy;
    char elem;
    unsigned long long p, o;
    int n_stones;
    double rnd, sm;
    search_param.evaluate_count = evaluate_count;
    string raw_board;
    for (i = 0; i < hw2; ++i){
        if (arr_board[i] == 0)
            raw_board += "0";
        else if (arr_board[i] == 1)
            raw_board += "1";
        else
            raw_board += ".";
    }
    cout << raw_board << " " << evaluate_count << endl;
    search_param.turn = 0;
    p = 0;
    o = 0;
    n_stones = 0;
    search_param.vacant_lst = {};
    search_param.vacant_cnt = 0;
    if (board_param.direction == -1){
        for (i = 0; i < hw2; ++i){
            if (raw_board[i] != '.')
                ++n_stones;
        }
        if (n_stones == 4){
            board_param.direction = 0;
            cout << "FIRST" << endl;
            cout << 4 << " " << 5 << " " << 50.0 << endl;
            return 1000.0 * (4 * hw + 5) + 50.0;
        } else{
            string board_turns[4] = {
            "...........................01......111..........................",
            "...........................01......11.......1...................",
            "..........................111......10...........................",
            "...................1.......11......10..........................."};
            for (i = 0; i < 4; ++i){
                if (raw_board == board_turns[i])
                    board_param.direction = i;
            }
            n_stones = 0;
        }
    }
    for (i = 0; i < hw2; ++i){
        elem = raw_board[i];
        if (elem != '.'){
            ++search_param.turn;
            p |= (unsigned long long)(elem == '0') << board_param.turn_board[board_param.direction][i];
            o |= (unsigned long long)(elem == '1') << board_param.turn_board[board_param.direction][i];
            ++n_stones;
        } else{
            ++search_param.vacant_cnt;
            search_param.vacant_lst.push_back(board_param.turn_board[board_param.direction][i]);
        }
    }
    if (search_param.vacant_cnt)
        sort(search_param.vacant_lst.begin(), search_param.vacant_lst.end(), cmp_vacant);
    for (i = 0; i < board_index_num; ++i){
        board_tmp = 0;
        for (j = 0; j < board_param.pattern_space[i]; ++j){
            if (1 & (p >> board_param.board_translate[i][j]))
                board_tmp += board_param.pow3[j];
            else if (1 & (o >> board_param.board_translate[i][j]))
                board_tmp += 2 * board_param.pow3[j];
        }
        board_param.board[i] = board_tmp;
    }
    if (n_stones < hw2 - complete_stones){
        cout << "MCTS" << endl;
        mcts_init();
        return 0;
    } else{
        cout << "NEGASCOUT" << endl;
        search_param.max_depth = hw2 + 1 - n_stones;
        return 1;
    }
}
