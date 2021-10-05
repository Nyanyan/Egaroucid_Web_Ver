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
#define c_value 0.75
#define c_prev 0.5
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

struct node_t{
    int k[hw];
    double v;
    node_t* p_n_node;
};

inline int calc_hash(const int *p){
    int seed = 0;
    for (int i = 0; i < hw; ++i)
        seed ^= p[i] << (i / 4);
    return seed & hash_mask;
}

inline void hash_table_init(node_t** hash_table){
    for(int i = 0; i < hash_table_size; ++i)
        hash_table[i] = NULL;
}

inline node_t* node_init(const int *key, double val){
    node_t* p_node = NULL;
    p_node = (node_t*)malloc(sizeof(node_t));
    for (int i = 0; i < hw; ++i)
        p_node->k[i] = key[i];
    p_node->v = val;
    p_node->p_n_node = NULL;
    return p_node;
}

inline bool compare_key(const int *a, const int *b){
    for (int i = 0; i < hw; ++i){
        if (a[i] != b[i])
            return false;
    }
    return true;
}

inline void register_hash(node_t** hash_table, const int *key, int hash, double val){
    if(hash_table[hash] == NULL){
        hash_table[hash] = node_init(key, val);
    } else {
        node_t *p_node = p_node = hash_table[hash];
        node_t *p_pre_node = NULL;
        p_pre_node = p_node;
        while(p_node != NULL){
            if(compare_key(key, p_node->k)){
                p_node->v = val;
                return;
            }
            p_pre_node = p_node;
            p_node = p_node->p_n_node;
        }
        p_pre_node->p_n_node = node_init(key, val);
    }
}

inline double get_val_hash(node_t** hash_table, const int *key, int hash){
    node_t *p_node = hash_table[hash];
    while(p_node != NULL){
        if(compare_key(key, p_node->k))
            return p_node->v;
        p_node = p_node->p_n_node;
    }
    return -inf;
}

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
    double pv;
    int n;
    bool pass;
    bool expanded;
    bool end;
};

struct mcts_param{
    mcts_node nodes[20000];
    node_t *prev_nodes[hash_table_size];
    int used_idx;
    double sqrt_arr[20000];
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
"8DB5PN7my7Z@5q<7a}7wz88*8I99!_8Hx8R{8G+7r*8E[8$@8Ds8ZI8d|8g;8T]8*58Cg8&i7xx8Uy8@!8R*8qN7l-8mK8TQ7iF8HL8^77K{8`[8SH7ma8|{8j{7O+8ZC8]C7Iu8VT8*[7n;8w{8{a7;'8H@8X17P_80j7]!8w98538J~87.86;8S97_g8F/8^Y8vx8?k7k787-8?!8,k7q|8AM7Jm8//8]V8:,82U8`B8YM8-387q9LD7Y;7xq9Gw83I7N#8wu7cF8Jc9K!82T7op9/q85<7Lz6:A7^/8Nj"
"9=l86W7sb7L;7Lg7_b7'p7[06{.59{7Be4LY7mD8#V89p8cJ8en8Ic8!r7rc8XG90q8GA81~8Os8Ny8=$8.r7`c8,W85H8,n81p8Uq8SX7PW86>8=z8'S7/o41D8(07I'7oZ8QI85K8EL8B18!K8r08Fj8R97zR8fX8,C8EY8XL8::8+q8U'8JM8HF8W67{z7t)8dW8e-8&y7lz7wx8|B8_?7nQ7{z8OT8{j8:)7p)7rs8p^8UZ7U_7e-8_>7hh9ig7lM88t8;l8.183v7qx8B:8i}8rH7tn8?_81p84480q"
"7gT80x8)x7gM8+C8?M84^8Hr80J8tn85o7wQ8?y8_L8O&8i98C+8DA8Ko8/M8kU8f$7:O7rY7xo7W#8S@8LC8u37QF8Vh88w8&g7~s7}R7tf8Li7mJ8d(8pb87d8#!8@K85]7l:7u+8DC87G8_}8cv8b,8eM8@l7Zz8?h8d)7Sa8U(8^~8C|8jw7KD7NY87K7vP8Ex8XQ8Wu8KP9-I7u{8K_8MD8/P7kK9.,8028N}8Oq7O@7VZ5$H7V_7#@7}77.L6oN5eI84=8?O83:80294t8UJ85U8DV8<C8Ri8x78Qw"
"8ua86g8Fo80R8Sz8T:8?$7v{8N~7df3&$8<;82B8BR8>>8_|8+88E08@k8L&8;R8H<86Y8Gq8D>8M<8<?8E^84^8$c8]U8fS8:&8<F8i$8D?8-@8.g8'_8XE8}j89Q88W7q<8U&8^)8:P8E78KP7~c7s/7rb8.]8QM8#j7^19&~7m27zK9@T8368(V8Ru7R{9'&8v08*e7}=8@A80#8'o8JD8PE7eo8n[8+27ue8='8Zz8J884V9*b7t]7}u7|Y8Pr7y@5p_78E7mh8Ij75Z7rC6P#7@w8I08VR8Kp83Y8dD"
"8nk87@7{l8R58,G7aL82J8[M5l)7fB7vr7L483|8Ak7Ry8$88g[84t9?995z7^U8+?8LH7sP8;N8l)80[74f94f7mB80P8bG8,I8,i8/:8hh8^{8jO8im8V68c/8418&g8!L8q982s8d*8h48>R8_H8*K6P_7@=6A&6Sl6yj7U46iV7l~7|`8&b83`8lg7{m8dK8IE8&H8EW8&k7yY8Y^8:t8jd8e#8Vu7re8H087}8J28lo7'T7^G8f`8#87gt8gQ7=#6{_7<'91m9(78}=81i7~e8Np7<Y7Ga7kS90B9&5"
"8p!8CJ8;s8`?7>n7sF7Xk9=S9?n8OH89Q80-8Q88SC8G68[m8R>7u17yT8GG8_~8G!8KL80$8>l8Fr8VY8J*8FG89T8KF8Rn6{28&g8:06IY6_38pU7U=7Wz9)S87(8[48c08Kr8?67Uj7gr8(f8{$87.8fE8pz8F=8O=7x~7_J7{|9'a8sC8pH8iF84[8=>7^)4IY7ln7Q|7{17/V8Q?7]@8#G9)g8GM8]:7hY7$l7:P8UX9_g86r7|O8D_8IG7|C7]h8YW8f.88D7z78+j86~8538!P7CZ7ev7ac6Nj7`P"
"7wz7D(8;?85!8h!7fj85X9628,W8$;8JT8P38K88^I85G8Ub7s-8Yc8<>8gU89[7s09;n7m^7tQ:7:82.7=38ZH7QB82R8(V7vt7yK9PE8,V7YB8ut7bH8;x7{W80F7oK5n+84G7b46?X7|Y8=h8[y8858/z8T(8Z58:x8NF8St8308Ur8<[8?f8ps8n>89J8r88@j6Gp6do6se6a}6FR7EO7yk6sY8</9-c8O;8Sj81s7R{8k!7ny8'(7^V8<M8QI8a&87D9D484)7|86pJ7|57}/8Nk8BF8Xs7|<8Tb7vm"
"6qD8:T7g;8V$8UK76b8WI8OK6xQ8qU96Y7jG8-b8hl7FX8_L8`17j,8P(8~L7{68R983^7@D8M@8W@7x18f*8q`8$r7/T7e[8jD8JN8d58Fz8A>8^T8+=6iB7]o9+=8;E8Hf8VF8KM8R17bJ8Pw7jH5f^8I,8ZL8;*8Vl8^#8^S8Jg8~>7lX7zc8OD8Fb8.B8c@8T38IU8TA87B8Ep8$C8Pz8+J8fe7p/8K78X_7X_7K{8&}8K}8]^8h&71m8LZ8k:8/O7}q8sd82.7Xj7X`81U8iU8x;84.8XA8>B8.f7|J"
"7ZF8+x8gK8vu88R8@^8d/8-I7W!8W.88Q6Ku8>l8Q78mz8om8>I8<n8Qk8,S8vk8@C8918`o8C-8<08(V8Q]82+6{j8Eh7?s7$R7fc8<E85}8ND8^F7o^8RH8+l7eb8j+8na8;b8g,8V28-28G(81d7WH9+m8DR86B8yd8XA8*:7|28)*8RJ7JN8:r7~=8q_8K=7:@8N`8H[7nV82z8.@7eu8>-7@y9,H8^d8Kz9=+8CW7_*8H87{u8AY7dZ7<f8T>8/S8*t7qf7hN8<(7t/6u,8D68GV5k/9c;8$+78D7|E"
"7r78eL76M8'D8vE9*@8)d8+e8bP8_N8)j9Jd7pb7<K81{8Vy7x{8iM7vy9$a7S97W38gg9KX8L>7&$8E*8D,6;q7cz82o5Y@8I;7q.8]$8Vz7H97P]7yI8K^9=`8FV88:8SG7:K8>64t(7#&8-v8_F9IG8MX8Oc99N4Vu8:$;0T5ju6ck7r@8M'8Rf8zA8TO7GA7S:83F6{v6il8Y$7m!8yF8BC8iV8,p95u80B8&b7fl7n17z#7A08XS7r~8/J8_T86>8=$9C78q{8>+8.V6e97za85$7O;7{O7:U7358vK"
"8'-8Bf6bw7_-7,k7}L8}87Rn8'27&K8Vh8AL5Pc8kS8nI7qK9)R8AG9248bN7bu8hY8>m8Q58Ym8xR72'7Qz6x+8@w9MS8|X7Wm9)+9+]9>S7uy8?88(@8E?7I@7XA7O,8)R9=v8:/89P97'84k6^v8pV95X7zU8678j*6d#7uh7m@9(E8xl8,68pq7t87WY8128BP8Z=7xy7497]m8CV7BC;+58ax9t(5?X8yL7B;8o*8A^8[99e28$p8@M85q7|E8J055<:4U8m88Sn8;|8s48b*8|*8@}8J486U8;A7{i"
"8zo8F,8O$8dc:2w8}X83!8W772d7(88U^7B.8(?7Z]9VV99Q7s37Aq8DE8B;9$d8Ea8:27vI8)*7Z'6?}9@L9Oo9)L8Y48Gf8[j8jo7u_7PR8dM8456u)7qj8e77A28E58t#9(?6J/8kX8iv8]$7`57QI8?|8'}9@/8#38Ep7sh7R;7v;8H?8?B9-]9&R8LB8F$9+29q07pI8578R>7JL7L#7Mw9/C8k,82|8k;86a7dP9-38mQ6t79|e7o=97x8Rd85/6~18Aq7ZQ7oq81m8W39/X8?[8k@8RD8!98fm92M"
"8Hw7]s8;i7`N8(x7n-7^c8s/7Zc8H{9|J8n$8XY8o+71n8FI8Ie6wD8Cy6i#7Yq8E*8Nw8_B7ck8LO7pO8<#9)y81V8?S8jp81&8iP7~V5wD8_Y8;:7/[8XN7n'9.[7PZ7Qa7/;7j{66.7{}9&$7vA8&l8<U8P|8'r8:/85E7i~8?v8j66bv8>[8+]7N(8w38>?7M$7b48lM6e99$m8Y(8+`8+o7RN7_*7n082[8Sm8Qd8^y7mV8`(6+e8a^8Ta7e#7v08$$8~f7i99#D9/F6ub8|K7Dk8CM8I97co5BO7EB"
"7hD6.37yb8`j8X|9T-75M8s684*7WK8'D8@A7ma8)^7xT8)<7A[8>Q8go7T(9wi7Nl78y8&=7LY9.s7XI8_Q8<)8SX91R8SB7q$9F79P08Mn96]7I?7Ye7z680k8m+88.8ya7fK8SI7xt8YJ8'|7I=8]28Dx7SP8!F7NO8:k8@a84y8fV8RZ6}{:.,8[g8Cb8]y8)P7l_8NB8Dz78.8'O8Q>8V[5cx8oi8mZ89y9N*7OW7P)8/b9-H8Sa8YN80[6U_88=7ue7UH8F<7958S796'9^a6YT8N|88X8@n88p8q="
"9+<8`)9SH8=28m}8xY8_K7y>7gU8,;7aK8.^7r+8c081W8FF8A$7/R7u=8E97Ds8=V8Sw87z8U~8(`7F_7[h99=7h38fp8SE8+E8=F7729r{9D28LT7h`8;i8Jh88z8$*9!H97y7q&8MG8]O8=+7d-8g~6bn9pg8fq9W)71j7}B72J8T'8-68:>7n#78o8o18;&8?<7l^7'I(-h7Ky9(U8y.9B+7Wl8al8Js7Z&8M$8-[8W58Wb8gn8/C9ts8d;95{8U18Sq87#86X9r)89X8Bh7Y/7X48=08R}8=G8697bb"
"7{T8o#8TO8@Y7|k7s*6Ta7SE7~M86C7ll8,283k6p:8!.8Rb8:*9tr7}T:YK9WW7Kq81b8~.8(;7rB8l`7g85e$82c6s?8#C8XX75k8CU8=599y9K=9;a80p7e_8e~9Q47Wd7jk8rh9.]8F&7R`8^}9-m8QA7g]8hc7~58Aq8c68Fs8<98c}8MQ7;k8Cx8EV5r]7sI9ke7Yz7j!6uz6Mv8M983@8od7me9!f7}p8P|8&o7jV5wp8G!8i{9L*8KZ6Hn7T59-E6-O9>O8BO7kk7]k8?R7Vm8>R7`'33(6c`7L/"
"8G)8Ya8T|7P#6-g6{E7vk6p?9jy7Ha9b;7Gx9*88478;r83)8sN8c<83`7c07p&8Vj9)n8rb7Rz8b|7Fb6Vf6dU8;<8IF7nC9^}9+D7$[8Ug9&d9=)8Q?7aS7#R8=:6c^7|C8=(8M^7ZT8.u7n'8$'9)}3i_8]j9wp7FA9(*7gr7RD7yO6~u8$P9&44d78P|8d99YZ8$d8ef9$t8kF9Xm8)$8)e8d,8hg8_*7};8l?7yx7]27c38B)7Bb8]b7cb84g7<06F.:yH93-8wl7@285~9bO7zO7$U8``8G<8Ip7[{"
"8NY7`u7Ql9+089H8Tc9$w8nn8328oh8`78)j8g&7p07H'8n#8>>8dG81+8-_8j?8P!9Sk8}$83|8aP7PH8Nd7y]74*9I(7f58{^8&h9V47~!8bW8.985e7]{8848:965S9767+V7f#8P-8nS7A_7M?7v.7Zr8;58Pq8TL8(37A'8^-8_s7dB8S87z,87c7x@8^V8P+8B!8Pg8PP8>`7qX8vQ8?d9)A8S57qy7i]7^,8Y38^t8?`8M=7gn8Uz8ap7jL82G8MS8!V82#84m9$78:#8;Y8*x8X68,Q7|c8')8JV"
"7uZ8b*8f_7bm8+[7n/8^)8Eo7mS8n)7qx8x687v7Zs8uJ8QD8DW8yB8YH9sP84&8I&8PS8cq8F^7^L5my8;|8*98Do8gU8Ed7p17vt7g@8VO9D&7fH7Wh8HA8r]8#d8[m8fG8@w8=H7pK8Oj8+<8BK7v58U27xb9#08Wx7bl8j$8^e7&j7nE7P?7q;9#(8Qv8Ho7Y17hp8`p8I^7|(7v97G789#9*m8*A8pU94N8)_8Fr9F/9.p8]57iz8XD80s8XP8O$87+8=e8jc8lF8&M8!F7jq5M17uo8/|9B@93|8]x"
"8bw8Tg8,#8L{8EP8O-90+88<8Jb8ig7xb9)27=g84`8'C7l~8EE8]A8+~8EB8`@87e8cF7Yl8(~8Qp7o37<'9+A84f7GJ7[&8d689h8F_8eH8`B8Gp8NA7xe7yc7z]8`:7z/8Z<8r|8CM7P08^]8te8BE94K8H!8IE8/a8IQ8D[8Jl8Lo8Bk8H(8bB8z484#9~B7Vy7j*7UU88#7RJ8I&8E.8Dm8}h89-7k;7kA7_56{N9918O_7r48wh8X(8!49B*87@8].8iC6E17qe9:L7tM8C=8+C89V86`8zj8.c8R}"
"8u^8E*81d8#~8c$8D~8s<8057:$8i08v;7Ib86a83e8IX8oL8Kw8L37mT93&8.Z90$7wV8Qs7=#7:97U*81m8W'7O#8bZ7;v7Hg7ch8!b8a@7~E8J*8aD8o:8Ki8oU8:v99&66G7cu8*/82C8oD91K8I^8l{83H7q78hK7}g8Kr7lw8ls9>)9//98m87<8<z8:&8@<8Ii7e:87)8&192Y7o68C&7w?8$z8'X8nj8+78Ub7u!91?7pH7p6:5W7nv9Va7,28(*8j>7Qb9*17yv8PO7qx7KZ7|z84M6R28':98:"
"8@]7{Z9:L9/|8V>8s$80o7im9:J6i@3+:7DO7ln8#!8Iq8HH87E7$~8Y>9Nr8|Y7y$7H]7.N8B>7>'7h58YK9/W8@P8Jp7y:8_f8'G8$18oq69r8Of9,P8488H<8ei8pm6k|8!:8@78Yz68I8f+87v8!F8}}8[27og92-83(9;k7ok8!_7lb9*h5S^8ed8K977k9J78z$8PP7^w8AW8567=,7D57NR7:x8G,8Hj8:88q.89_8oU8&Y9KR9Ib86t8H58Ky8_u7}Y7~_8R}8O~8o47Xo8WZ7oL8|N7T'7f58DM"
"8rx8]K8$Q89f8Y:8DB8*=7qm8OL80W7;j8+{8J{89^8dt7`i8B88gW8GQ8'L87i8-:7Sk8V_7Z}7XK8s37cI9SA8a*8KA8$99<G8{*9U[7=|4j88_H96d80H7DX87S7_<8508q)8`87Nr8YX8Mh7aB7:m96P7Vt8c-7t=82H7n!7QS8a,8uC84.8Qj8w479Y8?&5hb7qI8n77t>93*9KC6t,72J8xz7F~8B184|8?V7VC9dn8B]8N-8;/8LG8[T7xo7c=7cT8'B7vG69Q83:9S!7`-8`88hq8/=7t?8&o8`u"
"7d58a'7yx8FM8$+8f=8cD7[A8O!8gu8U78,a7au9(v81q7sq7bZ8f.8_w92B8b08A]7cV7dJ7E/8e>7Oz7|;8?68i-8P.7o67[-80{8J>6sO8:p:&=7u^8x:8Dw7y'7}X8_A8T)8X<8n_8JU8f$9&}8b;7e>7|S7X#7yu8Qi8eb9!$74h7.v8A28-N8=.7#p96<8$u8p-7Wx8FB7?L8P684}81Q7Zq8gQ8`M82(85j8bH8h`83L8VL89q86?8fP7~`7xK9Te8=:94{7NA8Kd8?B80r8/L6DC8zq95}8,?89|"
"8u670?8|X85V7]r8^X8Ed9-X8;@8D^8Su85W8=j8^08Us8:67v/8#c8+g83}8Vq8Km8Fd7ct8]28W-8538<88mM8/j7u86FO82J8ER7+]8(}97w79(8^h7Qx8_Y7OB8$u7k$8~`8/d8.B7R{8r18g=9'E8-v8d^7sM7P68Ae7pb7}z8J]8(N8[#8^+7~?8g.8z)6gk6Jx6an5fe7kW8JE7R/6_#8`o8o'8`n8Z[8cI8r_8KN7gX7X[8R*89:81U9ly9-,7z?7ts6V-8Z59vZ8nF7yT9+97ni8^18D_83r8fL"
"8nz8-L7X87z68407k;8D!7E/7]s8O(8S|8/J9kr8NG92m7i'8oh8658&O8eM8Av8qw8138QO8@R93a8E_8.*7wr84p8$s8T$89U7|98oh8u+8!?9!T7_w7u+9G)8ic85S8G}8J}8y37uq8@?8D_8V58L?8p479T8Jc8!<7s@89R7Uy8OU8x18r#8fr7tC8=67897B28__9#78</7~>7P48459+$8Yu8Go8HJ8XO8^&8>r8w[7k-7)#8/88Q&8}48M,86I8R]8Fm7i!8sG7R[8C68!68zX8B+8=675H8Em8SG"
"6y88nR7>,7d29+F94f8{Z9v(8/H8TR8bS8;/8c|8ta89H8Kf8KV8N47R38`34Rn7Bw8>z6r18i=7Sa9=[8RZ8;_8*J7s/7{#8JM8as7Ma8`F7Na85M9>F82Q8.U9@^6X^9(J7O58<=8;T7:78[y7yC9.76A;7nt84R7BY79L6^V9(?8G970F7Tm8{,7`g8yn7_+6t~8C~88981k8cV7q:7z'7l^8PU8,)81t8JT9YO8kz7n}8;'8,d8T&8`18vX8RU8)b6[W7y883q8yr8qo8U?7SS8GJ7-R7]78[*7oK8aP"
"8e478C7ox8A&8Bz80O8TP8z-8ho83181691#8108SK84c8dA8)H7{s83R8qE84T73n7TJ7k(8Q28]<88u8ms8<N8u~8i?7r.7$68sf8[t7?^7v?7728E$8`r74`5tj8^R7/N7K#8_x8Bs9T#80R8(<7zg8>48K(7|q5RX5R67}E7]v8*h7zb8@}8)v:A/7g*7H*67t9;x5kq7Ns91)60|8*S9OK7019T/9?G7p$8cB7{!8eU8c08Dl7)V:kL8C&88?7]<8.|:&]7g17n{7qe7G28768qF8x]7ir8a$7{G9!,"
"8Ug6@X8M=8(87!S8)47>+8PS8b,8cY8he9?}8F-3x38eC4z{6r#9So:si7AP7a<8i=99)7j37hu9UE7ii7Qo9~l8]s79f8U{8`/9EM87g8Qf7W}8<u8<e8,S7UY8N&8s(8/B81|8r.8v,7sK7W+8y38_N6y38n(6pR77f9U]8n<7y37hg8#E8/s8R68jN82K8F284O7u97s>7m!9Ek8z*8qZ8Y'7b17]B6WH:Hf8v<8NG8p78l08?q87!7l/7?,7o:8OW7*_7yR:MK91Y7nT88/7rQ7Zx7bZ8*R7^i8y69]7"
":Fz8(082L8e88T^8!67Qd84V8Fm8rO81'7sR8Lx86b8*E7_r8[^89r7r&8tV7O47d37l67RO8#h8D[7Yb9&u7jh7h#7wG8mw7W57jY8}(8bH7nV7Ie86179}7^z8f`7N[7::7XR8#g8_v81(8KX9^w84O8.77su6`56qJ8697`>8<H85F9(!8R$8]R8@P7bf8Iv8lw7N68&f8#c92b81;7u678<7hz7y^93398V8<&8MY84&8ZK92C8`t7`A7lM7dp8xn8NC8Mn7z`8Yn9$I7}a8&Y8:(8]i8)^8pq83k8XN"
"8GQ8}_8Xu9EQ8b78t48F58}n8;(7t$8Sk6|U9a;9K-7xh7^>8NV84c9,:8z68J^9K}8b[8bN9@E7{+82j9(=6lQ8.!7uN8Dt8?A7M}7p=85.8v'5z(9O#8Al8Wu9[{8xT7c86cl7r67<<73D8~R8[K9[>7k88zD8jY6z97T!8b/8,h7uo8Ck9:f8F;8kv8aK84'7<96^v7+j7z<8:o78!7_V8G=7:!6ji8q,8zP5uv8~P7{~88I8&Z7Fd9CO8K>9&b9=e7Pp9D]7U#8(67xn8P/8OJ8?$8mb80|8,885V7iF"
"4V]84T8Q<8c88)X8Fq9*q7R{8Wv78:8>v87j9Vr8+p82(9'r89/8Is8#P8TC9Rl7kh8+W82$6R38D{8#U6i/4Dw8c{8)C:.^8-j8*v87z8=z9I?8'a8E17,$8=h8c>7V16Ku8Mo6Zg7ot8qL8iC7BM7}s8V58?+8Se87E84V7q}8958=c8CM9!)8dL8Ai8Ro9T<8m08*w7f`8;/6y,7nQ8J@7Zn6KH8rL8*z8n78;N8)@6~A94b8><7JT8Uj7kU7Y~8)A8C=8bg7QW83D8@$8-k7TJ6['8`07p=7jg8iF9J:"
"8)H8x~8>X8c47v`8T#8[s7ut8cd8&B7BW8NR84I8=.8*e8XO9]P89y7u(7.'71C8RW8b|8B*8Uo8?C8=*7u/8|>8aF8/M8GR8.C8=<7]:7zI8ac7;#7Yf8C28G~7fL70r7d'8(i8=V8P!7^X9#x8!c8pu8zs7UZ8k-8(<8,P84t7t.7?y7=o8qh8-{8.N99*7f@7w)8Dq8<K8o>8ww7AY8Yx9M?8B08V[8;l80d90E8R77K;8Mn7yy9sz8A`8^q8P`7p>7sH8N67Xg6zY8G@8Z'8&;8Jo8BP8,D76w8H87pF"
"9Im8>-8<o8VJ6rS:*P7*97L98hz8t386;9(58&77Rq86H9L'7*?7gN8867F+9P87~88?[7?u87#8/y8)w7np8oq8HV8Lk7s68;m8[*7[@9.'9UN7w*7qB7wi8Lt7;r7LZ8_>85n8q_9Ub8;78NN8+B9'!87z8Ip8ke92X8T59@/8BB85.8.o4xJ74)8IR8N,8N|8GK96;7at8&@8EG7h48t)8_38R?9#;8#Y89j8#{7Ph7n-7r-8Vc7Q,8Ym9IB6*B8+I7nS5o)6K88oC8/J7Eb8km6~H7lT8Q'8Py8Wb8`&"
"8A}7b#7P07dK7qc7Wh8B18*.8A@86I8Eg8b=7P-8-$7pn8MF8az8<28{X8ZE8)L8@^7zy8Yz7~`8oe9:|8bf7VK8**7[16u18Mw8(C8m_7_B7zq92G8L/8+a8J[7tg9A(9!u8?J3U;94j86[9>38b#8=*9&u7{z7s18or8PG8Vw7c)8QX7jS5~D7^(7~e6aJ8AG9.'7]h8(08NI8i:7iC8$s7n:8R`8b!7A87b&7HZ8cI8E5:}e6Hr90Z8~d6Wn8'K8D&7j~8@/7ly9-L8.m9sD6TQ9j@6pt6>'8qe7Oe3Z?"
"9TU8BW8Bm8/68LW6K78P08if8/M7Hp8.k7vM8[Q8a[7V58dk8N48/;8>f7kp7zs8_y8&B7a'8>/84?8I'7l@9-X6s46w'8<h83I9^'8;'7Mp8P78b^8Q+8Hj9(.7)[6l>8P|8_88gc8:u75g8z^8~U8PZ7^_89u82{8c;8rW9OY8D48Q49X88DH8Z67oI8Wp8U?8vT8=<7RG8JU8<I7c-83f7W{8s99+05Ax9_L8oI2mc8qZ8-_6qh8Qn8A_8>n7Rg8?{8Ie7w(8a}8&b9f'82Y9';:NY7dY9?Q8B88L!7)2"
"7?B5g~8;K83<5i497D8!m8rR7yg8EF8FB7no8,(84g9Qz8?z8R'8u:8mT8;H7lE9)#7se7ji9-W:UF8qC8-g7JF8sx8P&3O38.x8Q}9.X8vr9O$7fC7sK8I]9?f7!d7{P7vW7_o8T08?&8n>52^8]18?W4<t8UM7][8yo9ux8sB7pO8Sw8Da8d.9GK8cD8Y>4RR8ap8cC74q7ZU7tu8is7|38$k7hf5wJ7dj7!A7;&8`X8b68_Q6CR6eg4Is7k+6U18vH83>6[N86^8S,7`'7=!7q{89K:.S6O^8Rx8<*7[7"
"8FW7}p83J9&;8P.8Ty8{~8Ml7io8O}6[K7Vp7e:6_u8W`8s58iA96O8v17{F7>|70D7gJ8E87zb8df90A96B8Hv7:A5g78HA6jS4Ee5Wa73$8Pv8-r7]29^#84;8Ac81N8iD82u89Y8h57]o7fh8mR8rn8;'5zK7,K8Bk6Im7Jv82y8YH8=z85a7?38/`8O?7~*6r:8F{85}8}V7m(8q!7549&N8;W88/7~)7f}7}18&B7qw8(J98.86?8tU93h4_I7k!6_/8J(8ZB7M~8P~8z07L|8:v8KA8IJ7x>9PB8m8"
"7?B8.W81/8JB8P.8ih7D68W(7qu8gA8SJ7|_8D#81X8^#8*78nB8nb7n;82P7q67~18h_7qs81R8)x8Sn8;)7X18J47>R9/H9+<8.o8W27Xv83;8-#8N(7Iu8zC7@.8Mz8`J8>-95}8_,8KC8wt8Ek8R:8P,7=u9-88#f88u8/u7Ve8mO7ys9:|8Da8&l8d$8Sa9*D8d18R=8K*7yg8<U8,Y8$C8tX8Dr7Vu9^_8Rh7_[8[e8X(9-*9@[5.=8F`80O8cF7qa8@l7217r27v*7vU7ox8M^7ks9608A&9$x8F]"
"7b;9E(8*-8!X8SQ98h8ii8?I8bO6|k8Od8I,6g&8dI8e:55@8I!8_B72N6ll7mK8bP9=]8Ye8p57bb7Ed7rW8px8H/8bJ8]g7/F8ai9uR8/[8?78vy8j<8D^86o82}8a)8/W9;P7Y;5_q7vb7S37)&8dn:3Z7Iy8d07hi8^_7s@8}:8Ub8_08y-5C@81L8B255.8IO9,#8fC8Hu72W7lx9078^780a8,g8&m8L]80R77v84=8$57r$8*U8dA8TV8Kn9fY8aP7C48yX8i)8[]8>[8(B9dJ8FJ8:M8s=7}W7fP"
"8V379y8oC8Rj8_!8Ci9-U9,>8>,8Vl7C77C98oT7&18Fr8fh8`)9<=6NI8'u8DU6b67Yz7+x8W57|V77/89&8m:8oy8]78`b8rK8Ex8gl7}97rG7S/7fB8^v8bO8gI8(@6/,7U*88I8j|9p:8SU7l28K78p.94E7Wp8<o8b{7/!83-8Xx9=O7FC7R_7+<8t#7248tX8w76z67U[9/&8@b7Wj8BJ9B38&:9tT6^A9VW7n66<Y8/78<h6Cs9,]7dt9,k7m=7~N8n27rS8z+7@,8JJ7EM8!J8Ve8DK7g!9#v8Rs"
"73v6va8C37bm9&?6{d6mf7Ec8M$7q+7Pq87(8K`8>n8B^7]#8MB8(m8*U7J]7XI9bH7t]8y_7l)8A&96l8D27Z=7^17},8s28=E8q48Z68;j8{;8_B7Wv7j07wM8787@p7V$8w>8/q8+c8L$8al8fO7q;9-^7j684891^7D&9Po8^L8la8r98;!7bE8157w&8!.8dt8JH8j*7ID8fV8(67'X6;98@;8YC8&H6HR8q&8!i5,w7EE89785=9pT7?n8vx7j(9`m8qT6S`:8n8?/7L`7SQ8S681n8b#8px7;P7zL"
"80V9!$8z25~I8Vh7qo9UD7db7hh7dc8}j8k:8/U94862M8<V8G480s88U8$K8gj8oq7v48|K8AM9mI7Zt6v76SI8&U9$_84z98P8Q&90,5BT8UZ7MF9ft6I:8ZC7Lb9)<7Uo8<-9CI7996St87r:1V8?:6GB8Fg8D76xR7gF8)>8)w8=!8816cH9@D8}[8s{8C$8lh9Mz7W}9)888i6y{7_)8Y487s8578:n5[T7Rp8Xw7Z38c/8t-8q881D9$c8:86mT8xu7y17,=8^=8G78!X9m373.9a~83|7J37#t8G["
"7ub7wV7o'7gV8Oc8eK9XE8*>7SU6`.87B8<w6}z9$e7`s6YX8{'8<S8gW89$7~`8'S7{/8]68;w7{u8Jz7i{9TG8:=8@I8dV8p}7dT7[B8.493893w8En8~^8#!8.36Z17o48(47]?7E}8Za8;181;8.c7bX8J!8[Z8Y[9LR7{|8*G7h/7N;7Z'3|<8?>6ck8^$8A98hu8N]8:<7c87GU9<-8:Z8rW7Qh8|u8e/6~'8.p8!(8Lt8P<89?8.}8VV8O$8iU7ww8SM8}27xS8I07s?8J*8hx9$d9/w7#@8>D8;1"
"8cz8?I7Tq8,q8Lm7m@7II8;G8Ow6j78Io8rd8@n8hl7~.7f,8,Z92+8X58Js93k8o38ne9lT7yi8p`7~U7aI8M;8le7>y7B{85K7SG8:18|=8oU8{x8->7mJ8Wc7K<7<.8]B6p36xc8m}96r7j:9D,8AS8`Z8Oo8C|7gP8T&5i?6+v88~8(2:gp8dV90V8m.8@67b;8sN9)X8?P7yy8578$B7o&7]X7PF93t85Y7B37l.8BD8;h7~37d[7E78uf8oM7Y_7=37S371:7'K8s29u.7iD8Uh9,>7~B8+*8ig7@]"
"8*{9*M4l29<V7e}8#<8?h7Nt8L*8Sc8n,7L(7=(6=^88w7Ei6<;6qq9~e7Gp9E=9P_8^C7K}9a]8>i7|V8{>8`c7`@8Q08sZ8uD9#y8SM7Mz8*47e&7iV8jR6^g7]-8,78Dk8)M7Z_7/+7lL8Pl8rg7q#8I-87X7E/7U#6qB9F)8N38{m9+!8O|7`S78`8:07Vk75}9:b9op6U#9r^7O88`B8QB8;d70v8Bu7m+7IW8C!7rl8ev8^S82?81K8I]7v[7@88KS8'e8J>8CR8tA8}J:1m9m?9g_97!8@!8DJ7lT"
"7)C8f28Jq8dK8Jc8ZK7un7kK8rf8:i8'O81Q8'o7q37?J8M`8.f8aX50&78;:Ms9?w7U@9Q,8P;7|,8J!7pT7q98d_8uK8.I7si8|58Oz8kz88~8V67={6|q8P799i8ca8mf83.7|$7wI8Js8Ts8Uj7L(78n9IQ8;F7Aq8S#8!L7Q680f7,G7fy8V|7TN76H8W*8Jk8?88{s83R8~q9*S7Pu8_'8B18}.77~7~-7ci7eb7nm7Vm8BT8x38m^9f>8h98+:8<58_u6xu7qZ8H,9G67rs8aF9LH8I|86)8tV9$f"
"9dg7eP7w}7#B7@~8L$8N-8(n7X+8!(8_o7v67},8S*70K8xL6we81!7u28vU7u98X^8V/8OG8dh8gF8Z686b8GP7877]k7oy8We8Cm7;D7|v7kr7|)7sA8s'8N*7vT7gp7~m9(k7[;8Qg8~`8gn7W>8g26zg8UF7`n8'78=m70#9Lh6$)8@x8838x?8:t75D8FP8sf9?,9>B8*Z89d7uP7306z|8z28>c8c(7v#9948vD7o>8sJ9.J4P&9-w7v>6I@8Sj8Bu7C.7ej7uv7[48i=7IX9aS95^8fk8$j8R88o'"
"7A280g7[F8|*8UL8FJ8@d7Qi7Se7~U7ag8ID8H-8b~8;97]!8Or84+6{z9#98S#7k/8B-8={7'78ZT8B96Dq8p_8707dk7BP8Sk8lq7~680g8j/8,}83188B7p,9<p8wp9,990?7q*8xP8He8A[8!s7t&6Sj8Pl8Jc9mo8og7}07[57s=8C)7}R87!8`N7tx8hx8gD8!@7fJ8J46v]99f8'a8^B6y&7rA8Gj8t;8I:8Z*8k`8eM8E.7{j8K<8D28Lh92_7@m7YB8dL8>78DX8g=7qB8:688,8'E8xW82F7ML"
"6ys8t08fA81M8;37k:8uR8]58+v7d=83m9k!8$?7}E9158;V7D&8F/8ye8b18Fl8@?8G,81D8cV9/t7ta9K[8v{8<(8FG7BO8JY8=E9?>8We81f64{8@v8`59Ok7.<8sK8B77'E78e7ol9^.7y57u&7a.8ED8EL8<77mt8b'9L~8Y)8?^8]998z9&w7y<8Rs8*_81T8[Z8+`7^b8G~6ey7u*9OQ9hB8<L6o&8^t8?q7WR7|o8B;8J~7e}84_76~61@8,H9&/8.78QB:R(9/G7~k7uf7R!8n07wb8-T7zi8@X"
"8e~8Z58gD8eW8fF7[/9Vo8E)7Zf7S18:c9Rp8E(8`'7:z7RU8G;6:48N-8ii7n~7X]8ww9-:8X@8$C8,}89p8Ke7d{8x=8{$7Lj93M93J8Zq8KS7w>8IH:'58YH7f]7oU8fP7F38ZV6y_8s[8>S8|77E;7X+84_8tc7r'7sB8V&9*E7Dp6Xj6To8ZC8JC8FK8w>9(W9k.9!;8e#8C`7|&8GH8LM6o58Jg8Vp8LA6Q-8mk7V67)'7&Y8:r9+e7wC8s]8vK9G{87886}7e&7uk6fQ9+77Hp8aV8(B7hh7iP8TV"
"8:V8cd7q<8m~7iT6re9[-7]P7uw8tz6x]8s[7wZ8i]7{G7P=5wS9997,?82,9-a9:095$8U;8a@9)f89x8(y9(c8'M8r;7M[8x;8HH8gs98c7bI8s=7i#7Gr8hN9mJ7p480T5{-8n{8@r6y`8fi7$s6Kw8^t7Iu7=w9i97V]9$;8A=8<:82.8TI8C.8H-7pz8#L7y*8y(8Bf7aY8.d88j99'7BN7@U8`A8gb8>38VB8C+7?n8On8kn9&e8T97/b7W{8a#7Dv7yq8.399W7xf8*B8n=95O7So7pR7<c86'6gc"
"65O8=X6rb5&q80J7608xK96s84H8SV8y,8h!9Hc8R>8!j8e28Oi7Zx8t@8Mp6Hw6{C7nM7o28dQ9!i7)G8}w98C7Tn8P67e~97u6n{8G=8j]8#p8?F3#y8M691y5E<7#$7gO8Xg7kj91/7.&8RF8)Y77<8rR8=$7Pc7^]9UJ89u6`~8&=6nW8~|8w27}y9Qt7x`7bp9W37gp8p>7T'8Bu6|c8Ua8bv6Zd7gN9'^6x486-7xG8+(7l:8p;8/h7AZ8Ku7A!8!q8Be84+8j}7}J8)D83y8pZ5yx7tJ7ep84E8H&"
"8DC7DK8GV8WN81s86D8?&7ha7v]9/u9M/8l28;m57g6WI6e^7:k7yr7Wu8XR8;X9E99:!85b7sH8hq8g`74S2NH7Vr8Go7_?9+=8|K8C}8998Gk8E?8D*7yL8=q9JO7}_72S8Gx85s6]t8=$8`<8:G8eO9Bs7H+:_U7P74^l8aD8Ui7?m8hg8sM8C>7}S8Z-7H|7y488t8.@7[J8rz87?6wY7~)7$o8$99#n8VF8Kn8Ox8Qn7,;8w#9g+7E;7K(7.y7J]8')7LT7=y8*E4=!6Qv8d'9&d5fm8`p8+_8k`8D("
"7>N6Ns8-47SY8109WT8<=96U98+:=Q80W96k9oq8{E8w-8)e8>u8Jv8F38[r8QY7k;9&+7|}8Tb9-t8`w8;O6tj9968u-7YZ7jt7x)8:d8qp9*/8EE8$~8_[7R'8=r8T09xD8^,:qV8mg8z@7z?8bK8Be8Lq84C7Tl4CX8067t$7+(8Ne7o68&F8p78=Y99j8#W7lg81A8pu8&Y8S58X38(87r>7[d8~i83v8Zo8'68&[8m<7j@8vH8EL9<A9)J8ty9Q+7ob8KH9[28E.8^~7l)8@{7|h8Us8+!75o:!g7>["
"7D28#[8MG9+K8M28?L8RA8?f8!;8cS80i9;y7n=8xN8nN7jb8hY7t97tV7VO7Hz7xb7wg8p*8ow7K/8N'7Nh7.|8)P8Ec8-K8S27jb88w8(b8ew6{d8;|8($7_?8@C7u=8;a9')8D.75;8a:8XL7-I8Gu8Mz7BN8ZN7W/4(:9-f8?$8Nu8x:8E'9(g7X:8:Z95F8BG8:@7k(8^-8?G88f8)T8&_7Px8Or8<*8B}8J687(8778Xb8@37c^7Z/8pP8tR8I=80~8V^8v>7~m7f88Mw9?g7g-81l8SQ7aw7g,8/i"
"89/5~=8[$8?G8N*8F68t-7c(9H-6Ey7sE7s;8S?8RU8V96|b7/^7:E8Eq8A}8..8mv8d?8RT7xD9(H7EO6Nd8s(8-=7L;8T&8Cz7zo:<*90N66a96O9TK93w8!L8z!7:?70n8|67rn8KC7V67g+8.)8T=8jY7*a7tU80s7ft9#|7Mk8:'8wd7pT8I|8k{7CN8h388x8<$8d-8(s85a7gp84H8Qf85S95U8?[7yL8Z^8=a86O8j?8i&7au8;68}>7sk8@D7ef7DC8I|8)n84W7kf7wO8]?8-a7wd8$&8<b8T2"
"8VK8Jb8Vn8ak7a17zB6w38nR5wN8_B8H.7<n8Y:7Ba7Xx8Gq7[37tM8>W8z;8+28dJ8M-88Z9Lt7YE6Nc7z87cH7UL8qN8RQ7E;8Zi8EK4TM84`8y;6#;7vE8-S7R-80g8'E8,>7]W7y&8Jx7sQ7;58dg8QP8^86;:9CY8w^8BV8c,7x`7998-|8lM9+Y92|8)a8[;7c78f@8rW7BK8Yv7=o5S@8Wi8fe3;Y7:f82(87}6XY8|S8aI7>l8fe8=D8Jg8x18Ie6Q=9[d:dd8D27=J8c/7xw7wd7g)7.J7{28*k"
"8<C7y@9,99<w8Nk8`z8q67LP7(H8I08c;9}18-O8rt78,7U58$s7w?9P{8S67j<7&y8E08~08m88:47;y7Nz7hH7p$7u883X8,Z81=8=#8e~9#@9OL7e*8q97,J6-]8_T8:'7GQ8]=8B(9AD8}i9Nj8<68[/8w;8>b65m7u{8]l8Nq7sd8=X7JT8@_8'q6m_6<t7a*88a8@l8l.8m<8Yx7Q46a48C&7kb9;u81v8(R8hj8[l8Wt7mg7hg8Ny8=>99.7f_7rS6Z^7Sv6]|:2]9YQ7QF9={8BN8@f8n:5v-8gX"
"9wI7z+8fr7k#8DG7y.7zz9F37SX6r=72N8Uo8y/8'47KW8U+8]d8LU90s8`y7bX8D'90y7^`7i16b]7t69Q(:1O9.i9@h7@#9`n8*i8H<8f19Ya6QL7=687y8kN8]88DP7BY9/@8Jp6zp8596f{79H8i*8K.8l=7PY7hn9+U8W99Us8PM7l=8>+7-=7e)8Nj8ox8q'8@G8<z9GG8,P8aw6Na8c06k]8V)7^x8L58#I8/y8zk8H~7k05qO76c6mU8(W7ll7W<7nW7cR9&o8*t9rD8V07Ry8s!7Zk8zy83y6v,"
"83S9dV8`[96/8Mf75i71<7T+8(W8FQ8_C8qe7~Z8YF8bG7q@8hC8Ji6F^8:I8Yz8P27II9&b8'c7i-8;{93!84#8=W7[?86E8/l92:8&z7Q47rf8)w8=;8iz7wS8W{8Vy85$8S27wH8xb9/s8aA7uk9(Y7xH9'C8LN8XB7YC8K06oe8,.8hZ8Ln8o;7}27BO7w88CQ8&S9@c8b,96S5t&7zs7WR7)P8g?8Ez9B^7XU7|`8~o8:M8.,8AR7n?8~J8cF7~^8.39j975B8!]9bO8#H8T>9(S8S/8/]8C>80|9C."
"87*8Z58_f8(^8LL8OO8-97|78nY8d`7jK8n77Pw8J*8(`8?Z9#_8&v81[7Y780<7'=9Bp8eU7o285z72?82`8Al8=(9Z+8[]7x68&f6`v7g)8;s94P8H!8+b7ro8[=81.9K38*h88<7?Q87/8'v85!8?d7bs8H}7}t8P98g/8a~81:8[c7lM7Qb7V68SN7`m83P7xH8On8^O8&=8+37dD8`X8Y_91g7~38>u8dv8Ra8Dc7v#8<#8Co8Mb9/z8eI8_(8g)8EH8cX8)G8l)83R9Wt9YJ7jE6:3:x28s$82K7<K"
"8>(8?}7Yu7QP8<W8RW8?'8Ie7~a8,)9EL8@S8yC8=k87t7l482n7_v9G#7rU7GC85E8Cz6p<7L=7HE8;C7T,9Ju8@v8)#7{&7m.8d~7qp8MZ81p8d+7}^7v27U38>]8}38M!8C&7t77|Y8SA8Ak8ev89J8B.8iN8N=9)T7Ez9Ml7z26rP7n,7~d8iJ8H38WV6nr8>z5r,8lY7,37!$8=p6hr7ww8Az6K=8`+88s5f]8lR8>r8Ic8W591+83k7PR8t<80c8iA7oV7{R8l28m87]M7~M81d7A|8<R7}/8?(8]-"
"7Q]8Q:8Vb7/<8h[7k!8?F75z8qL7v06z/8j79FO8]#8R[8So8C|8)C7u06k?7fZ8e|85i90=8x18<i8Ic7|~8'L8Cb7v47Xe8E88<98cc8M$8SG7&^9^F7|/8k47~*8|-7X,82`9k=9&n7s]9.R8668x(8`d8Q,9$]8=K8+H9/A8Ab8yv7hr8@C9K/7Qb8Vh7Vq6xB8d?8948rI8Yc8b48QQ85-9J|7Tx8-$8<h8dT7f/9LM8Zl9`Y8<p8uN8G26MH8d+8oh7C!8XK7VY8H?8|h7-49Dl8TX7P-9068307I>"
":8:8nm7f'8g,4:17Gs8D27j|8:28tW8qd8F|87d86|8#J95f8BW8Pk7IU7}=8;}8.c7HE9(Q8)z9:;6eM7m69b'7f57`@7q&6jP79g7!{7O/8au8xM9DB7b[7Ke8E*7S`8067XD7dx7]!7Xt8A68408gR8{M7l@84,7!08LO9.{8L>7[L8N>8*&8>h98`8X$80s6xC7y:7jU8>t8Bz8uQ8h56zo76z7k!8P`:Qh7KS8Rc8Dg7eh8`37g{8=;8pA8W88C=8},8?A8.s8M:6S*9*_7^588]7dI8,*8m#9H(8Ez"
"7lC9W:8vd7ok8'=9HR5nl7h+8}O8PE8P(7}:8I78Rh7dn7q/8?H7u]7~e8]78cw7Hs9e/8h^7J^8sG8[097>8&}8U28a98k.6x|9#U7pK8-E7H+8=G8E_8mD8}t7{d8[G7NE7}W9+h7'68f09+n8='7/?7Tm7w.:.(8P#8Z;9wH8Qk7817q18Zk8*T8D$8_k9&;8lZ7`K8]T7vY8#`7/O8yx9D(7p[9+d8tX8QC6jh9BB7yG7`k7887m>8oe7:57ZP8$S8~#7t77T#8>S87b6]w8,+8,L8aR8}<4Hv9&{88&"
"81C7*k7#g8dY8#i7m+8/58>a82y8xJ8v,7sB7FH7A98/n8l08'281L8UK70s7u88GH8,q7Tw8k78!W9:V8,+7Kn72)8P.9$|7WQ:&j8V.7SU7Ha8hn8H{7MM7J$8Yo7CJ97Z7XU7KG4mg7*f8j192+84U7pw7k)8VU7yb8qr9-57sE9>:8C18kF7sU8fF8I77DD8<78t?8]m8&X7UL8+H8p?8D:6jh7yC7GD9!B8Kx7uf86i8!*9DJ8hg7356{]7wM8jF8l492>7v~7E26:l87N7Z@7('8W38[T82^7|J8$3"
"8At7^q8wy8X08K57x280i8+K7ZX8TD8lh8#h8vS87288L8=C7xv8I?7Nh6[/8{*6,v8d~8:m8=[9&@8IP8.>8,c83k7YY8_f7k68jZ8]z8WN7<^8/689X9$j8.28bB8UE7p38*L8d483R84k8![9y}8T*8^Q8nL8][7hH70X6Pc80;8bd8QV8L'8hF7`S8Py84T8fw8AC7K58LT84e8+>8{x73A8Hk8N@9,68TP8HP7!p9h/8Ie8w}8@d8ln8Le8:,8-+7~A8n48qA8VC8L&9;S6tQ8cU8386rg8/79+C7.H"
"7k!81~8O{9(&8P=6lJ7Yu8sb8Jq8Df7Z*7U<9#U8ae8aR8:|8UD77:7vD9.99)L9:G9Z18g88T_8O/8Hr8Ce8@}7V57Q=8iM9Fy7cF8[07E$8L,93N7KI7{U7[Y8i_8ut89_2PF8'(8a_8?27zF7l@8tX8*k8(y8Z$7MT7Yi8OJ8ap8wq8SS8>_89~77g94Y7q69>48G(8,&8N>8'd8lO82Z8-S9!r8/~95C8+b7xN7u^80T76N8}>8G*8}!7wR8Rm8p87_^7uk6yw7SE6f>9)u7`d8`K8];8|^8Hs7/!8^y"
"8JU9:79&f7wj7v&8G)8n{7{O8'_9ql8!88K:8:o75Y8UE9#=8_.8DP8'L7T389c7em8zf86D88L8qw8hp8kg7ZR89'9OM87|8bu8/N8CG84!8}w8#-7kV8V57Uk8XN:/j8;(8'694|8JP7r'8'p8AI8l]7nw7|T8mj8uu8t`8Z.7f|8^j8LG7v,8Im7&=8S$9<&7N'8cG8&c8S28E27?G7zG8p(8Ld8;t8)d83O9Sf8>(8k=8;(7Y<7^p8m_8[o7WM8=^9#q8U@7v(8C19:j8Mg8L+7kE7i{7Kx8{w7?l7mQ"
"6b&5ha8=h8068LE8Dn8lO7Od8u}7,;7.i8q{7v|8E=8uu8Pm8-y7xM7th7+w8eW82I8/#8~.8);8Ee7Ji6~E7eJ8Qu8aR7No8~T8{V8[K7Z,8Q;8$Z8nn7an89#8K&8R28?Y9>)6t,7[i8fa7Zu8Kt7ir9I&87{7)g9HV8gm7_+7X:8Ff9Wj98!8?$6mA8SN8l?8Ap7Tz8?R8N28_~9$B7h#7DA8?18ed8q$6|@8K-8I.6py8r(6SI6lV8>>8M98Y=8ch8GB7(-7Sa7zl7Dm8D)8f}7Pj9Bs9mh90u7ky8aS"
"97982D7V;8S58498:$8rB8?$8<x5N<8Ru94k8f,7777mb8_:8`Z7^e8+G8IY8Jl8ZF8]H8+88-t8m38F#8lD8S17;~8&n8!<9B29$78_T6zW8[A9*O8)i8]o8ll9?G8u^8K677F8&!8`=8Ib8]{7]K8)w9B27pk6CE5tI8WM8n]89Y7If8X19-28gS8'18TR8118S682s8eH8.q8<b6tH7A;7.B8-g8f)8g+9oO8@87zX8sY8*|8Ew8/*7gZ8TV8z?8+B8(b8P:6C`8)C8L{7wt8`]8|Z6>:7f~7#(4{M7L<"
"81397W7de7Y49668'68'a8IN7pz5Ei97K83p7Q@8n582t3A77Z~8*K8ct7YE8is7B`8Mk8_;7u$7Cj83m7#`8#X8Hj88M8hv8]w7Lh84y8-$8+$74087A8dm84{7c,8{!8d+8C#88F7^i9'G6{]8:T8157x#8=B7EE9/Q8^m8h164?87Y8(q7SP8CV8Ty8:O7ou89!7qq8Xc8LN8_O7vz8Y#8I&8v+8I*8xf9<P7Jn7;^8gb9+f87W8sV8q^7hE8ZL8gC7X^8{57J]8D:8kz93=8<E8L}7sb7z58-T8[B75l"
"7_j8gO6Qx8fg8;!7;k8dV8$D6js9878qv6[b7B]9J>6Yt7b#7217:-8WN7gs7_r8:m8K=7{o8Sw8t'8($8E,7Y:8:j8/c9+r80o8N57[B8q,8|!7uz9=F7v,8gw8kA8PH7kN5`g8Ky82h9>u7Vp8BV7Cs9FS7]a7|w8b'8N|7^P8BJ8ck8iE81/7wY79U8EI8Ue71>9Bv8R38!48j|8gV7Zd8*}8l(8AM8`h8E28H98@f8.87|29$>8a]5B!8oT9:O7{H7w#8E88^88C~8G^8~H7i:9Bh8KI8>g9'C6hE7yi"
"8H?7i985h7lx8OE8@.9!<98!9#]7zW9(K8J68Lv8cm7n=7U-8_88(j8'87~R8Jf8!o79'8?'8+R7|~8P881H8|.8/28$o6c_7wd8&@9dE8;=8,E:;49;I72u9^E8FQ7|@7AP8SN8jj8EB7o&9RB8F<80m7.u8;_8s-8&V8BN8CK8FQ8Y38i?8lo8JF:Lr93E8T:8}B7t[7R)83O7'c8--8p07~?7QW8(W9/g8'!8Y/7e!7Y98.+8K*7U48Zt8N@7QK7CI8g490Y76C7z?7mn87^8cF7<P8<g9+(7U$8TD8:s"
"9_$7c37BG6q66y,82A93#8dp6WJ8Nz:3L83m8K*7SO8,C8]N:!F9Q38RK9K.7Uo8G<9DH8.z78;96E9#-3o|8l<9Jy7SQ95f7WO89p9*p9X[7}*7Dm8gh7lu8N87c76|)9Qk7k=8uy9<l8F28U97x18WG7x?9)c8U<8C17{Q8O@8O~7|i9177G&8e~8NV7(<7ea7{d8k88R67y'6w|6s!8/78&+7kF8fQ7hf8N_8fR7nb8Xu8sa70X8[A7tn8DQ89$7xt8oa8!/7A=8qk6c<8#08+87}V87x6gl8~W7Jh7/w"
"7|58qT8FV8(m8!J7w~8#c9y78+P7'G8g>84w8EJ8(48IZ7'M8QB6RW6wG6Qi9*V7L07ez95F8A|7]k7oV7L$9Z88HW7ti8,h7by9$#9@Y8*z9;n9>D8Fy7037k^8$38ZU7=<9UF9L!:Lm71_7X$6~-4vK8Jb6K)9*g88L7yF7s8:!u9]:7rS8Mx8ST8|j9S&8&78!T7_b7x,6p+6uj9Tx8N&7C[8J~8Wg8227_p6wD7w,6Y|8,A7D*82]7vw8$25Wy9]z8FR9O?8=!7_h8}g7~28M?8Ow7JB8+M7XZ8:<8u8"
"8JB8?N7eL7#u7zK8C*6GY9Lv9Go7}P87}7Z&8=F8TH7,q8/D7n}85k9,48c'8TU8,I83^:El7S,7a>8cA8iG8bB9:p8i=6`?79j8&(7798u}9/L8#>5/]9<,8Ca7Y98Pe9AO7~/6RJ7Cy7)q7}`9CC87K:3k8PC8-n8Zl8/T8;-9*?8I[8Jl9/y92'76S7!z7pU8CX8(R8{Q7z77,|8LJ8Wz82v7tg73i86G8?(8v88qw7v=7eq82S7^]8HM9*$63N73:8PO7xs7w.9*X77i8#C8'75yt8@Z81D7>48oQ8F["
"7w~7B18b|7w>8hx9'n6d[6OO7c38-+7nW7Qw8nw81e8nZ8sU7~g7?G8vD6sn8057q{7ct94#7gx8&K8_,78a7j^8>k7pu8S(8DR7]'92|91i7N[7J/7=q8[|8Rj8yq8!`8A97ux8'*7Z(9)98DX8km8jB87#8(Z6r$8b37v27r#8.887;7<s7dy8]V9@|9T-8#B8r,7sK9.Y8Ip7zG8Y=8{F7548aK7z]7yr8S_8ik7kF9->7Z]7lj7Xv7eI8M|9.=8M(8>n8>07G;8,W8)K8K}7AJ8}{8Vk7?=8dE7pp6U`"
"8DG8dk8}Q8,V91~5~,8Bn7eN6vg7w|8RA8*g7`B7ZO6uf8Xt9;D7ga9B47PO7[}7Iu7KS6fs7Yp8+C87U6CL7at72$8`:7u(7q#7#y9j29q87|L8F`8G-7NH8[N7mM7fB8w87Y_8k>7jM83F8B08oz8Xt8u}8E06nb5lS8W[7}|6]*9156]X7R39Cu7b*9M^8tJ9FQ7sH8eX8[28Qj9,B8vW7kF8$~8sy95@8Xl8A89&;8g>7Td8NJ8@j7S-7F27lE6{z7<98d~8j:7[-8]y94E8yr93Z7zG8jn8KX73C8Ix"
"7@~8lC7?.95x8`089488d7t(8P,8nz8Ev8cK93q7[+8@w7a@8t;7p{8/_8rH9AJ9!s9,(8K<8yj9(u7/98T`7t-8jx8!N8_K89u9!`8[K8E*8g98@-7P29Gl7l]9$D8ie7PO8dm9Q!87#8|98Ni9:^7{p71]7jW7B,8V{7WV8b18,06sI7S$8)88QW85>7ux8(H9'C8Y!8TX70x8k<9-g8R'78p6s)7i-8);84_9Pw7iX84g7Ls8z08R29&y8j*7nP9_Y8]e8;,8IM8Lu8[F7}K9*|87A8:A7k_7^-8d-82^"
"8Z|76!7g#8.o9m18!(7cA8&H8Kh7{?7h.8.y7n_8y&7gV6@[65d7m07cR8H?8Xz9&/8?i8N39(.:/r7XQ8Cb8dG8;y7J(70w9_@6g{8,|7AP7du9HH8{a8+z8.y8.08IK9'F7~R7mE8(G9-48P68/+8ew8N78.|9'.80@7yI8$384F72p8Mi7g68'+8`C8,V8M>8Va8)w9208r88C(7(^5w88qA7zj8>P8af7tA7nV89)7j?7gV8QY84A8Ti8$:7|P8eY8o_8X^9#X9&I:<Z9o88t:7)}6'#84d7=Z6F?7]T"
"8t175W84)8#^8,u7JO7Wn8)37x-8T,94h83}6o45b#7a(7xf4PD8LE7tR8bN7Ah8a@8ZQ8RS82v8yX7(>8r88:I8aq8<^7I!6!58/w7ss8H=8oK8m77z57wp8Cp9)J8,N5Cj9;b95@92i96$81M8?I7vw8IG8Rw8GG8se92l8:n8q:8H^8bN9Nt8(U8377Gn8*`6L>7`Y7@I8V773K8{}8^09_]8FO8a`7#l8=e7pa7e26Mg8hC8j<89U7C#8s[7I38WL79x8'&9<w6NL7$|7o>9J[8t882*7pa8m37S487_"
"9qp8pr8a#6LY74+8nG:km7ei8(A8Gl7iC7k79Mt73=6rD9!}96E7Z98Ex7sW8<a8Sk8LK8LH91;8t$8z~8I`8L(8@88-s7Qu8Fj7tC8Rp8~U8O)8;W9&v7~e9X68@u70m7xP7{+8P@7b'8z!8]v7rg6$88y*9jb6}B90'8_p7EY8YZ8C18cc7tu8VL7jM8/*8-A8oF7|:7627~^7kG8]y7_?92A8DB8Um7eb9#D7y>82786{8jg8$17i;7h?8M79Rf8&J9qx81#96H8L.72I8mR8a=7wQ8wF7w^8ru7ly857"
"78N8R(8e/8s~7<J7jv:0Q7A48w99.F8Tj8Q@8.b7n.8CM7)K7db8;q8'68_Y8a>7|h8y*8k,8:t89'9Fd9)D7^d7nv7|,8vR7t^8Jp9P{8Y48GU95v6q{8FR8^U8YU8.m7?R7mr8er6s[9>k7qQ79y7]b7eD8-78x&8CS7jH8F67zd9A]7bT7aQ9?38dT82(6Iz7M:8,w8$592^7p|6jh7Rt7;+8#a8!j8R49Ax8J~9967oq8;h7qw7YX7ut8e@7+@8AR8uY9!-6^^81)95H8-;81k8$m7zJ8Ir8GN7{U5&U"
"79086x7[t6U+8'Y7uc81F8a)9Ba7d|7q>7sr7U/8$#8:[8LK7|i7S78^H7HB71i7~J6dw7rG7CH8K37Sa9Pr7d48B,8Ej8U(7UE6}|9lZ8ug6q>7z37nH7l#9+,8i`8.l8Cl9&)8p=9PU7To8ED7g>8:17L699U9.E9$!96X8d=8=$7V]7'U8k26S<8_m9[)8Nx8dM8z{7~X8Aw8:26Jd8IM89l7le8}k5jG8Ri88U7O{9_b7QF82)94Q8:t9^D8~X7Ni84g9VC8/m7v78JH9$G7WQ8UU8q?7t`8qW8s|99c"
"9Mp7{37d?7BV7`$8e!8cg8A}7yS7m*8$V7'!8Cy7l~7H@9B;8D!8m,9RC8u18CI8d&7zT8=57p<7u/9J;:!j6hg6sF7`R81$7Sz8D#8a:7?~8}h7W79;a8.081.7A38V57wl7!X94w7ST7n*7SG7yA9<U7EX6*>8w39*y:448L/7~v7:<7ge7y#6qg8O'8My8]>:(s8:)5og96m7t-8tn9qi8+28#38'K7g$7c$7Yu7uY8R^8@18V(8NU9-m8^[9'984D7NP9:h8a[86a72w7r^7CV9]?8@78.o6lb8Z!88J"
"7#u8Ok85m7,D84K7zp4JX7Vd8>78#>7z27{G8^M7^W7A}8=D8tR9W~7R^7NF8a]7GI8w#81V9ES93Q8$'8i?8R,8.+5197bX8^K75D7`591f8=68h68697~z8628/{8+.7v78m]8#&8&18@S8(i8qB8z(9kH7I;7z$9XW83I81&8i]8q?8227iw7cH7U68@;8F37+T9-'3lP8_|8*J6n28:O9;c8p!7p$8ho7@u9B)7Sy4TN:/`6W#7Vi9!]7d/7Q46Hv7fF7T.8?c8(.8]`86]7c.8Dm89`8u`6sq7YZ8[`"
"8!i9cU9'N8VW7D'8`w86$8Yr9cV9Df8Q)8DC8$U8Ny7O'71r8KI8o49R,8?z92(8rY9,R9#|8ap8A08aU6t^9):8Uf6~[80=94K7^(8IO81y8#Z88}85W7F=8dw72#70$8|O7ri8&T8h)8Z]5x&:;_7{88*=8Gk8k28Ru8.,8Il7$(88|8;`8'b7|~7{h87A7u&9Wp8=g84Z8LE9WK8w_7p{8|W7Wt6kf8Qr9r_7`49TP7pD8bW7b'8=S8Ni75X7L-7]N8kB:&v7[~8>V9(&6&-8;|7r69BN7`D8+x9:98:y"
"8$z8*98/_7o68VF9Db84t7Y47j_8';7(98;C8SO7GX9R=7G<7(.98w7H57&y8K.8/g6a}8mc7@o91b7sx8KK8nG7B09^G8eT6{76Zn8Zf6zX7<&86f8v.6[N8#Q8~Q8P38?[7~m8$y8[X8#E82q8N}7I>8Hv8#(8>G9@b8l!7,x8?k8KO7K<7n:8Rp6SD7Z07OL8K<6O776Y8sC8f|7pD7~S6E28;e7j@8d48~^6{W87n8=;8WS8f?6wI8&y8Qg8`g9!'8A77#K68^8hP8c<7v#82_7j(7G&70o8I}8,'8@)"
"8F]8l29[98jo7us8S;7`O9F@8V588$7pk83m8#g7uP9T(8rV9'~9-X8ku9+z8^096.4a}8bl8Wz98x7N'8^08>o8p-7X`7v&6&}9'd8t#7oz9(j84L8i18Z[6vX8KI8S[8l58528Qo8gk7sk8qN9*u7F,9,>8I;:6W8?B7zP74@8SN97D7y48*/7_x8}t9td7{]8rL8Kq4FW8L28@D8ui81<9DB7qR7)C7NH6(s7Fa7|e8j:9&T7sJ96_6eL96r8Jl7r27Tc8sM9479s=7b=7M]8y87pd7#B;!T6rY9c59*w"
"8iv8md8D^5m]9?h8.@8QH7z~9/S7~.9'l83l7y>8zh9B,8A@7+`8g#8Dm9-<7z!8e)8&c71@7ub8Fc74i9T?8Jh8!-55k7b!9)t9CO8|^90w80u7B68O)7k_83'7i28Mr7H18f{8dw87R82'5ph5tH7&o7mz7(h7Wt:R98RM8K'7q69/^7q38[!6#57^,7AY92R7n(8S@8-A7iN88@9}k8'|8hP7]b8!G8n57&I8cY8+s9LN8lA89Z8FH9UI7f^80+8#H9$I7v<9[E95^8SP8]W7[166i7p.7xQ8Du8ZB8tO"
"8pa8|u6ti7xz7`u8Be8[t8_z7P67g|7gg9PS8bT8(v80G9[l8b@7w*91m8(18/~7>B8#v8ad7pP8ae7xN8ue7|l7lK8Q;5]h7vG7nv8|+83G8Pd9L?7[c8/o9d08Vz8:(5]e9QZ6a183V8bj6V_8E98P;8047v48fc7eN7[b7yf8|*8cX6~h8ce8kT8Y*7xG89p9#78*Q91Z8:p8S.8r27h>8:[9D79-U7a`8u89(J8&78rl7|+8?V8/u7]M7p^8Uw7P.7F,8vL8AS9(W7t;9?'7BN8YG9$l8xs7yg8[Y8=|"
"9)-8=j7tY8fs8o$8~H8'Y8TL7sG8_Z8c05'?9=]9MB6XU8M98*<88j8Z28598y87xc7{w8Rx8zi8QF7@R6Jo85D8Al9,n8D]8(_8D_8[J8yZ8h<8TW8Y&8F@7m;7W<8Th8Og7mE8<e6lh6r'8lX71b7tE91'8!e7r$80e9KW7hs84991W8d+86{7g#8FO51e:Wl6958]Y7}27vh8{S9=U88~8QU8Z_8Wy8U-8M{9-F8]?8zv8:+8ho7cO8BY8e?7CX9FJ9h98,R7sv6~A9WI5w>8J-8]!7cR8qL98I7Da8`/"
"8QN8508bh85^8l[8[(8-j8UY8Lj8$l8w67UW6bn8?R8Q#7xa8+w8@79L!7sJ7h|7lA7B`9Tc70Z7Zv79/8f&8:*7-+7v/5^'60y7<M83(8*,7$P7`88Sl5w)8.?6DT8VB8ec8n+6GH7zs8p17rk8[[7=)8W?9!66v!7R18mN8No7Cy8TN7nz7bc7qJ5s/7H@9+O7r,8_c8Ac8iC7RJ8>]8:97I{8@]8'.7oT8yi7QL6NP8?$8Jy82?8(S8[.7Op89A8Rw7?W7l99.G83)7R(7.#7]w7ul8t29)T8B|8Le7Q+"
"9U;7Lo7Y'8`^8[f7[58Tu8W]8d(8U[7wN7=p9XC9+F8<B9?T9CY8'V7j09)=8:x8q(87;7:V6'68T>8H[8~W8zs8*;7d'7d79I39m=8'j7~28e]9,F5Mp7=D7~n8R_80e8Eu8((8v|8D99*I81)8<u8qd7=T8Yl7x`8Ca8(D7m27Vr7d|6DT7Q|6lb8.g6~47YM8vg8?47~<7IM7vb7E!6!n9UD9Z-6p>8uD8hN9dA6~D7Fx6:E8qI8O$7h*7m28eW8=A7q{7bu7'A7^R8Jo8w+8s!7CA9'Q9;t6/z73q6en"
"8n78|b8d>8ul9)i7Ym8E,5tJ9e18GB7m_7lL8268bB8=!87K9<?8JF7tq8r19(&8tg9$V6oN8/z7e:8:q8V/6pa7329'77Q+9+s9:,86E6q&9=68Ds69)8,q8X/8EJ8Po8!p84L8o?8sD9HH7r;89}8ci7f@80k7a99@c7uG8'-7OB9>88VU8E?8=W85(8$f8Au8mX9BX8Pl7zv8E>9<,7~y;$C7m_5_x8])7Cf7wP97E8J,8u`8767qJ8[D80v8H#8yq8`e84B8Pa7I=8FZ8YJ8H08mK6ju7M)7a=7R79;P"
"7`l87L8Gj8/^8Hg9/59/[9`T6wM8uT8h]7zX5J>4O/8&b7j26c@7M+:]X8F08G^7My7h,9M[9+>8dV8$i89l7a{9AY8|S81f7P@8Uj85E89W8'v7V*8(L7ck:&p7u(7jq7F-8@d8ZL9^f7c<8CG7eI7|N71k7zr8}58El7B!7yg75x7@u8)f7h08-(7Sc8Do83t9,V7AU8ky8918*@90l7]079v8fK8Rb8Wf8j:9BT8^e8ne7iP7$p8l(8H18@s8T$8Mf8237gh8Ub87i8@&8I>8:G8t=8sG6Z]8*Y8`#8L*"
"8Pi7kT9.46I+61)8vr6Uc7<,7Kr7TP83~8l#8O!7|W7/#99a9Zf7c:8{]8^98?[72+8nj85:9/^8Go8Y(7_Q8lZ8-88MX71'9tC8}q8G.7zE7GO8Z;7Ui8(Q88#8ZU8K:8,I8l}8.e7ZV8Q.8_H9d!7,*7]n7yx8GN8?y8Y687j8P>7~<7?A8498?f7y'9)m7g[8!F99)85n83?8(}8IQ9Ey6c!8UZ8J=9RT7s088W7z,8)S8Q,81O7[q7)g9#[8u<63i8g!8<A8YU8E68;]7PB8Ax7hS8O57a989^7vr85$"
"8G96$C8C)8}o82m9Fp8<o7V{8k.6gY8`D8|v7y[8:P8{97V28?q9Yp9Q=8U$8(B8|o6-k8_x7[~9+28=b8^B8*h8757f56p`7#.9L^9-~8Kr8Y+8MM6:H7M|73;5!D8o,6a59)S8HB8787G*9w@9Y)8cR8EC87,9`P87`7jK8I}8?p7w(8pO7Gk9Pk88.8tb8*G9z69!J77/8A@8zQ9/[9(C7lU71{9[[7qY8lo8>m8.A84(7Rq85w8l56nY8ou7AQ8G38E)8K{8dP7z@7Dp8R18d?8*@8Y{7gi7nK7v'86j"
"9C[9;?99L4p_8}k7ud9k;7h|8EG6nA9P!9?q6dC5vG7{H8QC6wm9q=7c.7vi8^S7g_8ki7Ww9V^7rf7>*98d8M58QE79|8Ux9>28mU7}t7d08V48|e8Yn8Hv8I;7z38jD8*.8nc8fN9^,96y7f|9Et7Kg8uM7qV7;&7|98mU8b?8UY8U780N7};8H88)g8CA8ak7>S3JG66P7}E8;{8TP8Pg7Qs86<70b8n@9#87]_7r[8~q8,i8@R7:j7588H;9#,9966YO8>@80_6|68+K99m9pA7Z46Bu8&^7}{7n78&9"
"8CA9+F6kE8iJ8@k9NX7z;8A'7lM7ai7H)9RY7_|7}B7?E8*^7Dc8('7He5k#7$Z6{67fv2gx7s48Nv8NR86>63(7?p8>{70s7e[7vg8Xk8V48RF7Ng8)c7w/8O/3ho9c(8tj9QC87X8&=7hg7mS7oF:#X8Qu88$8m:8?~7a:7Z[7]<9!,9Nf8s)7OP7KC8Lf8[*9d+8im9>L8<D5}R7cI8=y9t#8^L9$D7Yl6}j8fK7dZ9BP6Oi8cj8,68OW8+(:(36xj9<Q89x8S98r<9fm7&@7_17{I6Ia8I:8O09$o7Ab"
"8d[8m!8Dj95l7.$6|*8/?8d'8'-5ss8='9HC5wu7d=7Gd7|B7}u8ai9rM8nU7^J6w|95L8Go8(a8-=9*P8s'8Ay8(88:d8~66hO7ed7C&7A&7{+5~89184p=86T85r8x55**7xt7;m6u,7qt4i+5c77~H9F18)u8@o75X9uw8rR6Ua8Ve7.'6d*5n78<@7,w9.R9(37W45m~7Zb9LQ5;*62e8-s8IF8g$8M;9LI8607F28f|8ar94X8E*8Ao3M(7{'8<V9Ry8vv7GA7Xu73@7j>6#T8B`7*Z8Sq7w87wj9p7"
"8V=8Yy8UB8m|7>V7jr8:u8/e86>8'.8=$8Ye9#?8qc75q7c>8}'9&[8xp8M4744:RJ8.@6V66)l7;V8d48Tx9T#88T8Fu6h[7B19-78ss8kd7.X83U6d$7}Z8+V9d]9!67g07Dr5~r8z&85>7}O7F}83+8uN93w9h18Fi8Xv9&S8l^7`l73b85}8G(8qZ8^^8}'4q/80`8:87lD9(#8BY8Z+8#28^?8pN7iv9`j8}88S>7fd8#C8<;8B=8*C8Os8IU9*^9`g9~-6nc8nS8M$8qZ9)28fB81H6;77rW7s&8B7"
"7?o7so6zt7DU8;;8ra8cM8R,6oE6z*8f)5ug7eH8v!7p17U.8Z]73f83O8/.8$z9Ro7.t9/|6g58nG8ZP7w.8:_7+}92G8uT7qn7oO8##8:{7S/9!`7g48HM7u;6:u1tC8YL6q}:?384d8;Q7f@8l'7eu79>8W97/-5|i8bc7SR8#X80b8XZ9IO9'+7XS9Za6DY8}d9:28l389d6<B7Jf9$t6k;8<@7f!7@K8rj8G87R'87L7Fd7[48+48fy6~:8#O8XK5].8(k7A!6VF8{v79P5z[8dF86)8|!7f[8T06a1"
"8nM88-6Vn8lU8Wp84g8U+7(F7`V9$06N;6d58;(8oa8O08>L81g7i~7wp8U=8Ab7r-8S,8Wi8o18`~:gC9;?9bY8y@7w68d!7ng8.^6N}61w7_K8/@7^u8IC8l+7kc7dr8787]Y8b^8+H:Wt8~497e8N18Zw8v<8d<8e~8,Y83c8/57|y83z7{H7w#8N?83$8zU84W54/7&x2bR7^f8:}7Lj65f91f6hE8Lp9!Y6cJ75L8B':4w8a68,G5=a8ne8/y6MI7y78(w36O7ia8W^4=)9Ao8a08GO8k&7tV7r#8c("
"7=P6E_7KZ7>P7Gb8~>80j7Hp8(o8U*4[C8:^8RK8My8_08Yz6w]8488wf9F/8c@9>a8!k8]&8|x8p^8C{9/=9cO8+d7fj9&L8(17~44-g83J7Vm9fg8lO7~z4N38Ex6(r3P08@'7&R7>W7nH8tc7cm6|V8{W8,B7#(9FA6py8P-8]n5XZ8P^8QL7?;8Kb8aQ7y68d_5Oa7i)90X8.J8mc6(*8:d8ji8TK8Ed5kC7Rd6zz8)C9bU8rG4nj5OR5G:6RN8-06cZ7W!8m$7ze8DO8T(7bX6YP8Tp7pq6YO8=`8r#"
"8Mm6Jb6)L7EG7m&9#c8s@7z)8Vw7)o7p@9D*8Z67za7o'8E&8Zx8uE8sX9G;6_$6XV7|'7oV7J+8'i9#]8S*83e8a'6H390_80T7M68ka9.O7od7&28@h66j8bc8j)8E,8@48qP8?t7Y(7z(9n88zR7m98gn8>C8L}8Rg:O88d<8b{8Jv7u?8588?$85?7qb8<A6zo88'8OQ89h8D(8:[8IV8)X7Hb8D}8j~8^f83{8I?:.l8GG7Xl84j8;A7~L7lr8N684X8=h8>?8Mr9|b7i9:)v8}n8.=8LJ7K/8&x79>"
"8H>8U.7v(8Kd8^K8_*82C5}!8fT8Gr5P`7M[7=37f>:>58D!5cm6P;5s!8$$5~s8R18n*8yU8LK8Og8188Jw6nd8L@89&8<i8-080i8(e8xh5RT6v98*m7mU9*L7&582|8;l8Fc8Dd9`M<2=63^6ZY7uy8JA88w7E35SZ6@P7A#7/n8vY4|~7H(7kJ9$^3v{9pO8XR5m88!|8QN7Cu9-n8A+6Cp7f$8]_8dl7;J7+`8Kg8;18fQ:/B8o28LE7O594=8Gv8L-89|8}+7/j9!38608h:8Q87k[7vc87*9FZ7|A"
"8dG8^Q8G)8Ud87l81b6gj8Go7|{5c@7KI8YW7x:7Dw8U,8]o8`59+@9x@9!68S<8YP7)68Y?81/8)b6$_8G49aR7}37x18N38Aj8}z8H]6{!7`n87#8{E7H25Ap8-g8<X7!Y8wm6n167T8+R7v~7AR8^,7]s9G:5.H85x7&H8l*7nd8LU7Sg8i(84_8XH6/W5'P7b|88Q7wr7ov6M.78$7qc8.88T|8/272S82t8;p83;8S(8gX8&G8T28@E7{:7xR8o?8G.7Sc5xc6OK7Ca7~|8o]8+q80#8/z8.d7uW7`!"
"8J>6>_7vU8:Z9$m8)L8-j81G7>L5h28ay7}n99w8;n9I87X17x08GC8P{7gF8eN7~27FC8.i7jr7lZ8z[7HR82i7LK8J$8Pa8SR9_#8Nd88J8c[8aP8Og8)H7XE8uY87U8[h8p~7w&7Xs7l.8zb8'18z(7rX7588ue8Gp8Rd7rh8()8oX7Mc8_t4sz4!X3`+8Xa8mU7m27!<8cA9$h7ML6-.7D!8Vh8$s7qX8kH9-a7b099j8,R7<88jO8&K8D&8]H8A,8Pl0)k1aD8#s7MV6M$84D9H'8d+8>+6D18F69P["
"8Je8*|7oK8F}93J81:8S784A8lf82F9/P77/8H]7Bl8Ye8mq8M[8#n83V83B8=c8X^8=y85W8R87u^8_p8q?8IY8J:6bu7sI8$h6jP6x;75w9cn4K87}}1RM5v?8S^7Wp8`982W8K*9-28K]8:N8Gf8;G7Xi4&B7=78>+8=O8Ym7DM8l&8O;6qB8GR8@'88$8+-7SO7nL8Bl8W58K?8Kr8&)8P!8588xZ7^=8Xv8':8PO8p_7rc8Se8U,7GX90u8AK8.-8k-7wD8?C9a(85B8wz8f)87R8x|9=h8ge8$G7mx"
"7nd9DD8Sl7<47rz8&~8Yo8Tk8Qv8R37_i83Q8#i7sg8[=8,S85x7eH8g?8128b77qq8L+8AF8c88iN7yL8Af8G`6B+7G@7[/6Ug7gc9za9428H98{E7p:8wU8tO8Qk8l|8(V88P8@&8:38&I8FW8|>7hh8Ot87}8!K6i88tg6w19-P96g8(O8U78PQ8G96w78`68^J7a_6!r7zo8Y+8Fb8UT7#K6|H8:s81R7q,7S!8Q+8W$9G06x,7:58446|J7Fp9@J9?i9ho7R_8_N8#y8~H7h*8KV84C8o|8Ff8y18/#"
"7sG8kD7mZ99`8188cM5![8gd8G-8C`8O77~O7N+8&.8$q7`L8p@8&z7SN8Ue73m6L?8:a8wp8;*7|i80d7V]7g|9'|7~m8<w93?7y-74192#8S(8is8EB8be7yq8S98EV8w:8!18:S7p{7`)87m7q;2Xs1:c7dP7a34El8Ao6Z.8So7nM8s17S88g77v?8kM8F77u+8)o7}?6IG78:8jh8YT7XN8_Q8<C4lA7gr8mB8>77r87~p8l>8q|9(w8J,7l-9=h8r<7gZ8^87O{8Y(9Qy6G68IH8S^9/K7Lu7o67j4"
"7+i7Hn7YE8/S9+,8).8`#7eZ8&,8@J7Us7UU8}98'#8Jo83>83x75`80b:k:8DN9[c8Qe8a?5d39Cb6M=63}7TW67N5NZ7&a3rY5BD7=]8uf:KW7<e7[p:5$8#[8Fb85l79_8wX8A=8WJ8Bc8C78C,7Y{7^p8Ny7a:8J<7{T8[k8Ts9h08>W8ZG8Pq7Eu8GT8Zz:*78bO88e8,H9#a7sE8`38<k85)9>d8O}8K@7>D8E^7yo7&`7Z^9A08a-9117oi8{68AY8m27TL8aR8L;9,T7x68b`6n(9IB7z}8eK8P{"
"7X78A08)U8wQ7Ch8#^8>q8S}8(c7w_7b=8L}8Mi8P:7|c7YE8jM7ga7~P8T^8Y39QY8!g8Sb8P,80$8c08y#8VM8*X7JW7?u8sH7XG99|6ea9**8N$8YA8hy9HG8No8(m8bA9cp6bD7QP7G<8|F7qm7)E8xC8T@6yo7]57=38Kg6)H8v]7N[8Vk8^M7dS8K*7zi9=G7w}8:m7_~7X]7jI8'98>v8JE8?q9538A~8)G9E'8O+85l7H;8LG8SX7yX7{87rz7xj8^c7jB74e84/86m8Bs7mL9RZ8G<7G/8u!8j9"
"8k!8>}7hc8vZ6i}8f97v$8'@8eZ7k59C/8D<8B(7Ml8Hq8K<52F8]@7}88e=8_J8xt8-u8_[8eR7+X7^#8=:8p98&o8Qg7nS7lz7w.8&P7hA7Nm:4v8sE8WY88E7xD8v(9!p8[k8|Y7wm8,;81d7F>8`G7{z7MA8:r8vs7I*7eL8GB8-P8>88:r6{M8F;8TH5nJ1/(1,:6s'3:j4.i7q.3*74Xz87N8JW8WL8}H8J[7~~7T>8Eb6pV6#Q6h_7gl6qa8Cy8H08kD7g'8YC82-8Ue8py8e:8)v9KE8Z$8w*9-B"
"8J`7Uh9)X9,m8}@8b,:*@8Nf7Z(;{t6*k;as5cA2ka7Td7G18$Y8;.7u!8~S8<U7W~8dJ8;l79m7t*8W~8]}8+:6gJ8FV8c{7y]6xE8*/8wv8RG7Q]8KF8?Y8#D9(E9EQ8E;7j380]9Hp8)g9JT8$77?(8eG8qF7z-8Di6re8;f9)A9I08<$8Z88{O9#M75S6iX8MW8k~8nC8ju6eO:&P8&Y9>u9D,89{7I69bv7q*96T7$I9el8vz8X47W37#r8-r8Qi78o7m~8Ar7N]8NV7Yd8}H7)$8ig4f58+>9>i8IE"
"8o88O$8G^9-=8/R7Z&8:,97g8^58k/7.>8l.7(&7]z8^^8l:9)L95~7wz6Py9/@7nM8N97yj8OG8Rf9nz9KQ9k_79V83G;9O8=K87D9gO8@k8:P9]-88>7Hw6m284<7r87u187D8=P7cy8fa8Fd7d-7C?8SM6a(7cg7-o56/8fn6z04vu81e7#`7ch7?77hd5Yj8).6pq9Qo9>T8fh7?+56#8I22en8g:8q-8'09hA8T=8#W8QT8?d7er9-V83a9&j85D7sJ7@I89R7Uu8bv8e47@(7d'7]*6f`6~28U]84x"
"8Qh8A&8[j8+d8[Y8Sg96D8E+7Zn6f99+[85B9g[89{7x|7ig8-{8;G7RD8:489o8.*8Oi7ZJ7~{8-E8:(8@^8G}5S)7dy8OB88_5yZ8~e85#7*n8g=8)X6D#8O{8$q7W78b(7T@7O289+8$57H,8SV7s38YA8p}8GF8ik7)t6nz8eW9&H8ec9108XG8]e9)48I'8`d8S{7p476e8:-7WN8](9Kn86N85d8Xa7t-7ST6S'8^Y7QY9Cq8|f8FI7j&8GX7K-8(M8f/7yk8OC8lf8DJ7tT8__9[[8@;8ja7Cu7eZ"
"7_u8&`6+68*M7ek6r^6+66b^4;^7N+6z_7es6|<7K88bP7dw6eL5yR8SL8md5RV7Sa7t)96#7BJ8LL8047:q8*a8v)8Aw8|c6Xr8rz7{o9):8.{8?c9I#8NT8${8oB8+G8)=8We9+/8|m7^78Z#7[u81q8KF85W8*}8n58SK6m@85:95)9_I7RL8n48Ps9_~8qS9N@7by7&78sT8I-8bO9.l8->7j08`>7N78dZ8W/89o82t8Cr8Nt7jl7Z(8Ci8S+8378*p97h78)7r78V39sA8#o86.9/98pb7L(7{x8kU"
"7fR8<:7cp7*V8JG8^s7H48iC:<A8$i8Rb8<08578I_7iL9<S9>17I18ZA:I+7y=8cT7T;8&<6?u7O~85v7&?5r17r871i8-78cG7Z|7ZI7QU71U7U680<9p>4;K9IA7m{7s_7NW7U29'^85D7_p92^:5v88Y6vZ8;B6tz7M~8C$83@6|?9|W7l^7~q8u+7UV7qB82K7t+8?D8JC8'./k58&&8gh7Oz87`6|Y7nP8X/8#]8RK7I#79M8(:6hx7@X7tw8d&8448Wa8@;8G_7~L8/r7kl7B.7[(7vn7vL6Q68aG"
"8a~9-k8{:8BN87d8<d8PO7^D7o48]X8Ab7kn7w*8)h9,$45]7b37f598.8/N7Un6=L8YX9(78-D8!Q8Sf8KE8K68.[8`h8:a8P?8TI87D6zf6Y}6vT7.k:Vy8DW7Ri9H[8VX8=u8Wd9WR85m8X,7bY7vD8G77*48A<:#L80f93e5rb7]D9/s89]8._8:v6k<7<08Uc7EX68h9dC85(7H'8DN8r#8dQ8_@9E98NB7!,8c08r,8ce7Vf9S18DZ7qg6n?8G,8:{4mi6106PA8D*7W$8Ym7}G7$[7WH9MJ78K9,I"
"2D[6y/6lw7+k8#z6XZ9<E9UM8j!8ME8CN7EY86H8x>9RR7hJ8A/8?L8O,8RJ7[08^=8(p9~o9Kp8cD9'R8_q6EK7Q>8rv7s>7r:8at8|O8KG8[27K78X^84:9I~7mr8>i7pM8Xw9CU8r18DM96T7Ea8iR9,w8+A7iK9*#9*^9,'8{)8lm9*_8Of9X?7_`8@N7Z185t8C~7ci7F48BB7Vm7bc89H7x27N|7|`9BC8(A7q+7s/8KU7jE85<8k29Bb7h>82`8LC8T]8:<8+T7II8-A7`~9WQ7W-7SM83i9EU8bL"
"85w78(6oZ:#?:X'8>q8@o89T9BK8db7v&7b:8?$81[7^{8Rj8DV8$g8A#8>t7sN81o8Q*7j&8<A8Aj8I-8688Ps9nl7?:7|o5jv7hL8PE8-x8/08!i8PF71}7T)87~8N-8#V8lB8HJ8Gg8Fn82R8Iy8:f8;g81~8`s8<A9)*85x9(Z8.;7qg7*b8'P6sw8.)85.7yV8WJ8*G6tY7T07um6Wk8{b9+V8q=7s;8P<9.98&'8gV8z47tG7y)8=q8m57oi9!+88X8@h8;c8Zs7|j8qn8=,7kT6nJ8Ag8U.7:q82w"
"8O;9D78>e8L$8~q8-78><7Gf8,k8pO8@w88P8Z(7}L7tZ7Za8:^88Q8F)7Oh6tz8n@75>8,B8aY8wA9'.9'y7.U5S+5PB5VI50a2hg6Py6F<5A28527s`8kR89/7k*8Nj9:X8>&7U;8Z#8$h8S48lM7^#9-d8,V7d+9~N6X38k87fy8Wo8FA7gX9eS91l8EQ81{8RI8Lo8)R9&Z7y|9O,86:8oF7Wo:E=:(E2!D6lR5ry4dg6uo5?)8*J8nm7168Uo7xu6h{8oj7Z<7{m8Oe7{'6kI7R48Bg70x8bQ6_b8^!"
"7w~8=k8XY8If9#e75280O7pa7sB8:07Ot8Vs8{V6x88a*8uh8yL8-k8l48@}8'K8JC9$c7nF7r@7XH72b9J!8KS8]|9+q8L}8di9>z91<8s):,~8w*7lM7n28[G8Tv8KS8IG7~l8&A84N8Km7as8pu7q]8-|8Lx8ej7ZY8#p8Mf84s8_,8'h8(E8Yb8i$8L`8c_8lj90V8l#8R>8ou8Gv86d87N8[u9UP7y~8Uh9Wn7Wd8x|9;k8uE7S*5+&7:z7ix5/c8o*8gZ8uM8aL8vA8gc8$n5qU93X8ut8`^7iE4UI"
"3Z54Ft8[Y7-89'*8tK7_U8Q.8&69B}8.D7Ut8(l8pw8&B8>q8BY8]d9Fm6le6|a89m8Fd7mq7&~8[q8`n7B$70n9ZM7}87aH8rH8Qu8627ic9]`6k=8u{7y)6Kg8>r8{j9NG8j_7r67nC7H|8)&7uo8[E8(Z8`L88K9,G7p77w?8$u7r-7uC7hB8mj6*v7G_4O/8kI8aG7H{9+(7p-8,&7Gr6p,94@8r'9)M7]y8u'8*l7VQ8R}7UP5U{8Ko8PL7^J7vE8Sr8VD84k7pS98t9/'7O57f}8vZ8<a8Mp8C;8Sg"
"7OD8-?81<8A$8`n97q7H14Q&6yf9R|8j!8b`7T^8Nc8CO7FG8BH8;Y8LX8^*8G`8Z+7mR89{8VW4q^7ac68o5>s6[f9397N^79.6n)7N`8+[7y689`8Z283o8Q;7JW8H!8Zc8lX6R+7xT8I)8D)8$(8:_98H9-y7u&6Vo8>h8~a8L<8=f8RG83:6,(2km4Fq9'y7Em7W|8$;7s{4Rw6$b6-P8oI7d<8t[8hh7X97!A83':'/8K]8w+9(/8HL7m27;,7So82X6](8@`7Q$8~V8x`8&-7#48@e7B68Q{6lq8:;"
"8TX9N,8V58f/7n=9$'7Yf8SU6/u8q'8oK8>'8YS82`9P08e[8|h9N+9!I8#m7m`91P8L_88e:(X81(8_$7zf8d!8Tz8P880d8M899;9?/7;$9=a6|j8T^8k@8S,7N$6>J6QA8=]8(g8O98^_8659.-7<*8oy7no7F88O87l`8x/8EZ8,L8!R8,@7}J8u.90I8Yo7y78;o6LA5ai8y08mN8Kv82a7<I7S~8'78aW6Cq8/y85^92h8E-7Kl8_o8ox7vE9AO8('8+>8,n98w6X]6u99h'6_o6av8268>*8IP7wc"
"8*M8>o8u-75*9;O8C;8Ze8BH8Rf8M{8TO8KT7ZC8#B9=:8|b8-H8sq8d|7s=8fx8UG7od8;28jR7Ue8yD7~=7Xt9=`83.8em8oU8GO8wt7|H8CZ8d)8zj8&299W8@18Z=90n8Cr6qs6U[9>z6xR7;!97A6N;8xA8UZ74h9297wZ8)H7^78Gq8,K8Y[8W'8Y-5208#99B=6}f8=?8<x8oA8QL9A~6hi8rK7IJ72X89$8$N8c18Gh8I^9@T8.t7*S9,+8;/8Fu8Xv8IM8P.79&8qo7gy7{`8a(8Bi8Nf8'i6Yx"
"8JH89t70c8`d4@D57781$5X`5xY61z:oV2719+s8I18$Q9/W8W-8vt8Q(8Wc7`t5z]6[+7|T6|y8Su9927^Y83u9@37+R85c8px8PP81'7aE8IR7mw8O$6Y=7Wz7U=8p>8KQ5vp8w,8,@8$U8Pg9)':8?9QT7[-8)t7Z'7wv8Og8X'1Gb5^k97R50'6lX8j~2bN7/A73q81|7`-84{8l(8^l7Tx8Aj8w_7!B9+X7pu9+Z8E.7h:8vY6UP8,g8]<8La8dg8&p9'[8`h8B;84}8:N7x/7Up8S<8_!8EM8+j8MN"
"8L!7qR8{j8A{9E{6wY8C[9?$7A#9>x7HT8R=8iX8J{8o87|c8z899f94t8dV8rg8Bh8'P8_d8`#9YT7h}7rK8:r8o_8am8@|88P:$z8OJ6Tf9Rf5Q35s68;h89i7p#8m_7sM9>}7O=8X8:W47p'8Ha84c8Eu7<R8U{9=@96^7>w8:c7UH8zw8`A8md7z68,b6U`9)U8fD99t8au3e58f48RU5T)8Wc8K{78n8:07s16Nf7aG6nl6W|8157]!7@C7}b8SK6x-9!h8(p6n}8bA8CP7RT8Av8`Q6>f8y<8c=7ez"
"6>Z3!s7z087t7)c7uv8l37|/8^J8c]8ni7kW8V-8=/7zI8:Z8Lr9:W5KE7ka7eY8^s8iY5{U8hy87X8_98]Y7[b7)a7m28]#9Gw9M_7a-6L/5pC8l[8mI7]=7Ak8y46b+7p'8Ft9;-9$x9>J8*J8c]8[67y{8h38ni7ss7j07lT79t7^,8@{82e7LF5tz8Y:8Ki9'd8>i9K=7q28cP8Lk3@l8!r7/y7vd8>C8^r7E-8H*8(]9vc6T~88[7H87dH7N1:R$8&#7!S6d^8$b8Gm8897JH6or8Z&7B=61)8]J7Cg"
"3968i.8[s8S/6b@6g|64Y8Eg9#i7uE7t~7or2id8?b8SA87(8Dx7}S8Ok8j982Q80t7pH7t}9>}87.8EH8&i9xk8ml8>a8ia7tD8/V7dO7d/8,(4gZ8M<94*6|j8[;7VY7d28f&8y`8lC85~8A08+x8<u7}n8468W!9ax8)`8#18(o8bi7])8m}7u<7u{8?78`D79#8pW8L07_37Oa8n77mF7eT8A]7d/8;.8r[7u18#F76Z7;B95p84w8ya6UB9-G7|M8ds8E88}v6`.9{77hb5fF9&d7>d9A!8o@8G:81u"
"8j-8]?9!<6pk9lr8&H6JF8LD8X:8Rd6qr77|8c_8nY8@-6_;9>i8t081)7)#8z.7?$8~[8z27d*8[;7Pu9J~8jA8fL8[h85=7cq4|b6{885{9eq8?P8=r9Er76_8bU6eQ8VD8BE8(-7H?7AE7u88~$7i~8xS8.^7iG7`&7*N6s.8Sz8<L81~8/{8IB8s27]L8M38KN7Zs7QS8CX7|G8-y7Wk8cy7,y8Zd8U:7s'8Q28?58,!85o8jn8?]8.U8@+8XT8*18@!8928,&8[Z8Ww8JG8>M8J-8L08P=83#7yv86]"
"8|t7u;7iq5nI9$-6g57Ks8}@6PN9,T83v8R&8J(8.*7W,8F:7~_9d=9JA7zL84x8At7uQ8!Y88>8Uf8`#8<_7~y7#|9&57&{9#U8<78Dl7jM9LY9^*7PF9=;8'Q9SK7kC8?L9EK7hd8|f5gy6?)8?28&28=n89x8(l8KZ9RF9'-8X68h>8}~9I28F28(`8/-7t[8)v8O/86&7n^9A77g{9(^77(89e7>28#i5uw7|i9/V9AC3Xf3d58(56U04Gy76u9A#50/8H98[68818r473_6e67FJ7@'8^;8:67}E6Fb"
"6zL7q;7@o9'?8ub86W8=387V8aT8jA9&l8cx7e57D@9*N8O}8$s7#g8`)7gd8f/8(/8(&79{9k17x89JH:0R5hy8N|7Hs8t18Om8_68:48j<6?(73p7V-6Jm8e68+_7|J99t8$Z7jc7B584c86e8Sn8#Q8JS8Vy8Eb89Q8&K7o#9,f6sY85.9(18aq8`u8+C88j8I{7t_6z18688VA8q|7WM8`/9*A8W&8Bz6X39IZ7oj7B14|e8GA:UQ9fX:6b9Pc8,j8P89$58QD88L7b@9+O8K>8u>7:`8K;8@D71l8w6"
"7M48#Y8G|8~:8>d8UZ8Ub7lH7Rf8t$7sT7df9,/7&581}6EE7<_75/9H98$<8q984:7pw8JH8NE7*~8SP8-v8m]8ou7.a7aN8T=:/Y7@'8,Q7~t7<n8;j78O8/O8:c:P&7of8c<5SU8M29}l7.R8H`79I5-g98n4cO7p@8zT5-N8-t8Sv7i_8`_7}!80u8'x7pS8i?7kb8KB9+[5c@8F]7+A6*b8~W8+r7}U8:R9(o8hC7^{8qP8K37Te8Q#8f)9(78/|8qK8(-1~P8E-8eC6w}8&}7lC7Xo8a17{k6wJ8y4"
"84m8<J6Ea8TM8,=5[v9)>7q^6Li8g`8mN69_8Jy8Hx9T08+u78p8@68PU88:8o,8K{8&{8=x7};8f`7q{8_'7p48./8eE84X6S293l83*7sp8Pl84o7c?8X`8_P8oR8gU7ab8dC7fU82^87&4s-7497lT7:N8Q&8GG8?:9/[8?.8&_8+f7O>:>_7'_9EV8;:7qJ8ve7TP8H=8i/8J<8Z:7UQ7_J8AO7s_1IF8X57nY5)66>f7Pg91j9;*87+8DJ9//9)r8Oh8!i8*P7kt8]K7ui8IJ8Df8ST7|M8|v6Gb7x^"
"8J&7f_96@8@w8Hv87>8_R7`m8gQ8M88g(8*m8/!7>O9F^8/k9-e65m5l44ep64k9rk5oY5|>3HM8j389<1P[8_$8gX4V37l/81A4r;7E$8Se7d)8L<7k57UQ7b09E|7cF8c39567mZ8^t8$&8h+7y18mz8[b9&B99q8?D8=E8$}8F08Lv8q#83P8m58=B72,8Ro8K$8]18?.7zs8?@8Z!8:-9?Z:qi8;07t}9g:84^7WJ9#18&A7g58)(9(28M}73o8go8Y&91e8fN8Ul7me8oP9+]7;g6{R8Cr8Yq8@F87."
"8A#8:08_]9>(9.)4<c8|57v36xn90/7|z7bR;7_9~77{L9:C6k-7|{92585B8=S7=X8aM7w(5;R7}28P58T78tT8n:7h)9e79*.7{u:KP9!x86Y7rx7u,6VZ6v>8tk5fH7Bt8Q/7&U7py8;U8,P70_8Sh6d,8Wp7zf6eD7[R7z(8[g8:Q7}x8P}8M_8'i6,_7a.8]B7ot6[98u*7=U8Z*8-16[<9(!8R47bi7tH8TC6HZ7d!8CI87e91O8jT7G,6QY8A46os7$08Du6GH8P/7Z}6kK9'o8AK7w36{h8b573U"
"6}/8_z9#F8$.8<+7(y7F47Oh8-58x88_U6vh9E28N]7W}9/X8>P8$@7sS8@4:O*9[z7@67jN61|8]X9V!8'o8S$8vc8V@8^c7W>85$7fj8B-7[58Z&7j)8*A7uP9#e78$8hv5w)88x8,q8-.7vx9B!88'8^|7~09069GK9F+7/P3634+^7(u7K)8;Z8-P7bW8?I8rZ87e8}O81&8q^8HW8<.8Z|8Qk4O,7fU7~T6`-6&W8nA93b7|<8;k7I28Fp7rO7c=6:N8N{8Z=9Q_8AY9#Y9+F8'`95.8J^8_W8Jw8O/"
"8h&6~w9I383^8EJ6*_4va7lC6-_5tc7MU8gK8S48=48=,8Mh5zf8r78+f7Ry8G}7V46`s8977A;8@'8KI8RO7`;82j9kf8K58:35[f6387u_93U79m8UO7QA9W,9)N8588ps84A9I]8?i8[#8!V8(h7<i8:y6G:8=;8k`7;#6_i8u57,^8ZG8Cy:QO9!E7]w7[07QN8?H:LV81c8+r8|:9-G7PY8!l7'J:a(91r8oW6G08u:8b:8`V8{B8Y(7$K7eU87m8m'6Q87>@8aJ8MG9sK7Ij8^W8b18cC8fQ8EX8=J"
";/X9T18D?8bB8:48>.8Tu8_Z8Eb5+E9;::#I6^77H38Xj7:j8dr84e7Rr8Of7fU8*e9E<9Lf7t:7Fc7SW74g8Hw8`:7-d7os8@C7PF7Yu8VB74H8OX8-C8Js8Rb8lY5>$7L^8G'2h(3L-8De7!T8.W8]o6`=7et8?^6#D8cX81G8r;7~#86h7vM8On8lE6aw7cI6-d7+:8BF8W=7m}9'78eg8tB8s18AY7MD9.d88,8a]7.@7}k8=~7wl8]08JH7vB91a7)D70j8mY8@*8,$8;L8.;8'V7cc80M8Hi8Zs8Nj"
"7q*8978>Q8Vg8md7Y$7Ru8qW8SP7+58(y8d68,I8pw8;>85=8?M6H48.p8ax8,S8Tq8-y7]N9'Z8[|4(_6p,6z97cE7oR8*S8Xd8~p7xS9Wi8NW7]B8.V8#v7C]8X:8[O9M;4n:8];80{87.7u;7_;6I+5xs6R=7p)83/6$L8TR8m07fn8Hr8ID8UG8:]8yG9)@7ON7R:7sb7l17y-8m+7g.8#V7z?7z(6j=9(!9Nr8)M8`R8-*8:,8G`88*8Kr8I286(8oH8jD9Xo9FU7#R7kJ5|a8pq9]H5eu7ci7oq8`q"
"6]F8[J9/J7cC8?=8U=8=w63x2Sm7,c7Ua7X08jb9H98Jz7X^7Us8QO8-m8Ip6[D8GS7lH5o48{*9_s99393s8k.9/58;]8878yl9!?7CY8!48B+6}'8)d7_28Td9DC8m<7Uh8Tv8HK8A}6NI8_o9!d9aM9&h6?}7ky64.9!;9JI8=s8+07jj88n94d86/8[P9;B6ay9CH7;`8qd8YK8Au8Wc7uD5m-8UB8lS7qI8&P8!68(a8iH7p$9/H82^85n7'l84l8;h7,V8q{9B|7lS7>x80N85P8aj8oe8br8s?9Ts"
"82[8Gk7d[6A(7z(8SN9;Y8D99,h7=b7`k9({7u57/*;Es88v7Ri86)8Sn7bB8X08=l70#7;$8bE7md7@h9(D8a'6L}8;=74J7V;8Y/8dl99^8>q7.}6W}9=k8D97,181.9.?7}@8|,8i!8o/8Ib8MT83u8h=8e26u19'(9<A8LH8W`81b8&q7L}9#~8d!8G17Ui7r08f)89`6{)8.48~s8Uo8_57DW6S^8Y?7Qi5ff8d<7xA81?8(v7|R7kG6P#9(@8fS8Sz94m8L;7rI9LW9;Q8588?17<R88!8v_8e_85b"
"7dr7yY7w78-q6wv6`87N*40_8'K8/<5pU8Qj8D,77r8eT8S29)}7de7N08uY8C?7mF8n*8=S8-_8G-7T=9:z9@;8]I7$}9:/79&8D!7S.7uC7ib8bI8`w8Gd9A.8X+9>j6BX8Y.8Y88Ld7n/8]U4No8Fn8$X5Qr5lX8{D9c+8Sp7B,7OL86A8mG71D8L/8Er7D28UK8Jb7EU8.&8_J85u7ch8<F7BQ8L89Xy9e47)l9Ws8b[6UJ6PX6n/6{e7yM7qE6!>6dS5dE7Y87vr6c*6Z&6G=8U_8QA9;?8b78ZO7l0"
"8EQ9?#7+29OE9'O7zH8G(7z18*s7g{8E78SA8uL9+V9088f785&8ws9BD8XY83l8MD7p.85,5dU8D}9A)7a68tF7`#9F~8>B7u:7at92;8@~8;$8^(7it7Wn9d;8aC7^O8;w8LP8wQ8`j8Y$8y+7pQ9:V7PC8s*7v99Rv8bn7vx7NC8ED9&b8iE89@8Mu7I16sE6T36a(7`<8GH6e/7N!8LG8qD8Is8+98z89,s8I#7lZ8tJ8Ls8f/8HP8(07y68~^8*O7dv8+=8P&6j*7&k88S8sX8hg7d{7}:6UZ8.69')"
"9?s8@@8`a5@+5('6[26k[8hr7yj8?V88,8+$7z&7{|7/G9!P9$;7ni7<T7p_8uu8/467b7q^8Ew8NQ6O&7t;7?!8X69Y86S49.=7u=6x^8`m83?8mK8;j7X`3~z2Ds7;K6wk9+q9#x9iK8<:8>^7`07ol7{J8L37~W85N8z_8.67dz8h*8#]6uR7a58`I8Xu7Tz9@:8*/73Y7I68EL8gP6xK8C28E=7r,7n-6no7>F6**9NN8L68xh8f:7g^8p$6RF89R8I<6,q8F28><8.o8]A7sS8Zt6tG7AN8_{9.M6(J"
"7Qt9o.6u57367t<7UN7z~8yt7:W7_.7FS6!S7|37&J8Om8I88@P7Yv8bq8BM8sO8#r9#H8+?8/W8pv2@944j8WS7@N89D::58Cd7]s7*.:,47Wd9.K7xe7`'86j8cK7If8D98Lq8.d8/k8QK5yk89v6v>8ss8NQ8Wn8NX9$X8Ks7:Z8v.8]48Q481O7x78M!8Fs6}>5'f<rB9=b62V5e=6fN70^9![7V@8#G80c9GX6k:8@*8~B6i58[B7Xa5HQ8qI8YM8d'7p,89&7~G8A]6068s48cx6'u8}$8kc7't7B'"
"8$B88(:9V6{B9&18BU8H[9!I8As8u+7:]7Kw7w]7rP8TY7AT8V08n^8l.9(^7ON7A*9&b8]L8R~8la8Ux8qw9eb8812y87Oj9-88y27cv8Hh5q*7V_7Ga9R#8[+9#p8YE7Fb6|!9/&8/48xL7P;9*75g08Ut8h18X-8Qp8S09.n7.b90_8vS7-*8TA7t(7G[7HK8JK98j7g+9!d8)H7U#8#w7Xl6[$8CQ92v6EY8Ue8!H5v`8L_7$A7z@8RZ95&8m783r8WK6/h80H8qE7yK7p98)H7a$5g,68U8){7p]8Pe"
"8S47g(7xZ6u*8P?8Jq66L8]r8F:8+>8=z8sN6?|8de8i[7a87<w8:t8=T8JZ7=h7yE8],7aI96h9'87y/7Oo8jI7fN95'8b88<T95h8Vb8mJ9gI7UF8<#7U~7S$8O58Cu6{38#E7H58LF9A?8*|6[W68[7Y<9Ln69G8dt85684470;7gs8n>8&A8'^8,273_7p88f17>k8V$8m[8Jf8Zl6e;6{&8{'8/s6:G8[p6r$64U7W:6?u8+u7f19'r8ZU7NR95V8$n8)a8a,8&)8Jq8AW8/!8!77iH7r_8TG8fG8L*"
"7e{8868_L7Bm8?58}17vh7qk8x]7846l$6b.41u6s+7ok5/u3)J8G~7}27g&84k8Gg87i8@c8m37<E7|p7~T7^Y85/7yq7W,8(88178ZS7CJ6258C+:&n6BD8Mg9x^5US8&)8sv8$O8N88't8a@9#/6zC7G'7p#6NV9_h7i&6:*8VM8W87?T8gy8bf8B+8Kd7|K88p8T]9!o8P_7[949v8:E8E]7w68M=7s>7hW95N7zK8IR8C27CN8cB8}-8ec8mR8[`7&f6zR9:P7*|6rk89?8VL80P8k361#8@O8+C8yN"
"80:9Ni7,D8fX7dR8c:8cF8UK86d8/`8ZP7&'8AB8MA7a69#B9!F82]8nW8g&9MQ9?W6'67dg8Tp8C?9UC7hk7K]6{x7d17]q6q28<o8<G6wq8zE7SP8$H7x5:KY8Mm8S]8h)7PS8VH8/g8N<9!97{x85@9Xp7a09Rg8cp7{l6UX7SH8SI8Fm6$}98o5i;5de6U)9,R7T$7f`7Y87&c8|E8rZ6k=8pQ8HK7C|8QT9HW8|-8A67d78B`7qP85}8pA8@B7cA78>8$r8?r8o07e,90)80,8q_7})6y$7eI9(`8#9"
"88g7dx8W:7v]8@l7I|8w-8`98Cf7j$8gN9G_6p47Q~7>B8Fy8r?85k7qz7~n7;}7~o7iJ9X#8DH7I|8M88&+8/h8^)80F8;b9!S8Ge8I;8Np7Q^6rp6MA8dA7-59k985@8ZI8]c8O17<58!B7J;89a9aY8uq8Wj6}*7=f9'm7`}7TA7iW9//8(_7j{7[j79W9Gk6mW7j>85P9$~8o<8*S8ct88A8Wz8yk7|R8yD60J7ad8f~7OO7uF7i|7{p7NG8xA7bj7Y~7z>8tH84_7[i9K[8{l8gY8Xg6d:6Y^4-C6nq"
"6wX2$&:ev6Z25_48V38&Y8K+7!o5|<60A7H98=r:&l8D98[L7~}86(7]j8J`74a7EG8Ve9=*8TA7O)9$[7!A87W8jM9R~8)p8?m9.E8hl83H8:I7v@8BH7x~9=:7Zj7&A9i28368)D89D9-}8A39!P8/+6hI9(z8qI7xb9lo79f4pY8()8mO6M^7Qr82{8/`9-l7zi6iU9TM9'`7q29K}6I'8.b9A`5's5/+6z'9<K8/$8m;7OH7+,7C*93:7gU7Vz85m9]|7X+97O8T&63}9(/9;Y8Nh6kF6*f8*^8H~8tr"
"8bA9pn8FL9M48P[8O_8[c7pp9,~7=n8@g8h{8!;8KJ8Jo6}o8,[86C9[Y7RA88'8AE7c)7x=9'*8s*92567A8<=7z07H(8h+7S>8#F9bC8~i8Iy7~37G<8CV7='8t99#s6jF8'X8Qy7dg8TE8ll7I36ti8xy7:W87Z7a`8FG7Ep7t$7P_8#'6.H6{p7$?6^e9^]8y@8c|6O}6!e5qD8'^7yd7]T8=z8AC8uK4Ds6p/8.e5nw8Wv8SF8FV8b.8P@7--7WH8HF8B{7Ss7WL76$8+w7Y.95k9/S7jk87Q8+;7O2"
"8&r7?J8Zp6_G9+&6?Z8Dd4TN7jZ8@Z8eC8=C8r?7Gr8r<7m,7n.8Q*8LO8Jl8ao8+)81N7i)8hF8aK7m59&W83T7{75o{8z28R78Tk8u?8FS8J;8.08Sg8zs8DZ7Y=81_9Q>88+8}g8Id8(*8777.38Wc9&M7gg73p8nW7~R7(#8K@7rv82[7n.96g8:)8y{9$78G-7Sd81?8k`8jG82c8By81l8hg8:k7q78Dq9'l8[R8OU8ju71<8?m7Zm8h78?}8fn8b+9(r7|G6Kw2T75=|6C_84.6jN9J#9?;9<P8[W"
"8?y8lv8n=6JC6|#8`h7s.8]g81h81^8Of6Tv9G+8@&8/&8so7Pn8GG8E56m=7W#8gf7(77UM83Z7kX8XS7mC85o9`69`/7eA6NU56&4I`8-77ke7lc24a7N)6m46Q_9/l88R68+85=8Pg;T{7w*7eB9[47@Z:0{4k[7g07?]5#S6Br8VT7zi7e.8d!8r18Q88)U7|$8->:](9(M8rb8F<8jh87U7tq9BC71Q93!9+m8#p8Q'9I?8X+8T(76p8:<7~n8Nq8Rq8KH6>09oC8Xa7IF6:L5MW9,f7Xa8v?8:V8bp"
"8MU5gT6k59C#8I;8Xm8(f8AB8h$9598+C6Q[6^v7v^9?274Q7uD8Z@8jm6rk7^f9@o4XL8b^8k58(88Ou8TF8@d8N`7Zv75?8NO8q{7$B9#89tP7Et8h|8DJ84D9-*89b9t?8]x7&I8Y16<>8}v8c~8#z7el7oH5Vw6468Z18U(8_?7c38!.9ZF6,K6,^5;98+r8:}8z.8tj8*m7;)7GS8Ig72?8Hi8X=7p.7B07xN7]i7<S5EP5_>8a18B^8:77T{8,M6j#74P8C;8ik8-w8M17Hl7S56$e83N:F55hc7Z5"
"8Qx8[v8?/8lY9!:7bu8}*6RR8BI81f7xt7lD8Jp8cI8lK7vv74u8$D9)@8<77l47wP93'8/A7P48WA8[T8$.8!#8D77,R9>H8+q9O_5ql:468BI8BK8EQ8:v80.8Co8&f7{H7fq8?p8@X7sc7nh7dm6vu8N)7-86d18'284j8^?8{l88P8L`6^U7y(8GF8W]8*`95?8G#7jc8~R6|Q7/.;EI7{!8'F8a>4rP6(D8=I5k,8aU7`p7+m7L!6G26F+6~y8Z;5.h:n482V80O7qY8E&8Ej8Nz8)a7M|7e~8>L83K"
"8C,8un8+~8b?7|]7Yo9LX2MI8hM93`8$t8>F90{7WL43z2~.6q}7Bg8kR7g68JO80&84g8be8_08Y[8S@6T|6E:64-8(<9RW5C75}V7|k80486|8>>87M7h_7~O4QI7-<8``8d/7_C7o/7bX8f_7+$7Vv8X/6W-7QB84<6#*8}}8WV7yy9//6cC8$D9F!7e_8vr9dd9A*6w*4fQ7Sd9U{9#q8J}8K08E/86;8@c89m6jE:-s8,u9vD7XS8Y.9C~9M68:c52T7B{8q-7p|8By8bX9D[7{37CZ9!(7NC6j`9F6"
"5xH7x(9/.7ze89u7YR8Yk8bS7yk8@s8n{6cC6!:6b/6dH7a27?g8t799R8@g85*8A}8^+9#M7kW82N8fj8:#8I(8mR7Kt8If7ru7j!9Mc8,i6b#70h7w,8[Z8L68Y(8L#7Mq8d{84G8&j82)8Z08R-82a8&Q8JL7PV7os96T8hn8Ib8ME8KW7sh7lI8zT7{C83}8Q38oP8DZ7LS7[Y6j=8!e8H18bZ8X,8oo8p'9*@8Ql8*n6!385u82E9Z'8$,7WT7HC76=6^P8By7mS7O;9U}7a;8sD7RZ7?S7A18R85vz"
"7nx7DQ6[-8M^7N/7('8e94943KO8._7=~7Zy5he7eY8-M7cJ7A98$>82|7cZ7Pe:938177<=8K{8Zc8i-7Ij8zY:5x7mw8In7m87o[7Lo7m$83H7nK8;L85:91#8Fp7jc9a97={7oW4~)7~l8:u6!|8De7/^8Kt9GY7$r8?/7us8u^8d<7e68S#8JN7yy8d.9-'8U57X(8&x8+D87I7CS90[7$<8P~7W.9+{98E7`?9/&8?Z9#U7aO9,E6[F88'8I$8Ox9]]7j^8>J8I77pf7zB7+l8g+8.]8aK9&48HU7pl"
"7y08C$8'f8Yu8'N8EM9oz:);8y<7Lt8av83r7nF8(U7W;8?88F@6t891g8A:7}f82l8;}6wF7p37uO8*)8oa9LA8A)88@7o989`8=38sQ8r(7Jb8ZE7m`7w08h68aU6.u8_T85q7^$8$E8^*7pC8768(E:SB7z_9>[7as99L8(V9T(70Q7c98_/4ZE6dK9#m9D*84J7_#8F~8(&7=p7o48.z8E|7d!6Y<8|]7r&8AW7Fg8AD8a<:;)8sv8WW8{f8^|8vj8g88.B8P_9'l8KO8;N8*_9#R8UZ8M&8;C79j8}("
"8'J8<^8?28J!6oy6p^7Z18ZT8{<8At9n-7^Z6w]6T@7Wb6zo8DE9,r9:a9)G86_8d=8(99@y8LV6|{6|19k>7Wz9qH86&8tz8(l5~+8$C7[k7[*8ag7c28>!7g=9&17oO8F08N,8D(8Sm75D7<u9$A54>5xx76582&8ei7Ri7sl82G8Vq9]m8C28Lt9-&8Nb7b/7.K6|Y7aF7~i7;i8u*9y#8ET8=S7,?8A&9!K7z_7X*8!186]6^68:c8O*7109G_6Zp8d:8m<7(C7t48?^95y7zx8?{81C70_8;J8'98]$"
"8K]7yB7.S8L'7L~9It8DU8=k9#B8WV9:.77(8-^6f$8??9877[N8'}85d93z7UZ:+h8`J8T-8Im87t7c>7An9r28]_8Pc5RJ5UN7uX6Vh8<'8!N8LZ9#_7>#61#7B&7in7RF9c78&n8wP71`7a(8I&8Gm84[8h78Ai7q69j-7PZ8FK9P07[V8v87PI8~T8o.8Ao7nQ8b^8t28*57'76[c7J:8FI8FA7Oy8zS8AO7V$8m17sE7|@7rD7R&8t89T>8U78,{8ez8]_7V<8-68b@7=*44h7|~7!=8<M86:7cY3YK"
"8;z9;j5J!5P58ik8(M7~R7b&8=!5d,8T29j'6jD82*8Q<9c)99O7]z7SS6~$7>C8,O7Jx7`I77v9=y7N)8My8>{8gy8@V8RH8dQ9$I9!/8z(7WR7Ny8Ma8~d7FN7)09GO8q|7xM8*~8M&8.t8J=8Lx7u]8Q?7o570t9lN9?r8d.7nQ9a=7kW8X>8#x8{p8Sl8@g8Rn81679j8Y{8$D6;]8O#8;$7i)8'=7j{8fS8TM8=/7^d8Xx7h,7?57xZ8(x8oF7U;7f)83$7zu6p+7jQ9+R7.58*^75!8.C8y=6_68t5"
"7d39Q56`87t=6bm9u:7{K8=67Hl8YT6g]8-,8Bw8aT7W]8jd8?B8IU8`J6C&8NW81k7,&8[D7wx6~x89972J7,h8Hb8.y9(V9,y7u]7NK7x$8@F7lb6~q8i!7/U8RK8J]7Xo6dP8e68|x8t$7[D7kt8R47ov7t+8Y58=Q7qW9Iw9:}8,{7d]7wm9^a83.7RZ8]^73q8/s5}~8KF8Nq6R^9-]8@78IV8IN8Rh:+k8L27cN8gk85#7~B7{P9&37wd7VF7wR7v06d[8?i7x*90,70I7y,91V8,67n?7az8O48QE"
"9hD7&S8h)6x:7P27Gv9*X8Hd7X97of8G38!|76q8ng8/$56o9LW8K:7S!8P`8Z$7~581N8V(8$v80x8Oz8'88f>8;l6zy7N98A$9Ef7sw7vF7qD6625NE5P:6]i5qZ8A58#2;B17~t8{57U.6o[7I[83_7VF7dY8ix8]58.t8nL88=6p$8+Y8a~7/U8@a7gD8ZK88Y9]08Q:74(7;v8d{7n&9*#8~U7/L7FT7^|8>18*Z8C27'R;Qs:wo;+54J*7An7G~8M,7zG9Ny8+o8R?7Pc8$=8JQ8D`8]_89b8-T8U<"
"8]58Y(7d48.T8?~8*N8]r7y(7Gd7El8}!8~p8#e6[z7/X9!<8Mb4gX8*y8`s8#38;S8388/78]w8X;7O|8z!8'z8q99Rz8;w3kV7^_8Y,4FD9vV9pC:2<6MX8^g8-M9:c7J~8ah8(68aY7sg7JC8a/7Y*8e{9(35gl8F.8<y6h|8T88/$7G*8jT8uE8HG9f-8Ty8Az85N8qO4Jj8Qq7IZ7bI5:m9@!7UL79Q87E8~i7kK8]Q9HL8;-8[|8@88(J7H78D:5!g8eW9.^5-~7gn7b?88,7r,6e,6Rv88~8L~8z`"
"75W82'90w7f.8a-9qS80h7nv8iy7`k6OC8n{7h!77v7O'6`/8>'8K>8)<8j(8Zg8iv8,56ob6,G8Nk8ZR8M67[V8SO8@;7W@8<68Jh84g8>?8U|7xq8BY8XR7lW8)R7~i8qx8}17z07[J88l9$b2mx6g36';8RC89[8~36168638e53f~8J-7R&8*~8=79G=6r=9TW7wM;}04V64dI8|w9-S85v6ur8BE7I^6a~6Ip7D^8$r8i37|y8sJ9ug8M~7Wu7q/6T}9G^8;!9.$7!Y8$@7U*6xS5WX8@b87]9#C8Go"
"8bF8&<8>c7#n3Fj61A8&q9*S7I]7T26d?8E16xO9^$85K8LH6f>8KW7vi7vJ6NK8g`8B18{.7*G6n!7|t8ur4qA7Ah7F!8Wh7YK8'T8uW7xZ8I]8S^6C;7U}1^D3tS8V-8DN6~g7Uk8/u8Pf8Wr79d7_h7|H8u28[Y8Qc7[m8/i8nb6ZT6hQ8,Z8-^8:O7U16c]8YG8pb7j#8qr:'|8V!8kt7z38#R85,8Oy8wT7Tc8o!8G.8fO8f#7mg7Fh4<v7Xi8?o9#)7~V7`s8Gm8iM8sF7`H6(F5C'8iB8@e8E476>"
"8&:7;G7aJ7[J8n}8RY8>29'&95!7bp7|*9Uw8if9a#9&n8;`88(8a98848xI40[:$37xF67?7Hc7BX9.o7f^9_Z83S:/w:?~8El8_87Rm8qp8O*9f&8Z>9,>8,l7gL8R87kx8fZ7fh8,B7^=7ux:/#8Td7xD7vi7:V9^$7q7:)D6v>6|;8=Z8AS8(29yz8dp9Db8/08qQ7XP8YY8&S82;8J09Z49Ri7(,9Sw6a@7}c8kR7i$8}#9+(8dG8@_8]$9A~8YS8z~7Xq8AQ6DD6bD8>Y86x8kn8Gs8-/5wu8;V8Zs"
"7{k7$?7L565W8>/7Tn8~#7Bg6jO83R8nA8>}86,8(V84c8-`8|c8`?7bo6EO6#T8u#8<_6it70u6qm8<j7Km85r7e18*58+<8vI8<-7]S8b77et8mN5w^8*:8+s8kA78;8VG8F;7aZ9[*8D:6KU9.p8)>92#8D18V/7Hc8A77Ki8=$77a8.O8d<9/!8L48#68@}84U5Ws7vh7[m8A)81H8q{8N(83K9a179A8gQ8BE8kP8(.8eX8dA8^,7{e7UC8177NI7648M.9'p7VO8D'7*58QQ7o-8T98/s8Q?8Q(9J:"
"7/y7[f8]+7/:7{L9~S8lC91:8.h7Eb74<9l!84]7qD7Wr4^d8.t9-'7dZ6sd6V[8n+7Zi8Ox6y/7XY9Wa86.8j49,R70j8Fs8zP9'n7fy7QP8_z8DP6y-8Kg8Z+8R`8Wn8w^8L{7~A8Dp6{.8qt6wg8R)6aM8(^7Bx7N^:RC78X8@]6&r;L08<&7{H9,i94I8fF7Z^7E!:uK:kW7x'4D/5o,6z35gY6=J6Z,8;w9*S8kr7HJ8g`8&O7k28]X8il7hs7xM8#-8<S9BF5}g72,6a26Ze8&'8^M8oV8!=78.8]t"
"8dZ7Wo8mc8rp8U~8BH6~$7M18v18i28.79By9CC9h[8VY7{p7w<8@p7]|8xF7Nf8;I7d29/38sv7dc7vk8[v7.F9'!8St89z8Lc8W;75c8N]8LO8Y;8Hw8,r86o7UR6<a9H37qg8Fj9/o8J.8K+8;17Uh7k_7Xu8?$8{r7lX8zF7|[9A98.?83j7x39S+7[S82I:3'5409Jc5d'6UT8nT8MY9;O7N{6Ko8AQ7Y<7II96X8[18!G8hw7Q#7}&7W456o7)78:n8{78-N8Dl7r`9cN8*n8;g89M8$@8<P8{)8Lf"
"8,98SZ6e67@_8O{7EW9J07*n7?&8nA7aZ9*@7#D:G07MX6/R9(57?r7R&6wj7EG7?57iB8xP64X8<,8yc8-58ab7]b9-.8EV8F@8_o78!8'+7db8ug7l>7:l5hG8W28yr7XE9BG7Ti6wI76V8/f99$7W586V8|17n,8[|8Gk7E_7?]7ow7a&8^h8RT9,`8O46'K7N`7ul6R>9tK8Y;8,68[o8*~8fu8ew8Ro7FW8.r8!J7SC8J)83q6=^9H#81P8(W4`V77b80h7u|8Rw8xl9,=8KD8j?96Z8<&8JA8mh7f8"
"7f`88C8f|8H|7}98/i4`a77*5_a6jJ8Bo8de9Ou9$e8308-c82?:G|5a~7`e81M7{o7PC7u28o67f[7t&9-B7jL6}I9Tc7q&7*}8=Q8mc9447y96jX8<r8WL8+=8Ji8,)8]V8DZ;./8xk8A179G6{.4^@7XZ7vl5N}7?57[n6E]8vw7W#8P(8s=8p*8JP2G?7u)<a59-Z7D#6gF7[*8^676C7h-8ym8t$6:+9D98#B7g}5~=93T8q09bO8~[6hT9={8-N7[#8CA88_7]Q8c:7j48i?8><8~X8np8d>8!a7a!"
"8=08=+9!j8cU9/u8]B8Uh9!F8'59Lp7_e9'a7hJ91N8548OJ9)Q6>$8DP9xn8/!8bM85e9&w8G48uc8Af8Y47Q17:b8&*8J68v]7s}7~59xn8:78E'8P^9-Q8H482.8@(8rN88P7q390u81-7Z75?.8tT6ug8l=82!9ki8i58c=8;x8R<7p!8:y7TH7ue9a78z07R<8O38Go9Lm7&97@l8W66sl5KL8sx7l27K47st7{x80F86c8;;5fZ89<9'E5DT8aX7+&9fD8kv7g16{#8X+8IF6BZ8]v7|b8$x8OU8J#"
"9hs8Lc7CB67;8Ye7!V8k_8w@9:76xi8A!7yt6^[8)>8rF:J*8Em7rz4_'9#l6t}5|Y8LI7US9;O90)6jX5Lc85h8]M7Rn86L9+f9Dp8ZF8,B8oF8Z67lB7fN7|x8/S8=Z7EU7K(7zE88i8Cn8r06mM8P)8|396e7wV6uF7rS8TY8&985i8Xf7jr8L)4uK7q'7AQ7.H7te8WD7xs9tN9(l9)u8FN7eU7fh8h/8V*9'}7Yu7r*9>Q65,8MH5d~7Qc7wn88w:>17vJ8E688;7hb8&)8FK9.Q9$68>G81^:4&8np"
"8D>6z/8Cb8iu7K240M7us8[B5}P8WN7xk5},7p>85M8CF88Z8jY8X?7dD8Y[6,)8^)8OY9*Y8W>8j06v>80n95/7Ra7rP7M@6Ej6J-86C57H6y|8Nn69i83L7h`6<65~W8Ov9dk5u_3Bk9qa8kr5z-8+X9O-7g)8}:6u~9j@7{x8>p8_d91l8W^8.]7)&8$X7JF9?-8q@8@[9ZL8Ve94w8r=8go8E>5x27vK7i25bP7E86xb6AQ8T78es9(=7IG8i19O>86j9$P7Hb7}Q9$M85c6|X8]C9708)x9=(9kQ8KD"
"7r-6ap8?{8D#7mN8SB8;P9TE8S&7aj8~#8Pv8*q9Ld8SP8*K6=C8b>8>e5Pw7gi8Ll8Rk8&!7si9hL9>j8Ip7j_8AN7BS8a#8c*93#83q8o}8Ay7Nn7K=7yD8{h7y~5D#6h|8K*7X38.h82P7;w7GU8Jg8tU99t7uY8]C8yM7^*8X58os8)77Zr81;7$|7Cf7VY7!<6IS7Ob8K27^#8Fu8+Y82t8ry8*@8$o8B48$I8V{8Sz81O8Z08Bc8_08@X86O9a=6|,8Vm9G58<h7Ne7jN8xe7`W8gy7sG82q8R_8)u"
"7}A8i[8<'8S19:683}8:<7x]8nY8)~7l|8e188A7pn7Q@98T88a8O384#7K]79,8*H8Zi8P`88K9+q7uD8<C8I@8/C7C)8Td9FY8<S8J-82w91B88I8U}8)486R7r[8bI8{y79m7Yx8(^7O&7U-8PC8U]9-`7kq7xd9XE8S!8{q8ke8&,8B47h_8=872K8:z8Sn8Qn7h585f8)y7s<7wN8..7H76Iy67'8'v5z06xr2uK6=96H20nT7M>6+i1<{7dI8bt9AT9o*6Mc7sx8k=7Bc8`27af7wI5wV8(=8dI8R;"
"9&T7Xe8/p7U@8KC8j16h$8}~8Wf8EL8Td7Lr8xN8E07aI7:^7YY93v89d8#U8P)8H,73b74~8aK7M}8&y8`d8d~8]G8eJ4eR8|28)23('7+[7,06m88-98eH8$76e#8>n96I85q7m97Z/8)i7W!8up8Tb7!?9[98'q86.81M9;K93x8cW8I'7_d8<>9(_7dS9!I7$:8.|8dg8`^8Jn81[8SW9#x7{+8zQ8vg7eD7qR7uM8R*8.I8c_8?}9UB9?`8s)8,i7dL8vx8kD9:i8o{8P]8PH7az8ff7h>8>q7g#8=)"
"8zG7_|8LL89.7>$8Y08-l8.197J8&17as8Rf8^47|t8H_8I28Kq8{99LO8/=8@P8.S8$b6EC99x7Tv8A28L;8$}7~47)t:)57cz8]J8K39Sp8L!8(p:S>:d?8|D7]+8't85g7bQ8/a7tY7v+8Qk8Bf7!R8O887*7gn8VJ8Q78M]8NM8&.7Vs7z#8X?8sd8>)7~`6]76~$6+N84#8s'81G7Xy7V38+f8Bx8z~8F17y~7|48EI9[.9)s7oZ9(;8@a8#@7|z8G-8An8k-8fG8XF8Fd8?m8QT8Ll7-n8X35}{7Oi"
"80T7`p8SH82i6ci8Y`7Be7}]8x)8`K8M[85?8Z(7ol8S=8BK8f27yI83q9oV8$:8.O8>_96k7C]7e:8.+7Ww7Vi7T07E.6OT8:h8bB8Sa8NS8;=9!H7f>7-87|V8~69457?_7=-6tu7:n7=v7qO7w#8238WY8Sn8,78ra6xp8=`8q]8o{8]p8$i9JP9/08g|-Y@7#^8$D5w*65A4bE7jd6T~4=48G36y67Uk8HZ8><7Ik8c#8CT8{J7{{8PU8Z>70E8:u90f96g8TD6Wh8ZH8?T7ql8Jg9&#8bo8tL8ET8.1"
"8Pu8FJ8;j9d_8UG8DR7lg7oT9*p9Be7f]8XV3PV3)o7/w6x[3d$3HM8=;89J8/a7V@8sE7r28dU8ev8Gx6RI8844E38m&8M39,M8#07r)8H'8cN8pg7v'7|f83C9!'7DY7^G8(_8>28dm7Y:9DA7TV8hZ9-Y8Bj8W&8rQ8Zl8527Sz8_P7{n7Os8178Is7qq7~'8&D9G^9U67t_9`U8^$9*P:eP8RY8.@7|@8UB8h38Fy7[W8]O7Rb7h]8EJ7mn7d>7{v8o88vz78l6{486)8iT8^i7hQ8.^8~d7gx81(8o>"
"8169DE8Th8['8Y97_Z8(h8DQ89X8k'7v&6228;u8-78l~8L_7zz4@}8kt7WI6eE8j;7Y<7bB8E}8:@7VV8yG:4g7:R8zO8A<74c7uV8wQ7D+7fQ8zl7?i7IH85a6}+8328Vh7FW6}*7Y15k'8K{8[;9fd8*P8,u7ET7]S9+|8CA7P*7rA6dn7}y7Zw8Zf7zN7Vu6'=8g587/9x<8#*8*I7MT8h97ur7?I8-F82r8fp8K.9(t7$^6wU8]k7I-8]x8ev9+f75@8Hl8;O86q8j58.R5Rt8h?7??7t+8iA8>:8Sg"
"85N8GM8D37rk7mp8UF84n36(8S}8U42s(8$u8c_5~u8^.8&)87>7x$98>7^d7LW6pT7OW7X*8NR8/m9*m8]#8=;8Q083b8427P.8i387(83m7|}8X78,y7q$8=48:K8@&7nR7.V7yV8,U7n/8k69?&8LV:3$5#*6}<8g}6e@7p78-)6n)6~A7D38jM85y8=C8ZQ8Co8c=8728*98C*7T=8bB7v(84r7p:7yi8D'8Py7vn7__8Qd8ZK7|$7x~8Pg9E86g38dw9@)8?,9)j7Df8508OP9K.8[l9-T:)@9;u8#)"
"3Wf7!S5en3r|6j'5r48'!8eX8WT8-s8Pg7y`80u8Gz8NE8tV7X)8tR7[b8+q7YQ8UH81{7~B8TF7fG8(k8/o8U.8PI7vi8hQ7gg9_D6_=6X;8h+8fV87'9Dt8Vq7z,9'!86c8Fk8+k8gJ8EW7^~8hO7EX83+9?$83^9Xn8dz8Qp7,o8!a6,!8zX9s]7a66bh9<A7l*8978*r8Kd7pW5U584b7hM8X]8E;8EX9'Y8*C7Lz8TT7q{7?e7Y68^i7O-9:*8J`82v91$8TN8Pp8AY81n8M98!I8Gd8>!7>y7:*:;|"
"9_^5Uh7,Y8gb7_595&8qI8UW7$&6N78h>:5-7]`7nx88}7u[6m.:g29xc9]>5X87'(8zN6g/8F#7eH6Ys;/r8'78(x7+)8PL5SA8gF8d(7TG9$f7Fa7i>7Xo8nO8:{8k{8cX6eo8GX8f77ON8398Hw7fM8ZX80<8U`6BI9lV7u38P78_F7uB7rF8?,7^z8;y9lu6xN8mF8PG9e/9Z78ri8J*78`8)o7c*7-r91n8168o{7q.7W{7gB9YY8[17'c7X17tY7Pc7zd7/'7v]8{(7JV7s=7G68z:7Yu9:m8#i7x)"
"76,9F*7G18Fa8xG8-s7w`6d18s17vn82L7_~7Ow9Zm6hD7Xa7t[8M87Q)8_^7fJ74h7MV82d8hL8G_9,D8Mo8c|84H7&96[~8$!8-u71~7hd8Lp8x280A7!A83|7{S6bO8V[9;]7s?8SB90M7o>7mQ8nr88l8w@8=86w{7cu7p_7KG7Fm8R,8k_7mR7na8UU9)h8ag85F6m38mt7o>7&}7)r8,Q8vG7GZ8Iv7lu8Hr7]h8j@9-Y8^y7Bl7W&6Bc7qF8.B8M#8um7|)8Z$8Vu85*8AQ9t}8BA6g28*'8h'7nH"
"8Aj8pS8$D7{v8`X8AK7b~8`R8&{88e8+*8`E7IU8eK9G|8Xu8X:70A8(P8148'w8dJ6g,8T{7'>8-.8<>7i77B*7:a8aw8+W8b18a89#$8>A7=Y8EZ7OH8i&8N^8o=7C28=[8bs8#q7mN9<K8'$7`a9-S8bp7V(8*i8?r8Hd8r$7tH8Ye8rl7h98K,80J8ir8x^7rO8IN:.m8+p7Ju8z.8QR8eP7St7&P7RL9#*8W[8Sm8>?7vY7oj9F>8797/!7Wk6gP9T69*E8Jb5mr86K6aV9r>8ev83#7wO6Iz7[h8e`"
"9&i7sJ8ND8R!8F.9<.8WB8_-7'M7Tf8W*8mq8JG8[a81>8t?9J<8/Y8-38BU7|m7}&7!O7yH8wQ94A7/79Y;8Qc8vw8068#G8.]8=S8zP7w-8(}8:|8pW8.s85*8cB75!7.z84t8yt8dE7[,7{+3G$6K/8<?8)h89M8k67t!7[[8P48Hy8;.8#^8wa8y;7(o7DP83`8nc5tL8948)j8'/9ft8.P8C38g?9HF5.n8{q7Oz8gB9$b8*t7o:8v_7wU8|l8k>6oh7IV8d'8pn8Y:9yK7uj7:q87H8yG87]8J~8X,"
"7fl7.v89B8{n7l84<J88V8Zp6+66(X7f39>Y7@09'V8Zi88:8q#8a*7y*6Y~7LP7@_8$m8kL8_O8`u8=k6e)7D37g(6<m9W(87.7h;8Te8ER8;'8m-7k)7p)81B84l7~u8;Z8&b8Zn8q387K8Fn8V37av7c.8Yv7`p8ug8|m89;8Il7'_8/?9'y7)k8L)8=y5Rc8uS95e8Ia87|7{05Zd8!R8W.70-7u89_<8tI8<J8s!7-68LC7XK7|Q8!(8A78J:8n$8,r8Au9gD7w#8dG8E*36?8hM8;.7<e7Qg7sC8:{"
"8&V8+M7Fz86#7~y8v<9S}8{.8lA8cU8av8xJ6=K8[H9E98Aa88B9!+8pX9$e8dI8d[7w#8lz8W*8wW9q(7C!8)W8oU7[^93b9b/8gM6<,7-/82K8@P8`K7^B97A7]]7)E77c8Ee7^V90N7Qm8O`8D07^j8|M81F9,48/#8;M7t08iI8K/76V8u]7R,8A:7|t9>183H6s^8+d8$X7P&8;>8g(8t!7.a83]8f@6Pu9Ub8Tt9AB8:p8D`8<c8WM8Lt8(U8V;8cP7V+8M&8,]7<g9/^7z+7W_81w7{Y8sd7qw7ie"
"9]z8I^8@77.*7B.8qh6mL8YB85N7W47|<8Do8pJ8s<8@H66I9F?8CX8S18bk9[b81o87e7y'9LH8E<7c74a#8yH8Pv7pw9!(93`7-N8MN7r46b)7#27!A9#P8456qT6w09l#8+T9X67iz8g47uB8[.5Ei8*19W[9KO9,t9Ek7sn97U8GC8b68Af7am9*18N68Ws9CG8&g87w8^v9vz9CH7i>8(h81<7+f::K8r/8o`8yR8BV6o*9+H7xi8)n7b27{^6y86RE9:a8gk8u|8AF8Ai9OM7D!7Z,5ve9K+7Ym7v["
"8c@8.<8BG5_B7b=9iE9`Q7?L8lY7B07@l6ku:9c:)<7P`83N6pu8o57v@8JK7_u7G+7e67)|:1j8pi8*H:>v84z8j/7E'6e359T8g&8^W8,>8Ss9#[8u|8139?w7/H6N!8B@8vN8zo8lB9#W7yw9/)6^{7ix8SS95E7am8Jg9Nq7i'8,@8+48?C:(*8LR87u7OL9.F9iq6{/7mx8[A7Z~63g89j9*!6X599/76f9=$7hL7O-8-L8UH7|v8ib8Y48YJ9e`7QO7cR6XJ9R98Op7e+7WY7gK7mO8-Q9XR6XY8-D"
"9F@7?,8]?8Z'7At8$(:0P64p95L8!m7_a8fs7)!7rP8,68Rq81)8ej7YS9>e9T,8?A7#L8Ly5rA8EY8K&7e76y$8[b7>$82X8O&6r(7hs8si75m8bJ8!o9b08_o8-M7aL9[58c18I.8+]5:z:a89e39$#7AM8Cd7a}8T<8si92=85a:fV84Z85J8f`7!K8{,7v[8i^7_I8w]8Cu6o(87G8-?9Xk8iB:Qz8fO8L]98e6[88`[7^G7:>8Z@8yd8XE9A|7pn9)s87U8_S7N^8NR8UB8=F8U'7r;8AA8hc8O}84?"
"8JA7`t:;L:'Y7m>80j8&j7s#9H@7e685u7y68fH8Ca8d975h93^9>p8-c7^<8^Y82x8}^8}A8b_7nM8JI7[N7r:8~Z87a7r@8i>7yH9M(7Db7fc8px9JR7_C9M~88B7n,7YC7iF9'.7ba7zT7u_9J}8<[8`47TK9hd9Pw8qm9F#92*74L8h=8;m8BU8w,7y&8oY8-88R189v8yF7Is9+A7cM7LB8BZ8~<7pJ8hQ7x985i8b$8A>81>8K`8s,8H!8'Y8NT8yX7jM8=&85q8;O7GM8sw8[K8*[8d+9$n7vV7iP"
"8U_9d68-x76d8o/7K#8Xn8NE7+F8&A6T~82W8#37At8D99<R8<r8UP6IL8Ck72M6yq7BK9M}7;}7py7yp7jP7Zb8/E7TP7+a8-F8Mw89z6qe8_H6;f7I57cG7GL8Ky7t!7vF8WE7eC78!6{-6?#7p@9,v9/j87r8K88@z81J8<O8538<J8:D6i{8Ae8N~8G;8G>8/58G78<n88X7-R8@H7J583>8D08GE8HG7`m88v8=.8IS8C#8?}8Lo6{a8Zm8G-8:c81r8D?8CG8Q)88S8A28A|8>L86W8CW8897qN8;B"
"8?m6wR8:o8Fv8>P89y80U8@x83o8Cg8>98A18Hk7|#8@$8>C88R8OH8?78{x86I80u79{7=y6h/7q+72P8U$81=8XH8:`89y8CS7=)88t66J82q8Co87v8>&8178Ss8!Z8>r6FL8$V8GN7]]8<}8=p80Z7Vn8@<8IR8?q8E/89[85h8_E8458778ER7T$83#8?W8CV85^8Bq9;88M$7ye8^X9B`8E+8;u8W`8U48XQ8>~8Wq8D,88l8VP8@u9FS8H98Xw8]:8V{9:w81-8M687T85k8H{8?t7}V7n)8Cr8EB"
"9!-8C;7w!7rI8R,8EI86g7Xp8?&8O$85m89v8E:8UM8#^89v8GX8Ep8R_:cc89n8+;8c?8#T8G>8K)8<{8Nv8Kc7uS8V58Ol8@*8:.7Jh9><8BP8Ts85w8]b86*8U`8HE8Dd8T&8OF8Lq8n(8@e8ij8T-8@F7vq93z7ub8Y(8gZ8e<8TR8BL8M[8Y`8=T8V<8I=8`I9[;8:Q7s&8*n8c499Z8L?8]d7[&7);8FB8Pg8fO8968FQ8L!8UM8Vn8[/9'+9E.85I8=a8S28Ye8qd9s{7ke8Hz8U=8DI8P]8m#8E7"
"9227xG8[`88a8w_8na8.)8Mm8D'8Cr8;L6iP8TC8NT87*9`W8F683/8gV83F8/18]Z8LW8DF8O68p-8Gz8EB80h8In8U,8<y8g48E78E08U.8Vn8+B8DW8kZ8<i8O#8R)8}x8HF8BS8PX8G38Hp8Gi8D~8P+8k*7j[8Xi9-98ID8_l6|n8Dt8HH87p8Q*8@/8ZP8;D8@+8;q8Kg8;!8Bh82f8!D7bc8/$83g8648:H81k85R7a_7|f8N47n87r>8Os8:(8:k8AW8BI8Xz8U88nS8Gd8;$7x.7XK8Ar7mE7|U"
"8?C8OB8##7v:7D88-@7|[6PW8$$8/_7Uw88d7f09+78EG8.E7eB8(R8AA8>M8uQ8rs8+88E'86j7j18'^8A>8AQ8978:@8DK8.K8)?8J988p8?A8;Z8DX7tt8?g8K{8!B8UL8k@8]]8Qw8]<8;=8T[8,$8?g8GK92N8?B7i{8el8?I8#V8U'8ih8]^8S>8,s8G08F,8R_8768Rq8M^8G(8ie8?b8E[8@68=g8?#8I[8Lt8'X8>E8,+8LO8,I83/8Io8U?8={8W.8Na7c>6ZB8168$N8H.7j58**8Z;8C=4`&"
"8108MJ8NI7YQ82S88e8g)7m68Ka7bQ7NZ8QL8Ew8>Y8-L8@=8I^8BN8@=8sm7u_8<W8p^80o7^j8RX8Aa8^/8Pz8<586u7^!85@8;k84>7cL8@G8@_8&;8>/8>p87!8>s7`}82598*87I8Gy8EY9#V8KA89M6J}8Fh76Q8im8[N8LE8;Z8Yc8ks8S|88y89T7kq8H280+8H_8Iv7;$8IG8Fh8m)8Gs8nQ8@r8oA8f98^Q8E/8CD8an8BE8OS8;u8T|8[N7V78[U8_98Lg8@s8FL8v^71f7wf81f8Q>88N8wU"
"8F:8HT8Id8KQ7Ir7[t8A387{8Bh8?q8Lk82^6S.8V]8Qz8E78Ar8;l8Zy8<58_47am86Y7~28cd8po98N8Fz8D985n8R38K(8>m8Kf8K18Ou53E8T+6~A8GA89C8rS8#<8?y85b7VY8:A6e`8>R8?|84&8<g8:t8Vq89V7uZ8BF7y;84P7F687w8B#8787eK88&8=}8D}8.b87N8Qk8w38]=8AA8o{8DA8Df8B{7Q98<c8:J8Sk8SG8wd8_C8.c81Q8]w8nG8`&8Cu8F.8Y{8OR82=8_q8388CF8-282[8@Z"
"8@<8(78QB8_@8#j8;t8&:7bT8D-8Xc7~]7z$8368E48DV8_(83{87!7sL8;@8Ih8]G8B{7s28]}8YU8U$8J|8WC62Y6{h87;8HJ8-U8K+88{7@,8-Z8Xd8Ze8Es8Pv8I28Ub89M8iw8LX8>B8Z78:S8Db8HF8<a8O58EO8J[9^g9OJ83`8He8K|8!o8EZ8Xs8N38R.8r`8_O7N:8Gu8/~8HQ8?980(8:J8I18S|8N)8:58b~8:]89q9R_8M*8GI8a~7v*8/i7M?9*X8/q8M>8dA8AH8Og8CX8cK8Bt7|N8|#"
"8n|80M8AI7n#8JQ8-J8IQ8?^8:k8E78,E8Jl8NF8C*8CC86b9`}8a*8HV8>I8[)8-'9&78=,8>A8B18A?8L'7vA8L^8,;7Mw9KQ8a38F!8C88@W7K~7Nl7SF81x8=B8J48oJ8Q!88^9+:85;8Gz8M:8m27ed7G79=R8Or7w.8Cy8Dc8U;8B~8sg8E*7xv8738i)8HC89'8N,8@M85h8?j89L88T8eY8948B=8`,8G98FB8?W8L`8IB8C$85|8228Af8?T8=56@;8RI8O,8=&7|R7s~8u.84p8P)7fu8Ot8<~"
"8@}8,y8Ga8.k8*D8EE7H>8BM7{C7i<8258?T8@18&;8N58q?8ZF8NB8?57aj8Cz8E&8K.8Qy8BG8$h8/@8fi87^8*?8CX8C[89V8VO8Aj8A283,8-<7qp8]R81w8#k8gq6[h8K]8;k8E>8/~7eU8N}84g85*8C-8l48?R8IA8q[8;s8G58FN8Zh8Kc8(=8I]8Gc7UZ8@{8BU96[5k48^78cc9#h7s18E78mz8Jy8Kt7#N7e^8Ph8^.8C!7]N8Bl8B-8OV8A~9&47q,8Vo8?}8>18MM8Pf8,<8`u8<Z8?28FP"
"8?,8bU8FN8>48g{8=I8>K8KP8FN83M8MQ8@68/I8@&8aM8:f8FE7cx8Hd8Kk8wO8S:8yl8?S8cZ8#H8X~8DL81/8A68JN8O986+8>d7+>8@;93J8IA8'H8;G8>(84t8PJ90J8N}8Hb7b!8fZ8D28Gu8P[82/8C38iJ89W85E8=|8>(8>.8A388/8c*83[8AP8I-8Kp8Bf8<u8Sr8UT6n&7|?9*$8F|82^8Ab8G{8;j8Hv8~}8B!8AK8S(8@R8ht8t[9#Y8I*8QX84'9)Y8:M8G08b]8@<7cL8gc8=_8/I8I}"
"7>68:*8^98K/7>d8<}8B18st8Ad8;+8@!8t:8@=88'8P}8Cm8Ff8D_7kc8Rp7[u8XM8Y87i&8A#7tC6-g8=f7Pg8Dm8o!8;y7hH8MD8KW8<S7fM8C?8B686S8?b8<V8;g8H/7I+8*`8L68Mi9f&80N7b38Yu8en8TK8bQ8J[7Gg7|=8/i9D@97(8@28i'81R8VS8XX8HV8IM8G.9!k84;8eb83+7~R8Vm8lm8F98G'86f80b8KC8@O8]m9@>8=;8@@8;:9&F8CR7$]8DX8ML81p8;V7fk8DG8V^9$V8!99uV"
"7~,8_U8&38a_7Ev8dH9S~8J=8F58=h8??8>f89>86T8@s7mf86c7zR8@R8>68Gm8:?8pQ8H=8*=8:!8PY7U$8Y?7r98k;8OC8HQ7~<81?8I88PA7C:8UW88h8F286a83d8>>7j]8AH8f.8PP8<~87.8N&8TE8.T7lW8>u8ps8E79A.8AY8?!80K8CQ83}8G07ql8P-8)h8G<8c07X)8)G8,w8C}85n8Cq8@T7&|8MJ6kf8Y07UA8)48~c9+88==8[z7W?8O=8Xf8=O7}i8B97`r8+|88p8d682m82S7|b7mN"
"7{)7mI8/R7T'8?~8)t8Z88@(86U7jQ8S!8/N8[>9'+7nV8Ju8888UU80V8u!8]!8/C8FT8:G8(n8?E8@;8@48T@8.h8Eb8,D8S48E|81]8J!82$8$y88t8+O8S97we7SH8<.8]y8Hy8;m89w9yR7y}85Q8,Z8<F8?h8AJ88+7sr89q8Ku8Ze8AS8Ha86[7dY80y7vm4ST8wy8At8F18'G8+r8X68+n8FS8^)8/I8498=N8X088J7{28IH8@m8:S88a8?R8'C8CZ:di7|s8Im82w8$C8N]8@&8Rf8KH8~d8Lb"
"8CP8M181S8Lz8i]8J^9#38<E8IU8kL8Ap;#W8AS8m88<q9#78*b8@}8Q=8Uy8gm8f{7Vt8X88nl8GE8[:90X8<^8Yh8],9&{7w~7Tl6c^8=~8RT8Jk84w8lI8v[8!w96Q78A8F$8S48Yc83!8,p8;;8?u8F=8tY8@E8Q~8=g8/)8=W8B98QA88{8=B8G58@(8BN8=,89T83Z7=K8;[7~]8C58<u8F;8@`8@18AR7pG88,80L8B(8?y8>s8C,8+b8EP89t8C,8;X8>?8?*7_38BH8Ow85Z7fg8?P8:d8?l8F2"
"8@<8@N8B~8Cp88q8Q180m8BH8A776!8C!81V88u8@h8:n8Ir8;d8C*8858<v8Q)8)W8=)8CN8N}8D&86q8*Y89*8>a7Cv7y17f:8Vp8ul8Li8B@7zN8r*8BQ8Cl8C98G}8.G82b8@V8I@89I8=T74n8J/7LT8CD94s84l7mX8=@81(8(28va8AR8A,89887]8;@8N186m8^:8=N8t88GQ81{8=T7q'8m~8CS8:*85`98=9:w8c98L}8Nj8a?8$187W8A/7Bi8#Y8(+7&G8=<8H68FK8E48G/8x>8KA8KM8#8"
"8Ds7tp8bO8@k8Dm8g@8`F8U98Cq7y18G_8Z-8^I8Vj7g`8T88D18*=9{k80$8e>9AQ8@J9#D8O,8L~8T58?p70v8P;84x8Yc8W(8a]8b38>a8=$8j/8Py8A68I{9}78GW88P8Mf85g8D`8A08I18`m8No8C38Uz7ne8fz86*8BI8@h8>[8@I7yr8Na7iO8sc8D58Y188X83l85n65a9/t6K#8-y8EA8((8Yf7qA8FO8BP8rP8BT8}E8N97`d7fX8xP87d8!56'@8jy9+*8Ge8P`8.@8@<8Y~8`@8Y=8,s81U"
"8CK6mr8DW8T.8Jx86:8<08D(8Et8F}8A^8A]8?K86f8G#85-88Y8-T8RQ84[8=y8>d8E>81~9Cp8k>7m48RW7so8H37q?8:i8Fh8}48J{8Gq8C~8<M8:68Q57}c8:87|@8*[8Kd7X<80_8m-8C&84e8FR8b98F@9Cg7#588789X7lT6zZ8[M6UL8pO7Us9D-8=k8[#8*&8S.8@j8-O8?a83~8_78K^8J$8BW8UH7x$8>F87v88#8Ds7nI8Mu8n[8Nj8/L7}M7}Z8A!8>97v!8*U8_)8CZ8;H7>l8fS8LU5Xr"
"7Uk8D~87=8Te7`y83_8^X8C@8A{8/k89^5xX85`8:A8ZO7Cu87G87m7ug8No8BA7SQ7{E7w^8$o8.F8Vk7zS8<}7Ul89C82:8BN8<J8OI89N8Dn7Y!7Ce7|98*j6vS80U8&K8>f87y8FT9'!8@B8Dl7'$7lF8$&8HF7i#8gd8J#8xr8R48K389a7s_8I98fy8qY8FC89t87u8)y7ni86=8e+:A18$A88h81{87p86n8Gt7>#88J8Bf8J|8Ap85Z8DV7hJ82c8z^8$99)>9!48?Q80$83s89D8MG88u8R~8F|"
"74o8Fa8>j8=68GC8>+8DT8ZO8-`94$7sg8Ed8@c8UF8L(8Dh81C8Ot8PA8!a8.Y8P+8CP8.{8A>8g}7]{9+N8E/8AG8Fd7je8a]8H:7f}8D=8;]8KJ8/Z8](8{l7&>89/8P=8/-8O98UB8>x8(;8Q@82u82t8@78Od8Uz8JD86,8gX8Hc6*>8S68/B8Ef8<28KE8Bt8D;8=S8H'8G:88T8HK8tR8,<8@R8<]8:-7n@8>s89M7dn83J7H`88B7~n8)C7tC86p8HQ7~(8D!83!8[u8A38Aa6y&73*8D*80I8J2"
"8>~8'O9Iv7sU8Sh89K8Gg7dD8:!85;8*687K8~k8jd8&>8G$8Hx8YE8HR8R17vO7mb88(8+j8:)8/185s8Qu8@s7z?8E+8<o7i_8*k7#[8+|9M08V=8ho8Cw8:Y9pS8CB8J(7zZ8S!8:r8kv8Lq7ws8c38K48<N7eY8A{8?.7w+8#38M[9]u7WW8e]8S<8.[8;p88;7~-8wy8GA8Lj8gt8om8Jl8NF8Tm8Hh7jd8dM8XI8?^8M581Q8Fn8f)8;h6Bl8?l8I]8FE8Oq8838@P86H9$?8s28VZ8`H8(x6d=8U9"
"8`O8h`8Is8hQ8Sv8n,90:8av6gF8d/6kQ9#_8^R8ZE8RH8/O8]d6x$9@v6(78uI8WG8Rq86m7NS9@48b49'o8X[9$x8T47*E8L}8@d80f6Jq8dz8[i8Yc8Z,8cL8Q48qc8rl8x+8a88Ue8hQ7q-7'D8L18Df8oL8~68{Q8PJ8Xw8,09;W8Ea8f(8?C8OQ7y(8xl8^T9(i7ka9B57cT9b19cd7d97[H8aU8ce98}7`683r9T28746e_6uc7mY69B9)h8ux9}d8Vc9$$8=&7-k8rT8bc91j86`7z^6MY9N98GN"
"9.-8(|7`t6nE8yn7S=8X/7hO9)g7SP7<=8pt8t.8J46qu8mc7r=8Iu6:M85G8Fc7?T8107Zk6x#7dv6<_8Q87k:8$39Gr8~'8#l9VS8z=7I388?9X`6xL8v_8Ie7>&8Qe7k~9:87/L8-A7eI8F57md8z@8{p5&]7wv8/|7C781F8lu8VF7(N8&87hR8X;7Tf8i-7Z[8@_8hH7}n8Mc6XL8Us7EQ8e]83w8pH7j}8zj:4b8g$87*8UI9L[7_P7p*8(28B_6!v7$r93v9ay9<T75u8>T8fu7iA8|>5U[94L8hu"
"8FN9S/8w'9+B82j8uK8KX95$7BK8kT8Vm8dW7uR7J)7hd7}&8b!82t8a*7,y9$C71a8Z78>E9B48Gf7,@8}28tP7Ty8)~7Oc80H88G8Dx9NH:*@9!586;9y_7Ul9G^90/93v8449S,97_6_T93J88q8nx6yx9037d*8QG8/S6jr7fp7g@7nW8dT8{A6{Q8WZ7{B7R~8P'8bo8M=8^<8oi7!R8Z977&8'57kD8HJ8,N8398je76G7L08)B8H*7o88?w7B(80@7xo8r+7F&8]@7Yw7nt93q7o398r:4f8(]8}f"
"8z1:MO8[f8C~6OQ7wZ76v88B7JX8xp8&S8C=8,I6P47>@90k:J'5dd5aO7y|8R#8KW8-L8CF8Oy9mX60+7g|8Kz8PU66a9`V8'$8Mb8il82|7b]7YS8OM6t{7yY7iG9D39`,8207=j99=8=H8WL9uN5+Y7Hf8Ub8}f9o#8au7jE8(J:cv7eK7KE9d876f83s8XN9Si8.B9)&6^V9*u9$L7O281`8Nh:@e9#18R@7;a7vC8nY6qH7Cj9Ad6Lp7)N7Y>8+r7Z;9::8]*8538W(8|U7nj8zu6|'7]A8^,7::6/S"
"85D8i:7l]8bq7^17dL7g-7~:8!99JJ7tE8n67F(8#'7|}7L/8y*8b^6G18lC7/:82B7bJ7O?8p(8T{7hi8'78Kj9c)96c9Va7yt8v07EL8R{8IR74$9E'7VR82?8gr8lk7p=8Bk7@]6t98pb7T57QZ8@t8B78h;6Z!8_D9=[4M.8m?7qe7!T8f#8$t7w<8y#67587w72@7J-7cx8q687-8J>8?g5q49*N7lv8Qt8ac8=|9.88QX8LJ9F78w/7ks9fo76_7<[7w/8X(4^&9/q7Oi8E?8L]9G[9!.7R(7a=7of"
"8zc7/B9To8sh4vA8.D8or9V<7d>7M&8Sp8WZ8My7AX7`47G(74F9*k:bb9y^78F7UB:ae93;7Af7}q8DU7nO9/<8k66(<6uQ7=l9Ra9WF8Xe8,T7U9:9^8Q)8^h8zq8P795m7F78iC8`V7TQ7d{6[*9*Y8a764n7h;74W8?}77;8xP9PE6]'8658[#8>|8vI7R18`16=082E8bw7UV87S77b8+Q6T(7~j8EK8qT9/?7.h8zi86R6V-67v8.K8N/7Uf8<}6[Q7CC7RS6I)73L8R+8[{7k+7uJ8-L8+C87a8B+"
"8Oy8ui7o$7{}8&X8.B7{47wV8Hg9Y)7$-7{A7}T7uj8&f8&p86m8Ko4?f7nm7WW77R7f#8$y79+6DP+UQ7Nu7O(7X<7tV8*A7vL8]L6};7tT7bg8eh7v)8+g8o^83(7iy8K=8j>8[V7_[7dQ8@Y8g/8(38P48$F84!8#S87m8$B8+N7pt7z)7|/7z(7{f7cb7zO7g[7g?7y68*E8#n8(f8!97^y87$7qq7{q8){8;;8,Q8<I7t=8&#8k.7Mn7T!7sK8#)8=?7nU7Bb8Ry7Yu7`>7X#7pt7r67iZ8+[9ou8RG"
"7[v7|h8F18DM5kY6S]87[8eO7~E7gh7_>83x0u`59>6v}8@/8hd7|D7Ua7LQ8N(7Ah7h[82K7de7gj8#}7t'8g08Gk7j?7ld7XV7g(8,u88[8=l8_~8:'7y58bI8EY8/>8;E7s<8?<7]67O]7bN8hv7pm8/^8>T7}j8E37^g7j?7XS8Dg8L:94>7<&8Wi3^A8Ww8N!7mW8-o8388V'9&i6sB8]U8:?7_R7xD8Lj8Rz9LB7cj8[^7k;7Gh8348Q68Dp84c84{8JP8A;8Fk8dQ8+p8Cb84G7sv8KY7qt7{(9~^"
"8!w8I88&Z8(Y7ji4p?2;W7zm8#g88m7Uy6z_6p91.$.bA6s78-d8;m7s_6ym7(Q7$}6[x8I`80j8+v8'27f#8Tt7x?82{8D#8IY8Bv8,F80,81*8:,8-Z88n8(_86E8.B8-U8>/8?_8Ad7v88oi7tX7qd4h6,fr4xB9vD8&H7rw84(8RH6{T5258@g9(|80#8)B7{w88D8*I5Zd8L<89:88'8*a8/c80Z78K7Gw83n8''8(y8*H7zq7tz77)7BK7?K7vF8+B8!.80N8&P8-X7|s7jm7em85K8+}8787rE7}b"
"8,/8*+84p7|G7{m8*r7~>80Q8+/8'E7y(8-88*`95-7{l8W:82}8208Nc8G$9)^9,s6_k7o(8=!8$(82p8008`86rm*H61Dp8?K8MP8,(8+48`&7I*/pV7lo7W#7Z98)d80|8b-8uh7o87po7nY7nc88D8/x8Mw7yd88]8/u7`!7~K8,t8.m7]17}|89g8FP7oX7y?85P8*s8/b7nY8MF8>&83&7i(7lX8@o8JI8J^87=8*&8@h8CS8CO2A-88c8@V8G@8/-8++8&{7nm3Y=8)381-7w#87L7u*8A~6>z0yN"
"80086Z7nv7S_7JE9iH7]H5)m81n8)K8MM7O]7^d7Mc8hQ6978>-85W84a8I48&-8BT8A/8$l8@z85b8N_85}7vn8(.8)g7}p8Go81<82f8327}S8*b8?r8IB7_77$n7k]7wA7bV7Z>6I}2,a7oN7';7d*7Rj7>u7ab4tQ3~P7bM7fM8&77dJ7uR7dX8X]5wO7|~7kC8.P7I.7N'8V,9oT79+8.J8&k7t|7_p7V,7>,8EN8?@8'o84E80{7^<8$67X*7Dv7wy7eV7Tr7ne8.w8L'86S7<K6xr7G(8*T8$G8@z"
"8497{x7oN7fb8UU8_89?I5]q7O|8GB5~P9cH5nM6WV9N,8047d}4^97ux8PS7Lx8[@6T}7N66b=85r7_j6Az8{587#7v<8Z!8Dv9;X5lp72s7@27Am7GR8J[8JQ8O68]b5JW7TO7x26aj7h/6h58.o6nZ7X(7Do9@<6d28.P5+c8jS8i^7hC6`^7H97Ua7jv7UH8Ig5rG7gj7s57md8Jc8V'8@G6q27>l7dY7}|86)8L28w;8]=9OD8[Y79r8/@8my8Go8SQ7Q#8V!9(o8$'7[N7rm7G}7_,7V66C[7Nf7d+"
"8Bg8vk8N:7LI7E.7X<90Z8d37~48LV8^{8Yo8(_7>/8+K8;d7tP87b8Dz8Q98sn7ge8AX7I87[08^q8Lh8U68mt8)v7uP7Ch6G56@!6697+f6Lf5GD4Gn9YW6C?6,g8#Z7}$5E44jr7Me5AR7q47v/7{67tg7U/6q-6;o5NF6x{8'z70&7lw8)14D,6FS7(+7H68-]7]27n'8(/8#Q7Kw3wB6yX7d28dI3(g8!,7N86pV7!<76f6rG7',8d/6)&7cH6;*7-]4n'32W76#7P66Xn8.y7*56Gr8;q7r}80987}"
"8-;8:M83Z8.S7w58::7iO7l@8(G8CI8./9(O7k38?w7Z)8;z8;?8yH80i9Bn7~m85*7yQ7S&7U<8Mc7qy8E68L&88r75)7R,7j^7Wk8[x8<W8mt8?Q8_~8oD93+8N,8BW8>J7Pf6q,6jb7,Q80,8Km8>T7p`7Wl7q07yU8Vd8E:8B=7eb8Cj8F17l=7v*7v(8Kh8Rd80`8:17uX7lk7lS8'-8cU84f8FY8@-7}J8[M8738D|8sU8$|82#81g8=z8/D89(7d@7oN6}J7_/7us8yL84L8fK7[b7{h7{|8Rz7k["
"7Y<6ON6Ix6FZ6N,8@-8n|85r8&U7}d8+48qU7~^83a8.Z8(E8yL70f7s782g8RL8448)979I8Mb8Hc8TG8o_8Wm8G>8=T6{Z7]=8'M8^=8IA7K~8Kd7K&8De7I57Yl8*j8y)7$-8*t7w|8wA7s;80N8Oo7n47]K6z>7u&9*|7v#7qi8?k7bc7^V7[W8'm8ns8IX7gZ7IQ7F87m|86q8/U8<K8eE8<]8'J8U|81Q7{?83#7N?8:G7||86'8oa8Ff7qr7oL7p-8`h8;N8!|7Y,8BV7|,7Wq8k*7h07i'7eC7?H"
"8FU7y97^d7_^7lV8Ky7L:7iL8hq7Xr7{$8]$7M]8UY7AB6.>6GH7L`8.d8J]7xW9#P8pk6Ih6(!63@8j^8Vu7u~89(8!28vM7_Y8(@8FB9#Y7{a7b?83f8J]8S`9&?83/7fh8]<82Y7sh7nr7l>7Sd8[a7x{89X8?t80}84}8548<[8AZ8'68&684>8)M81d8=`80782b80(86.87m84W8=|8F_89;83C8-w8?r8?(8Dd76?7K]8c`8)=82p8>Y85b8F27Cn7297:A7n]7}^82|8;68Lv7z/89W8j!8^U8;@"
"8=c8>z8A`8_s7NC6qc7bz8F[8Ps8A!8E_7M02}]!!!3+w7!F4qD1Do/E;2P*72X7nP8T58T<7e>7L}7</7ry8Tl8?58&N8'h80Y8?:8Hd8vA8-Y7<S8007{X8*)8&?8$J8QB8,66P:8'N8$'8.q82(7?U8Lk8>u8N18;_7~X81{7}O8#B7yB7|{7yo86C81m7qu8&~7me7~/87c7}:83`8.e7~A85k8!=8*#8;^84#8&z8+I7{a7m.6s;6SK6ts6b>5!t8Bz6Z56;67LB84E8<$7}-5!q4(Q5425BU7q'7i/"
"7Bd8UQ8?i7'j5I:8+p7$o7pG7m`59}8G|6/|5wS7UR7&/8&o7s<7fd8$66(88OT7DU7[87U08/Z7]h7R)6:h8iV8-q7N}7?O6@-6x]6Hh7:/6y06t<5N53:(5.C6DL7x(9RZ89W83H86M84@8,R7zq82+8>#8VP7go8:x8ae89$7}:7v+8+`8Q=86G8>)8:$7x87io8?K88!8.{8:D8FI7!C6v^8!'8Jh82'8+58E^8rU7'L6y_6uz8,08-t8?/9!J8]?8v~7Y07tO8_z82-8*=7>:6UP4Ao5hN6~q7lt8'."
"7w}8[e8iq7k^8X`8@)8E07TS8vl5uR6Gu5^(7Aw8GB5Qi9Mo5yq8L$8I#76G8Tz7]G8'O7(86~X8!*8*W7);7{#68s4e>4Lm6ik7[=7k{8&O85$8+I6YF5h[6ci8/[84I7~=8388(i7D75df6Tl7aS8BO7W'6*v8>y6,]3kl7807B&7]z8eq8+T6Ye6t58(P6:S7LU6!`7E-6EU8.z8WQ9`e7t17TZ8)E7qr80x7?]7!p7d17kD7[o83J7;!7pf8Af7j*7tB9/-7kb8{=8(H8f;8lN7g=8?M9.u7EI8;z7I."
"7E:7[x7HD7v-8}l7cW8'!7T17Dp7XW8P'8Av8XH8W^8Ab8g&80^7yy7}F8+;8Uz8N88=B8C:8&L8>A80?7Mr7{88,c7eW7_!7fi84b8Mr8*g8_&81Q8Oz8:88;@80D8?|8544>38MQ82-8Ga8AT8328@p8D92-G7g`8}I8d|8588S$8(m82+13c7e=7~q7[N7[?7AG8?z8464=F8657Ta7]07O?7Yp8=U83C7v_81t8XV8#r7|M8MD82_8/g8yR7yC8>/83V82:8>L8A.8;Z8LV81N84W8#y8)<8628/.8<="
"7ho8jD83N8/-7s{8Ba8RK8]#7VA9;29??7A46[k8WK8E28F=7e68:O8/Z2bI.9o7Q>8?883|8*K7k&7?Z7ig7kE8o-8]J81l8UN8RW86i7yR7pT7ok8Mc8<u8<<8*97v57W78(g8+t8A~7x`7xC80=8D/7tY7qj7}n8&o81|8&y8#(7jD7yk8B$7nu84n8Fx7|u8*/7_a7s'7u;7vT7sS8#j8kD8$p2kq6j=7r;7~W8&b8+n6ke5g,.;|6[T6P/7vn7t.7t-7z.73?3oM80E8(`7S*8147l&8)+7ki3<,8#T"
"8()8&c7~+7~R7RF7hA4sc5f#8Y*8!z7nK7nh8P$8#u7KB81?8?18!=8,'7{D8<,8&(7[f8,p8(e8+f8&^7~&8!I55E6<k5_}6Rt6{[4nh6-h4aI6B~6rC6Wy5TT73z6dd8;C6L*5{Z5`(6|A7H{7R!7@(7IZ5Fs5iQ6#]8*<8+t87q7Te7jY7l]4~B7R_7kA7oi8(v62s6{z6FQ7-d70k7$07s07yI5i'6iH5<P7Ff6h:8$s7DE7u07DX7s$64f7ww6LT7Xy7.U7Zt73$6,h7lF7z97l<8K>8778?Q8-D6c["
"7r+7i27WI7l?7YE8x88nY8';8BZ8Y58X+8C~7R@9;I7jF7:)8-o9?(8Rc6Pl6vE9/h8-k83;7rr8u77F-6SU6T26`F8$E8#X8-D7tv7xp90X8XW7Tz80(8ix8.X7d,7U:8Lj7eI7xs7p38rj8>27Y37ik84b82!7mN7go7|87z/8`y8'N89R8B?8O]8#>8<r7z+84Z8*$8M(8a#8^e7xx7n+7mz8,E86r7nS7bb8<58'R8,b8/O8..8S<7=E7Sr7Y{8(*7k]87A7RC6Qz7;27DU7SE8Zr7bI8sZ8'm7V&3vz"
"3>a7P.8AX8E*8Hm9*f8xr7&S6~b7ZE85)87{6_78V{7~:7>+7Wd8m~9'E85j8'm8>Q7g17V|7zF8wm8:m7x]8)S8198?)8n&9!.8S!8@L7vf8:v8'l8Uj7QB7.(7{-6_*6n}8387$#8#h71!7?V8>?87:8PK8GH7h(8-)71}7:879^8Sb8AV7w28<'8Q48P97^-8Hs8)v81c7Ru8&(7s{7zz8OS8f=8Dy7fn8)g8_O8vo8cV8!G7iY7q|8*]8/U8@J8/17~R8.w8/K8(F8FQ8(c7~=82182A8#(8-y8;U8nW"
"81X86f88:8,{7{U83g8,Y8M37|g8);8*n7ZG7U`8ET8v:8+(8.$88185k7MD7T@7iu8U$5E!8-h8+282^7tc84^7ub7!C.[.8*~8,{8&t8#:8)S8JZ7mh3&u8<17xy7|m8$.7wf7gb88n3718+L8Lp7~07X(7}W7Z27UF8ej7N-7vv7gu76y8:;8+b7mJ7[+7w+7wo8R~8.47V67`<8RH7ww80M8K!8Ew71/74_6kG7^(76h8D08]-7kr7E-78k7AH7:x78K8Zs7x|7ai7Vv8Om8t-8:A7@s8K58.l7wD8K~"
"85B8Y48/38=k8&r8Kg8|(9;C8Ld8z17|v7|r8Ke8<_8-,7}T8,67~L8,D86I8ji8by89'7x`8)_7sh8+V82W7IQ80P8<28C.8[H8/=7{J8):6I>7Nm7^*7QH7V)8!}8!`8(8)y860D6;a7Uu7lU7a:83|80h+s+7ZX6V-8[>81:7|u8,I8/~61E8eN8#28Pu8,28.@8;D8?V7jY8'k8/P8@Q8$z8,W81^8GZ78w5u.52T8@s7TP6y[5|{7?N6KC7TW4^06Lm4F{9Tw8U&7e<7/<9!W8fm6br8:Q7U!6b85f'"
"4(H81i7]f87581R5]F8Xf7P/6/}7{m6l97}~8*e8;f6zw7;C7yx73F7p:8k/7pm7UI8li89S6;:78/7H|6[]86d8)v6@25HH5pR7}j6i26]h7Q_5Ip5je7ho7{<8*r8$h7zE7tr7vl8$e88h7m28$S7z?8&p8237xe7s]7yj7y'8:u8H78*:8KS7e?88D8,r7r>7}l7k|7i<7YZ81h7l[8)(7C/6}u87+7oi7nf7rf7wX8;Z4s;5#T7@g9&|8I97Z08'e80w0430t#7#:7kF8)>7yo7fn8:c41E5c$7`C7~Q"
"8J+8-r7~a8'q87]7xp8DQ8Km7~I7z>8$I7wE7ap8838(&8YY8$Y7pI8'`8)V8>V8(W7}Y8=z82V8#,8;+8.48KQ8&x7](7la7fv80k7~?8G;83'7[i5a?7{x7~H7o+7y=83/7'm7:l,nm3pY7H;8Ul81I8:d8hr8]36}e7/s7s@8-A7~U84T8H^8Jq7<{8Fx8RL8Gf83{8/V7`P7LL:>{7l17i>8/i8#(8=*7zN7ye7SS7A(81)8.Y8-s7|?7{U7r|7eQ6sN7Fi7f27c~7i;8#$8)_7pM7pI7be7Ap7rj7yx"
"7gX7^27Vn8(87g?7o97h97tI7|.7n>7qz71n7p)8'57vf7hS7n-7kn7u07l18,37y67jl7eA7}s7i`7o=7md7r)7wn7s-7{E5OK5.'55M6^;7<W7tx4b/7$76$x7tN3rS6u;8^'7y]6D?4pH5(,6y277f7oJ6m~7_E7wy70Y4^k7IT8Rt7|P7iO2L>8M$7r[6Z(7-=7<c7hf7~=7kU7k)8>b4ea5x{7J97Ge8k*7Tv8Z?80X4nl6Q[5<_7*K91m6b{5SR6Zu4PL66+5dB6Wp7Du7$n5=D8-N82F8F_8FO8D;"
"85{8N58Cv80m8QL85Y88@8Jr8=p8338<n8@(8Bw8;`8XV8B&7Ph8-/8H-87m81f83[8=d7WM7G}7gQ85z8Av88n8*57`A7Qv7Co7FL8$S80w8ad8Aq8=:7yD7|B8V58*a83d8IO89r8-<8eC8<x8MR8/@88g3K^3o|1N@4DZ67!8A)8Pd8Lg8>x8BB85_86V8I58:h8EU8>.89P8NB8&w8)/8Bn8/P8J387E89v8H57}c7xK87=8@A87n80$8E}8A@8!07]97dO77u85$8*>8Gb86I8_A7lP7q77i&8D]8F("
"8$/89~7{282X.KA+>Z7:97pI8<:8BP8E17rn48W2J[7O)7l08J88C}8YW8oN6yS6U}93&97Q8!17i&7~Z8',7o27qn8/*8-N7vf7i58,H8#S7Y'7w*8#e7re7ux7z{7dI81m8*o7fs7~K8!F7yv8&L7w47CX7HX8VI8=v8Hz7qo7dc81|7Kj7L-7@j80X8EP7~87uN8*j7U?/p@3}q85s8m788i8&v8Cu8UV4Dv7Xv83U8](8a'8@;8^^8Oq78X9?{8vd85883u9'G8h=7~?5`!/,33Ji7jV8TJ8S}8HO8dE"
"7kF6Da7l;8G37h-7qm81p8YG8]m8@q8<k8C=8.d8.s8WJ7Qh7B|8uC82g8N682#8'F8437?'7D&7Kd7t489$8<!86'7|O7{q8L881+8;u8;288@8?E83T82L8.38<t8=D8EM8JG8!<8C#8H#8=C80881`8E$8#78287uo8)R8AM85y8/(8+J8Vz7X~8L'7nF7yz86!8@481(8g86y!7rG89q85g8:.8<k7|P7^a2ED6.{7Xl7h78?|8=@8696{q,Nz3Os7v)7c|7s|8(!8F!8?T5K^7=b6Xs8Ad8*B8.O898"
"8^&7Dl83X8>H81287u8*p8EE7qm7q.8(R8*S86^83(84~88@8@T7FO8'Y7nq7fe8_r7vZ7u/9?:8K28!f7z67XF7qc7Tl8#R9!>8CG7lQ7U<7zI7eP7~K8PX75B7'*40v7gC7Z)7-f7~n7}(7BG8W17p37P17FL7?^8*27w,8+D8;g8&+8;U8-?84g8I489p8{`8O38?!8-q7=>7qZ8A$8;<8&v7P67pl8:u8+w8#j8)e7Q;8@E7_G8*x8[@9*(8$98-z7NP7jP8M:8A/8<H8K.86]8f882A8'F8bX7fc7w8"
"8VB8P48n$7nZ7vz7O:6s`7YO7Y+7Ey8+08$~8Ma8?295@7jT7jm7Zc6j{7:#84-86F8tJ8I{7Zv8`A8[)86s7Y:7r[7_c7`78*f9Up8o~9:v8Sh8,.8#`7M|7LA8228sd84.82v8bd8JH8<k9N|9.181687F8678Cg8F_6pt8^]7n28S(7Hw82f8-g8lM82X9=T7c[8i+8/.8&P8*48_g7;)7<T7T~7n98,y7qe7t&7:Z7._7;l7,&8838B>8(~8k,8+}8JP86l7h>7~^8To7tf8QC8(R7P&8!I7uj8@982J"
"7:v8[D8LT8[s8cI7G57Ol7i?7t;8)I8&?8)T7y!7tp83.85D89W8(r81#8*z8+F7vG8.(8Q28;H88v8UK89J7rh86F8-_8Zt8<#8Jp8St7L87=U8PU8Mj8a=7y(8+n7a57;m7Cm7:G7*R7wr7Yk7o:7eA7`&1PL5sU8ji7*y8)37yZ87^8({85i9D_9id8PM7jU8F'8;I8)68BP6aY8Cp7>$84F7j_8P-8[T8808]'8fU8Kp88_8,|8/98<w7Hm7I487.8|:86R7y_8j_8)<7N~8Lo9*l80T8/27XA8G{7OW"
"7a573`8;!73_7Y<7Cf7^X7JC7?K7Ss8u)7i(75|7vS8Jj8<'7D|7an8Ra8.-7mV7iN7gN8g~8+f7Tf8#k87'9$38P#7}y8-~7x-7v98=m97[7:C7~&8,O8R98IT8+?8+Y7Wd7u-7I78;]8qv8dj8-s6ob7cM8=:8Wy8A:8ju8Le7aa8,/7|086F8FV84F7MH7Rz7)O7vL8!{8*J8)=8#&7O>7W+7H08R|8)F7~i7|}8<=96c8[$8WY9)H6jO8!P7-A7w*8;27^$7~h5w.6B&79f77|7O28#y7Yd78B5}'2wk"
"8CH8HF7by8'28f]81c8$28)Z8/z8KQ8):7Qe8ih8#F88{8/-8=P8x>82z7]B81z8;T8!G8*j7mT89@7`-7nb7iw:+p8rE8x@7u?8V48j37]o7Qk7[m7um8aW7X{7hM8I>8.Y8r27VI7Hu7-#7|38,i85b8>e8;L7.u8+x7LC8dI7v:8!W7nB8<m9fd8YT8[V8!d8!P8+|7v_87c8#;8#_87I8d<7^T7t47`#8E58>d89q8*o6]z.o82);2CH8)!88B88<84N4R{*^T1AR7p27tS8i97nP8DV8Tr5fs6A<7u0"
"8)S8+K7z68!+7fv8;58)?7y_8-d8#a85R81x7x>8!38218*W7~v81F7n'8)]8<r83G8-]8,78&a8>v87+88O6{G7T~7kw5+z7aK7t]6g'6L&8fR7cQ6Uy5W~7&B4lt8!B9.J7[r5$55H~81?8EB6hq4OW70r6903ba7FY8;#87b9*07538tm6Q[5u(8w^85N83>8-P5PC7aI7!h6cY5`C7Q{7y58Ce7VB7Z+8R$8-e7,06_}8bR7?z7cT7}Y6Ks8w46n08FA7v18AB6vR83?8B~84.7sU8&P7o}8$<7sf8Ov"
"8@{80M7{E8$s8$E87i8>h80M86'84/7ws7{,7s.7Y:7s67008/{83h89W7D'71w5{O5~83|L8*S84/8/)7;F7)A7'i-A^,z08)R8/J7tN88C7-w5ym3lG04f8/o87f8,T83$88q7uG8bw8qb8&[7x<7zu8/K8@d7h:7]Y9DO85Z87-7z^8!$8'R7}98)>8,v80382E8*m7~?7wE8#K8(_7~h8#!7px82@8(>7dL7g[81g8+F7z07}&8Nr7Em7FZ7ai83`8+67|z8Ah8::7D478,73W80289D8)`7uh7E?/a0"
"4u)8O$7gb7~H8#58P05z_+@r6F>7k#84Z83n9;N8}$65B4&V7wN7rE7xr8^G7|38/s87I8/+8*,81W6zj8)_8Fm81U8)j8.B8*'8T@6SM8*U82k8+i8'.82s8SO7vP7Yv8if87u8B-8CF7bj7aE0IU5da8MJ8!_8#|7u$7u57ph7_`.a67H?81v81o8U]8Y.6s42oV3I>7ja8A}8C+8HD8=C8,685Q7e]8n68L_8JC8@'8*f81:8E|86o94j83*82X8#r8P28E]8668?H7cX8Ee7g/8;j80o7n'8l~88d8T,"
"8NI7{07hR8E*7t}8mD8<38nH8M=8W`8gI7Af7/e7v67o(8Cp7la8[68wJ7.07-I7$.7y{8CC5xM9'b6eC6jb8F98u&7x.8>G6.-8ZP8{R7`z8b?8Vi7Y~7x;6@t8XV8XF8Em8D,8o07u:7zN8Jb7k[8$38[D7Ot9319$'8tt89v88-88V9G$71C7/k7|~6l88:]8'(8Oe8Xk53|2U-8=48:d8L^81P8,B7c{7{P.o`7fZ8=^8YQ8+v8U.7m(7g]7qg82C8GG8J'7yC8J=8GB8Sa7v'7|Q8?H7mt8-07~(8&g"
"8@38Np8<t7Z+8R07`R7}|7z68.$8Gs8;;7o|7U97s]89M8^}8kG60_7y|8{V83,8(]8,+7p97Ts3/{8S*7ji8@@8='8[{8(36|l5mK7TU7]a81g85x8AG81T8>U8N)8qh8d'7|S8/Y7n088n8=+80#8D77m-83b8/g8&_8W=8-#8K$81*84W84C8JW8*}7oE7u|8*g8!a8E~7[N8*37{?8A68$&8){8$i7zi83~8:t8(z7f|7zt7y38708qJ7u77vv8A28V#8cB7r682u7xo8>+8BX8X!7tk7tY7|z8>H8Z}"
"88.7KB7W/7/<76t8i08uM8R'8[u80m86J6{A7'E71o7:!7k^8+T7s{7is8!W9$D72472(9/}80t7zM8>88AY8Np7B27<)9FW83h8*F8,B8+K8A*7Hn78N8NE8lf6L6'Ev1z98*07sF7e!89R8*F8VJ1TG2bO7c=85:7s.8.K8&~7nY7A46]T7{_86G8&x8!'88+8DR7b@7#l71o8dm8/87}r7z68;v8&|76673]7#k7~N7y:7x18'?7t|7c_7nm8,y84]8/e84Q83~8-Q8/,7r-7sA8&C8/d8,q86B8(0854"
"8'e7un8.:80P/(O6i+7MC8+R8J/87:8$+7e~6m87y29WD9+)9-I7x480Z8)L7C!7XG7+T6X387J83c8LQ87l8Vc9?E9-m7J~7N|8,W81W82j81;8K)7tv7FH7I!7[b8/K8B}89.8E=7tc8.h8X08+77~^8X08V28.{82{8Zj8X.7yZ7y58A*7y<8.U87W85>8AG89J8&O7{^7y_8oG8PJ7>v6n)8cT8Ib8*h7Bv7FA738/@o-Y*74d8DS8*H84{84a7uf3Ht0i/8!88-a7~782?8.;8<g7cB7Ar8?H8+H8As"
"8Eq8.^8Hl7:@7]G7F^8FC82j85M8<l81c8V38638P483F8>+8=M8DN8=V8-g8Bi8#B8-N8A-8(M8Ps8Jm8/}8<X86+82?8<!8jB6s!8T08?^7{.8OM8tF7~r7y57]$7_-8e/7p~7V`8H+8oF7}x7tX7gg8x*8lB8$V7{y7nn7b=7ky6c<7Zz7O[8A78Ef89b9'S8j:7*w7F/7U87:F87O8HJ98W9$P7^>8Q17|{7~E7x/7ML8=W8HI8Hs8C]8>a7o.8-y7x56O#8Xh7{_8.d8@H8?v8(<8Up/9g4TE7[g6~)"
"7S.7Z85tt6i,6WY3Te7t17J,8>-7T*6T>8Oa7:{7/{8`C6v[8-(8Wx8.x8E'78v7zN7~_7c,7`F8qz7y|8Mz7GQ7{<7H/7^~7rp7kF80@8407a*7v885L7Zk7^i92k8C,8EV7=V6XC8:h8#48E{8297J&8?s6A@7wU7xX8,]8++8?}8:#7:S6bJ6_(7h/83-7wP7{b81382v4JQ0Ww6[~7m$7p(8$q8F87}V5+14w;6Ad79n7q@8'{7qE8$&7295fc85+7|$7{K7r#7{j7xt8?A7|n9@:82|8-K82=7yt7|T"
"87$8?Q8+E8T|7pT80!8!A8$Q8$O89]8<O7yK8!_7}B7qu80!8+w8;o8'U8!A7xF88V8/]8=X8-67qR7x&7}h7zE7Jh87#8Ds8)882|7xO7MK68/1LC5l57?V81(80M7qE6is2Uj2iJ0,10aQ8*f7j}81b77c7KH7lz8@n7{N8:b7}r8!*7Qi7Dh7Lh8H{8:x8/+80C7pK7U47}a84U8Cu7s;80a8!J8(681+8:U81'8C#8'q8-Y8=t8#F8-)85d8)u8@K8I88e/9im7l^8x~7A]8G`8*.87#92]7It8tc9W="
"8;~9xH8fl7xX8G|7Ma7Vi:4`9ZZ8(_8)i7a/9St9DB9wE7A37CT87,8De7J(77)7qI8~Z7<y7=O7?W7H,8Kn8M{8A39u[8+]8Xv8BB8K&7_C7~r7tN8au7y18tV8DO8kj7jC8t[6-07;07/n8F-7jT7m(7a(8:G88=8DX88K8G&8:^8Fd8Ig88=84e8Fb8HC88R8EC8HC88c8:s";


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
    for (i = 0; i < 20000; ++i)
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
    mcts_param.used_idx = 0;
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
            double prev_value = get_val_hash(mcts_param.prev_nodes, mcts_param.nodes[idx].board, calc_hash(mcts_param.nodes[idx].board));
            if (prev_value != -inf)
                mcts_param.nodes[idx].pv = c_prev * prev_value;
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
        int a_cell = -1;
        value = -inf;
        double tmp_value;
        double t_sqrt = mcts_param.sqrt_arr[mcts_param.nodes[idx].n];
        for (const int& cell : search_param.vacant_lst){
            if (mcts_param.nodes[idx].p[cell] != 0.0){
                if (mcts_param.nodes[idx].children[cell] != -1)
                    tmp_value = c_puct * mcts_param.nodes[idx].p[cell] * t_sqrt / (1 + mcts_param.nodes[mcts_param.nodes[idx].children[cell]].n) - mcts_param.nodes[mcts_param.nodes[idx].children[cell]].w / mcts_param.nodes[mcts_param.nodes[idx].children[cell]].n - mcts_param.nodes[mcts_param.nodes[idx].children[cell]].pv / mcts_param.nodes[mcts_param.nodes[idx].children[cell]].n;
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
            mcts_param.nodes[mcts_param.used_idx].pv = 0.5;
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
                mcts_param.nodes[mcts_param.used_idx].pv = 0.5;
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
    hash_table_init(mcts_param.prev_nodes);
    int num_registered = 0;
    for (i = 0; i < mcts_param.used_idx; ++i){
        if (mcts_param.nodes[i].n > 10){
            ++num_registered;
            register_hash(mcts_param.prev_nodes, mcts_param.nodes[i].board, calc_hash(mcts_param.nodes[i].board), mcts_param.nodes[i].w / mcts_param.nodes[i].n);
        }
    }
    cout << num_registered << " items registered" << endl;
    mcts_param.used_idx = 1;
    for (i = 0; i < board_index_num; ++i)
        mcts_param.nodes[0].board[i] = board_param.board[i];
    mcts_param.nodes[0].w = 0.0;
    mcts_param.nodes[0].n = 0;
    mcts_param.nodes[0].pv = 0.5;
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

extern "C" double first(){
    board_param.direction = 0;
    cout << "FIRST" << endl;
    cout << 4 << " " << 5 << " " << 50.0 << endl;
    return 1000.0 * (4 * hw + 5) + 50.0;
}

extern "C" int start_ai(int *arr_board, int evaluate_count){
    int i, j, board_tmp, ai_player, policy;
    char elem;
    unsigned long long p, o;
    int n_stones = 0;
    double rnd, sm;
    search_param.evaluate_count = evaluate_count;
    string raw_board;
    n_stones = 0;
    search_param.vacant_cnt = 0;
    for (i = 0; i < hw2; ++i){
        if (arr_board[i] == 0){
            raw_board += "0";
            ++n_stones;
        } else if (arr_board[i] == 1){
            raw_board += "1";
            ++n_stones;
        } else{
            raw_board += ".";
            ++search_param.vacant_cnt;
        }
    }
    cout << raw_board << " " << evaluate_count << endl;
    search_param.turn = 0;
    p = 0;
    o = 0;
    search_param.vacant_lst = {};
    if (board_param.direction == -1){
        if (n_stones == 4){
            return 2;
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
        }
    }
    for (i = 0; i < hw2; ++i){
        elem = raw_board[i];
        if (elem != '.'){
            ++search_param.turn;
            p |= (unsigned long long)(elem == '0') << board_param.turn_board[board_param.direction][i];
            o |= (unsigned long long)(elem == '1') << board_param.turn_board[board_param.direction][i];
        } else{
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
