#pragma once
#include <iostream>

using namespace std;

using uf16 = uint_fast16_t;
using uf8 = uint_fast8_t;
using f8 = int_fast8_t;
using ull = unsigned long long;

#define inf 100000000
#define n_phases 4
#define phase_n_stones 10

#define n_line 6561
#define hw 8
#define hw_m1 7
#define hw_p1 9
#define hw2 64
#define hw22 128
#define hw2_m1 63
#define hw2_mhw 56
#define hw2_p1 65
#define black 0
#define white 1
#define vacant 2

bool global_searching = true;

inline int pop_count_ull(unsigned long long x){
    x = x - ((x >> 1) & 0x5555555555555555ULL);
	x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
	x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
	x = (x * 0x0101010101010101ULL) >> 56;
    return (int)x;
}

inline int pop_count_uchar(unsigned char x){
    x = (x & 0b01010101) + ((x & 0b10101010) >> 1);
    x = (x & 0b00110011) + ((x & 0b11001100) >> 2);
    return (x & 0b00001111) + ((x & 11110000) >> 4);
}

inline unsigned long long mirror_v(unsigned long long x){
    unsigned long long a = x & 0b0101010101010101010101010101010101010101010101010101010101010101ULL;
    unsigned long long b = x & 0b1010101010101010101010101010101010101010101010101010101010101010ULL;
    x = (a << 1) | (b >> 1);
    a = x & 0b0011001100110011001100110011001100110011001100110011001100110011ULL;
    b = x & 0b1100110011001100110011001100110011001100110011001100110011001100ULL;
    x = (a << 2) | (b >> 2);
    a = x & 0b0000111100001111000011110000111100001111000011110000111100001111ULL;
    b = x & 0b1111000011110000111100001111000011110000111100001111000011110000ULL;
    x = (a << 4) | (b >> 4);
    a = x & 0b0000000011111111000000001111111100000000111111110000000011111111ULL;
    b = x & 0b1111111100000000111111110000000011111111000000001111111100000000ULL;
    x = (a << 8) | (b >> 8);
    a = x & 0b0000000000000000111111111111111100000000000000001111111111111111ULL;
    b = x & 0b1111111111111111000000000000000011111111111111110000000000000000ULL;
    x = (a << 16) | (b >> 16);
    a = x & 0b0000000000000000000000000000000011111111111111111111111111111111ULL;
    b = x & 0b1111111111111111111111111111111100000000000000000000000000000000ULL;
    return (a << 32) | (b >> 32);
}

inline unsigned long long white_line(unsigned long long x){
    unsigned long long res = 0;
    int i, j;
    for (i = 0; i < hw; ++i){
        for (j = 0; j < hw; ++j){
            res |= (1 & (x >> (i * hw + j))) << (j * hw + i);
        }
    }
    return res;
}

inline unsigned long long black_line(unsigned long long x){
    unsigned long long res = 0;
    int i, j;
    for (i = 0; i < hw; ++i){
        for (j = 0; j < hw; ++j){
            res |= (1 & (x >> (i * hw + j))) << ((hw_m1 - j) * hw + hw_m1 - i);
        }
    }
    return res;
}
