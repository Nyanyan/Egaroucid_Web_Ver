#pragma once
#include <iostream>
#include "level.hpp"
#include "midsearch.hpp"
#include "endsearch.hpp"
#include "book.hpp"

search_result ai_search(board b, int level){
    search_result res;
    int depth1, depth2;
    bool use_mpc;
    double mpct;
    get_level(level, b.n - 3, &depth1, &depth2, &use_mpc, &mpct);
    int book_policy = book.get(&b);
    cout << "book policy " << book_policy << endl;
    if (book_policy != -1){
        cout << "BOOK " << book_policy << endl;
        res.policy = book_policy;
        mobility mob;
        calc_flip(&mob, &b, book_policy);
        b.move(&mob);
        res.value = -midsearch(b, 0, depth1 / 2, use_mpc, mpct).value;
        res.depth = -1;
        res.nps = 0;
        return res;
    }
    cout << "level status " << level << " " << b.n - 3 << " " << depth1 << " " << depth2 << " " << use_mpc << " " << mpct << endl;
    if (b.n >= hw2 - depth2 - 1)
        res = endsearch(b, 0, use_mpc, mpct);
    else
        res = midsearch(b, 0, depth1, use_mpc, mpct);
    return res;
}
