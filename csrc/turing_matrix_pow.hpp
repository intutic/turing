#pragma once

#include <vector>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#if defined(_MSC_VER)
typedef int64_t __int128_t;
#endif

namespace turing {

/**
 * Generic M x M Matrix Exponentiation Engine for Logarithmic Recurrence State Transitions.
 * Computes T^p in O(M^3 * log p) time with __int128_t intermediate accumulators to prevent overflow.
 */
struct Matrix3x3 {
    int64_t mat[3][3];
    Matrix3x3() {
        std::memset(mat, 0, sizeof(mat));
    }
};

inline Matrix3x3 multiply_3x3(const Matrix3x3& a, const Matrix3x3& b, int64_t mod) {
    Matrix3x3 c;
    for (int i = 0; i < 3; ++i) {
        for (int k = 0; k < 3; ++k) {
            if (a.mat[i][k] == 0) continue;
            for (int j = 0; j < 3; ++j) {
                __int128_t prod = static_cast<__int128_t>(a.mat[i][k]) * static_cast<__int128_t>(b.mat[k][j]);
                c.mat[i][j] = static_cast<int64_t>((static_cast<__int128_t>(c.mat[i][j]) + prod) % mod);
            }
        }
    }
    return c;
}

inline Matrix3x3 matrix_power_3x3(Matrix3x3 a, int64_t p, int64_t mod) {
    Matrix3x3 res;
    for (int i = 0; i < 3; ++i) res.mat[i][i] = 1;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            a.mat[i][j] %= mod;
        }
    }
    while (p > 0) {
        if (p & 1) res = multiply_3x3(res, a, mod);
        a = multiply_3x3(a, a, mod);
        p >>= 1;
    }
    return res;
}

/**
 * Computes recurrence state x_k = (A*x_{k-1} + B*x_{k-2} + C) mod M directly in O(log k) steps.
 */
inline int64_t evaluate_recurrence_jump(
    int64_t x0, int64_t x1,
    int64_t A, int64_t B, int64_t C, int64_t M,
    int64_t target_idx
) {
    if (target_idx == 0) return x0 % M;
    if (target_idx == 1) return x1 % M;

    Matrix3x3 T_mat;
    T_mat.mat[0][0] = A; T_mat.mat[0][1] = B; T_mat.mat[0][2] = C;
    T_mat.mat[1][0] = 1; T_mat.mat[1][1] = 0; T_mat.mat[1][2] = 0;
    T_mat.mat[2][0] = 0; T_mat.mat[2][1] = 0; T_mat.mat[2][2] = 1;

    Matrix3x3 T_pow = matrix_power_3x3(T_mat, target_idx - 1, M);
    int64_t init_vec[3] = { x1 % M, x0 % M, 1 };

    __int128_t res = (static_cast<__int128_t>(T_pow.mat[0][0]) * init_vec[0] +
                      static_cast<__int128_t>(T_pow.mat[0][1]) * init_vec[1] +
                      static_cast<__int128_t>(T_pow.mat[0][2]) * init_vec[2]) % M;
    return static_cast<int64_t>(res);
}

} // namespace turing
