#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include "ifnum/linearAlgebra/matrix.hpp"
#include "ifnum/linearAlgebra/indexGenerator.hpp"

using namespace ifnum::linearAlgebra;

// =======================================================
// blockyMultiply — multiplicação em blocos (BLAS-like)
// =======================================================
template <typename T>
Matrix<T> blockyMultiply(const Matrix<T>& A, const Matrix<T>& B, int BS = 64) {
    size_t n = A.rows();
    size_t m = A.cols();
    size_t p = B.cols();

    Matrix<T> C(n, p, (T)0);

    for (size_t ii = 0; ii < n; ii += BS)
        for (size_t jj = 0; jj < p; jj += BS)
            for (size_t kk = 0; kk < m; kk += BS) {

                size_t i_end = std::min(ii + BS, n);
                size_t j_end = std::min(jj + BS, p);
                size_t k_end = std::min(kk + BS, m);

                for (size_t i = ii; i < i_end; ++i)
                    for (size_t k = kk; k < k_end; ++k) {
                        T aik = A(i, k);
                        for (size_t j = jj; j < j_end; ++j)
                            C(i, j) += aik * B(k, j);
                    }
            }
    return C;
}

// =======================================================
double relu(double x) { return x > 0 ? x : 0; }
double d_relu(double x) { return x > 0 ? 1 : 0; }
double linear(double x) { return x; }
double d_linear(double) { return 1; }

void xavier_init(Matrix<double>& M, int fan_in, int fan_out) {
    static std::mt19937 gen(std::random_device{}());
    double sd = std::sqrt(2.0 / (fan_in + fan_out));
    std::normal_distribution<double> dist(0.0, sd);
    for (size_t i = 0; i < M.rows(); ++i)
        for (size_t j = 0; j < M.cols(); ++j)
            M(i, j) = dist(gen);
}

double sum_all(const Matrix<double>& M) {
    double s = 0;
    for (size_t i = 0; i < M.rows(); ++i)
        for (size_t j = 0; j < M.cols(); ++j)
            s += M(i, j);
    return s;
}

// =======================================================
//  MLP com blockyMultiply
// =======================================================
int main() {
    const int BATCH = 256;
    const int EPOCHS = 400;
    const double LR = 0.01;

    const int N_IN = 2048;
    const int H1 = 1024;
    const int H2 = 512;
    const int H3 = 256;
    const int H4 = 128;
    const int H5 = 64;
    const int N_OUT = 1;

    Matrix<double> W1(H1, N_IN), W2(H2, H1), W3(H3, H2),
                   W4(H4, H3), W5(H5, H4), W6(N_OUT, H5);

    xavier_init(W1, N_IN, H1);
    xavier_init(W2, H1, H2);
    xavier_init(W3, H2, H3);
    xavier_init(W4, H3, H4);
    xavier_init(W5, H4, H5);
    xavier_init(W6, H5, N_OUT);

    Matrix<double> X(N_IN, BATCH);
    Matrix<double> Y(N_OUT, BATCH);

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dist(-M_PI, M_PI);

    for (int b = 0; b < BATCH; b++) {
        double ang = dist(gen);
        double s = ang / M_PI;
        for (int i = 0; i < N_IN; i++) X(i, b) = s;
        Y(0, b) = std::sin(ang);
    }

    for (int epoch = 0; epoch < EPOCHS; epoch++) {

        // ================ Forward ================
        auto Z1 = blockyMultiply(W1, X);
        auto A1 = Z1.apply(relu);

        auto Z2 = blockyMultiply(W2, A1);
        auto A2 = Z2.apply(relu);

        auto Z3 = blockyMultiply(W3, A2);
        auto A3 = Z3.apply(relu);

        auto Z4 = blockyMultiply(W4, A3);
        auto A4 = Z4.apply(relu);

        auto Z5 = blockyMultiply(W5, A4);
        auto A5 = Z5.apply(relu);

        auto Z6 = blockyMultiply(W6, A5);
        auto A6 = Z6.apply(linear);

        // erro
        Matrix<double> E = Y - A6;
        double MSE = sum_all(E.hadamard(E)) / BATCH;

        // ================ Backprop ================
        auto dZ6 = Z6.apply(d_linear);
        auto d6 = E.hadamard(dZ6);

        auto dW6 = blockyMultiply(d6, A5.T());

        auto d5 = blockyMultiply(W6.T(), d6).hadamard(Z5.apply(d_relu));
        auto dW5 = blockyMultiply(d5, A4.T());

        auto d4 = blockyMultiply(W5.T(), d5).hadamard(Z4.apply(d_relu));
        auto dW4 = blockyMultiply(d4, A3.T());

        auto d3 = blockyMultiply(W4.T(), d4).hadamard(Z3.apply(d_relu));
        auto dW3 = blockyMultiply(d3, A2.T());

        auto d2 = blockyMultiply(W3.T(), d3).hadamard(Z2.apply(d_relu));
        auto dW2 = blockyMultiply(d2, A1.T());

        auto d1 = blockyMultiply(W2.T(), d2).hadamard(Z1.apply(d_relu));
        auto dW1 = blockyMultiply(d1, X.T());

        double alpha = LR / BATCH;

        W1 += dW1 * alpha;
        W2 += dW2 * alpha;
        W3 += dW3 * alpha;
        W4 += dW4 * alpha;
        W5 += dW5 * alpha;
        W6 += dW6 * alpha;

        if (epoch % 10 == 0)
            std::cout << "Epoch " << epoch << " | MSE: " << MSE << "\n";
    }

    return 0;
}
