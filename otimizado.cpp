#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <cassert>

// Seus arquivos de cabeçalho
#include <ifnum/linearAlgebra/matrix.hpp>
#include <ifnum/linearAlgebra/indexGenerator.hpp>

using namespace ifnum::linearAlgebra;

// ---------- Reuso de utilitários ----------
template <typename T>
void randomize(Matrix<T>& m, double min_val, double max_val) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<T> dis(min_val, max_val);
    for (size_t i = 0; i < m.rows(); ++i)
        for (size_t j = 0; j < m.cols(); ++j)
            m(i, j) = dis(gen);
}

template <typename T>
void add_bias_broadcast(Matrix<T>& M, const Matrix<T>& B) {
    assert(M.rows() == B.rows());
    for (size_t i = 0; i < M.rows(); ++i) {
        T bias_val = B(i, 0);
        for (size_t j = 0; j < M.cols(); ++j)
            M(i, j) += bias_val;
    }
}

template <typename T>
Matrix<T> sum_batch_gradients(const Matrix<T>& S) {
    Matrix<T> dB(S.rows(), 1);
    for (size_t i = 0; i < S.rows(); ++i) {
        T sum = 0.0;
        for (size_t j = 0; j < S.cols(); ++j) sum += S(i, j);
        dB(i, 0) = sum;
    }
    return dB;
}

// Ativações e derivadas
double act_tanh(double x) { return std::tanh(x); }
double d_act_tanh(double x) { double t = std::tanh(x); return 1.0 - t * t; }

double act_linear(double x) { return x; }
double d_act_linear(double x) { return 1.0; }

template <typename T>
T sum_squared_error(const Matrix<T>& diff) {
    T sum = 0.0;
    for (size_t i = 0; i < diff.rows(); ++i)
        for (size_t j = 0; j < diff.cols(); ++j)
            sum += diff(i, j) * diff(i, j);
    return sum;
}

// ---------- MAIN ----------
int main() {
    // --- parâmetros principais ---
    const int SIZE = 128;                     // dimensão da matriz original
    const int BATCH_SIZE = SIZE * SIZE;       // 262144
    const int N_INPUT = 1;                    // elemento a elemento
    const int N_L1 = 204;
    const int N_L2 = 204;
    const int N_L3 = 204;
    const int N_L4 = 1;                       // saída escalar por amostra

    const int EPOCHS = 1000;
    const double LR = 0.005;

    std::cout << ">>> MLP GRANDE (blockMultiply) 1->204->204->204->1, BATCH = " << BATCH_SIZE << std::endl;

    // --- pesos e bias ---
    Matrix<double> W1(N_L1, N_INPUT); // 204 x 1
    Matrix<double> W2(N_L2, N_L1);    // 204 x 204
    Matrix<double> W3(N_L3, N_L2);    // 204 x 204
    Matrix<double> W4(N_L4, N_L3);    // 1 x 204

    Matrix<double> B1(N_L1, 1);
    Matrix<double> B2(N_L2, 1);
    Matrix<double> B3(N_L3, 1);
    Matrix<double> B4(N_L4, 1);

    randomize(W1, -0.05, 0.05);
    randomize(W2, -0.05, 0.05);
    randomize(W3, -0.05, 0.05);
    randomize(W4, -0.05, 0.05);

    randomize(B1, -0.01, 0.01);
    randomize(B2, -0.01, 0.01);
    randomize(B3, -0.01, 0.01);
    randomize(B4, -0.01, 0.01);

    // --- dados ---
    Matrix<double> P_mat(SIZE, SIZE);
    randomize(P_mat, -1.0, 1.0);

    Matrix<double> P(1, BATCH_SIZE);
    Matrix<double> T(1, BATCH_SIZE);
    for (size_t i = 0; i < SIZE; ++i) {
        for (size_t j = 0; j < SIZE; ++j) {
            size_t idx = i * SIZE + j;
            double val = P_mat(i, j);
            P(0, idx) = val;
            T(0, idx) = std::sin(val * 3.14159265358979323846);
        }
    }

    Matrix<double> A4(N_L4, BATCH_SIZE);

    std::cout << "Iniciando treinamento (full-batch) com blockMultiply..." << std::endl;
    auto t_start_total = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // ---- FORWARD ----
        // Substituído W1 * P por blockMultiply(W1, P)
        Matrix<double> N1 = blockMultiply(W1, P);
        add_bias_broadcast(N1, B1);
        Matrix<double> A1 = N1.apply(act_tanh);

        // Substituído W2 * A1 por blockMultiply(W2, A1)
        Matrix<double> N2 = blockMultiply(W2, A1);
        add_bias_broadcast(N2, B2);
        Matrix<double> A2 = N2.apply(act_tanh);

        // Substituído W3 * A2 por blockMultiply(W3, A2)
        Matrix<double> N3 = blockMultiply(W3, A2);
        add_bias_broadcast(N3, B3);
        Matrix<double> A3 = N3.apply(act_tanh);

        // Substituído W4 * A3 por blockMultiply(W4, A3)
        Matrix<double> N4 = blockMultiply(W4, A3);
        add_bias_broadcast(N4, B4);
        Matrix<double> A4 = N4.apply(act_linear);

        // ---- LOSS ----
        Matrix<double> E = A4 - T;
        double sse = sum_squared_error(E);

        // ---- BACKPROP (Error Propagation) ----
        Matrix<double> dN4 = N4.apply(d_act_linear);
        Matrix<double> S4 = E.hadamard(dN4);

        // Substituído W4_T * S4 por blockMultiply(W4_T, S4)
        Matrix<double> W4_T = W4.T();
        Matrix<double> error_prop_3 = blockMultiply(W4_T, S4);
        Matrix<double> dN3 = N3.apply(d_act_tanh);
        Matrix<double> S3 = error_prop_3.hadamard(dN3);

        // Substituído W3_T * S3 por blockMultiply(W3_T, S3)
        Matrix<double> W3_T = W3.T();
        Matrix<double> error_prop_2 = blockMultiply(W3_T, S3);
        Matrix<double> dN2 = N2.apply(d_act_tanh);
        Matrix<double> S2 = error_prop_2.hadamard(dN2);

        // Substituído W2_T * S2 por blockMultiply(W2_T, S2)
        Matrix<double> W2_T = W2.T();
        Matrix<double> error_prop_1 = blockMultiply(W2_T, S2);
        Matrix<double> dN1 = N1.apply(d_act_tanh);
        Matrix<double> S1 = error_prop_1.hadamard(dN1);

        // ---- GRADIENTS e UPDATE ----
        double alpha = LR / double(BATCH_SIZE);

        // Substituído S4 * A3_T por blockMultiply(S4, A3_T)
        Matrix<double> A3_T = A3.T();
        Matrix<double> dW4 = blockMultiply(S4, A3_T);
        Matrix<double> dB4 = sum_batch_gradients(S4);
        W4 = W4 - dW4 * alpha; // Multiplicação por escalar mantida
        B4 = B4 - dB4 * alpha;

        // Substituído S3 * A2_T por blockMultiply(S3, A2_T)
        Matrix<double> A2_T = A2.T();
        Matrix<double> dW3 = blockMultiply(S3, A2_T);
        Matrix<double> dB3 = sum_batch_gradients(S3);
        W3 = W3 - dW3 * alpha;
        B3 = B3 - dB3 * alpha;

        // Substituído S2 * A1_T por blockMultiply(S2, A1_T)
        Matrix<double> A1_T = A1.T();
        Matrix<double> dW2 = blockMultiply(S2, A1_T);
        Matrix<double> dB2 = sum_batch_gradients(S2);
        W2 = W2 - dW2 * alpha;
        B2 = B2 - dB2 * alpha;

        // Substituído S1 * P_T por blockMultiply(S1, P_T)
        Matrix<double> P_T = P.T();
        Matrix<double> dW1 = blockMultiply(S1, P_T);
        Matrix<double> dB1 = sum_batch_gradients(S1);
        W1 = W1 - dW1 * alpha;
        B1 = B1 - dB1 * alpha;

        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = t1 - t0;

        if (1) {
            double mse = sse / double(BATCH_SIZE);
            double rmse = std::sqrt(mse);
            std::cout << "Epoca " << std::setw(4) << epoch+1
                      << " | SSE: " << std::fixed << std::setprecision(6) << sse
                      << " | RMSE: " << std::setprecision(6) << rmse
                      << " | Tempo epoca: " << elapsed.count() << "s"
                      << std::endl;
        }
    }

    auto t_end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = t_end_total - t_start_total;
    std::cout << "Treinamento finalizado. Tempo total: " << total_elapsed.count() << "s" << std::endl;

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(0, BATCH_SIZE - 1);
    double mae = 0.0;
    int SAMPLES = 1000;
    for (int k = 0; k < SAMPLES; ++k) {
        int idx = dist(rng);
        double pred = A4(0, idx);
        double target = T(0, idx);
        mae += std::abs(pred - target);
    }
    mae /= double(SAMPLES);
    std::cout << "MAE em " << SAMPLES << " amostras aleatorias: " << mae << std::endl;

    return 0;
}