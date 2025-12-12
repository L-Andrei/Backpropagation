#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <cassert>

// Seus arquivos de cabeçalho (mesma API que você usava)
#include <ifnum/linearAlgebra/matrix.hpp>
#include <ifnum/linearAlgebra/indexGenerator.hpp>

using namespace ifnum::linearAlgebra;

// ---------- Reuso de utilitários (adapte se já tiver) ----------
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

// ---------- MAIN (flatten + grandes multiplicações) ----------
int main() {
    // --- parâmetros principais ---
    const int SIZE = 128;                     // dimensão da matriz original
    const int BATCH_SIZE = SIZE * SIZE;       // 262144
    const int N_INPUT = 1;                    // elemento a elemento
    const int N_L1 = 204;
    const int N_L2 = 204;
    const int N_L3 = 204;
    const int N_L4 = 1;                       // saída escalar por amostra

    const int EPOCHS = 1000;                  // ajustar conforme necessidade
    const double LR = 0.005;                  // taxa de aprendizado

    std::cout << ">>> MLP GRANDE (flatten) 1->204->204->204->1, BATCH = " << BATCH_SIZE << std::endl;

    // --- pesos e bias ---
    Matrix<double> W1(N_L1, N_INPUT); // 204 x 1
    Matrix<double> W2(N_L2, N_L1);    // 204 x 204
    Matrix<double> W3(N_L3, N_L2);    // 204 x 204
    Matrix<double> W4(N_L4, N_L3);    // 1 x 204

    Matrix<double> B1(N_L1, 1);
    Matrix<double> B2(N_L2, 1);
    Matrix<double> B3(N_L3, 1);
    Matrix<double> B4(N_L4, 1);

    // inicialização pequena
    randomize(W1, -0.05, 0.05);
    randomize(W2, -0.05, 0.05);
    randomize(W3, -0.05, 0.05);
    randomize(W4, -0.05, 0.05);

    randomize(B1, -0.01, 0.01);
    randomize(B2, -0.01, 0.01);
    randomize(B3, -0.01, 0.01);
    randomize(B4, -0.01, 0.01);

    // --- dados: cria P_mat 512x512 e flatten para 1 x BATCH_SIZE ---
    Matrix<double> P_mat(SIZE, SIZE);
    randomize(P_mat, -1.0, 1.0);

    Matrix<double> P(1, BATCH_SIZE);
    Matrix<double> T(1, BATCH_SIZE);
    for (size_t i = 0; i < SIZE; ++i) {
        for (size_t j = 0; j < SIZE; ++j) {
            size_t idx = i * SIZE + j;
            double val = P_mat(i, j);       // ∈ [-1,1]
            P(0, idx) = val;
            T(0, idx) = std::sin(val * 3.14159265358979323846); // target
        }
    }

    Matrix<double> A4(N_L4, BATCH_SIZE);

    std::cout << "Iniciando treinamento (full-batch)..." << std::endl;
    auto t_start_total = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // ---- FORWARD ----
        // N1 = W1 * P  => (204 x 1) * (1 x BATCH) = 204 x BATCH
        Matrix<double> N1 = W1 * P;
        add_bias_broadcast(N1, B1);
        Matrix<double> A1 = N1.apply(act_tanh); // 204 x BATCH

        Matrix<double> N2 = W2 * A1;            // 204 x BATCH
        add_bias_broadcast(N2, B2);
        Matrix<double> A2 = N2.apply(act_tanh);

        Matrix<double> N3 = W3 * A2;            // 204 x BATCH
        add_bias_broadcast(N3, B3);
        Matrix<double> A3 = N3.apply(act_tanh);

        Matrix<double> N4 = W4 * A3;            // 1 x BATCH
        add_bias_broadcast(N4, B4);
        Matrix<double> A4 = N4.apply(act_linear); // 1 x BATCH

        // ---- LOSS ----
        Matrix<double> E = A4 - T; // 1 x BATCH
        double sse = sum_squared_error(E);

        // ---- BACKPROP ----
        // S4: 1 x BATCH
        Matrix<double> dN4 = N4.apply(d_act_linear);
        Matrix<double> S4 = E.hadamard(dN4);

        // S3: 204 x BATCH
        Matrix<double> W4_T = W4.T();           // 204 x 1
        Matrix<double> error_prop_3 = W4_T * S4; // (204x1)*(1xBATCH)=204xBATCH
        Matrix<double> dN3 = N3.apply(d_act_tanh);
        Matrix<double> S3 = error_prop_3.hadamard(dN3);

        // S2: 204 x BATCH
        Matrix<double> W3_T = W3.T();           // 204 x 204
        Matrix<double> error_prop_2 = W3_T * S3; // 204x204 * 204xBATCH = 204xBATCH
        Matrix<double> dN2 = N2.apply(d_act_tanh);
        Matrix<double> S2 = error_prop_2.hadamard(dN2);

        // S1: 204 x BATCH
        Matrix<double> W2_T = W2.T();           // 204 x 204
        Matrix<double> error_prop_1 = W2_T * S2;
        Matrix<double> dN1 = N1.apply(d_act_tanh);
        Matrix<double> S1 = error_prop_1.hadamard(dN1);

        // ---- GRADIENTS e UPDATE ----
        double alpha = LR / double(BATCH_SIZE); // mantém escala similar ao seu código original

        // dW4 = S4 * A3.T  => (1 x BATCH) * (BATCH x 204) = 1 x 204
        Matrix<double> A3_T = A3.T();
        Matrix<double> dW4 = S4 * A3_T;
        Matrix<double> dB4 = sum_batch_gradients(S4);
        W4 = W4 - dW4 * alpha;
        B4 = B4 - dB4 * alpha;

        // dW3 = S3 * A2.T  => 204 x 204
        Matrix<double> A2_T = A2.T();
        Matrix<double> dW3 = S3 * A2_T;
        Matrix<double> dB3 = sum_batch_gradients(S3);
        W3 = W3 - dW3 * alpha;
        B3 = B3 - dB3 * alpha;

        // dW2 = S2 * A1.T
        Matrix<double> A1_T = A1.T();
        Matrix<double> dW2 = S2 * A1_T;
        Matrix<double> dB2 = sum_batch_gradients(S2);
        W2 = W2 - dW2 * alpha;
        B2 = B2 - dB2 * alpha;

        // dW1 = S1 * P.T
        Matrix<double> P_T = P.T(); // BATCH x 1
        Matrix<double> dW1 = S1 * P_T; // 204 x 1
        Matrix<double> dB1 = sum_batch_gradients(S1);
        W1 = W1 - dW1 * alpha;
        B1 = B1 - dB1 * alpha;

        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = t1 - t0;

        if (1) {
            // cálculo RMSE médio por elemento para interpretação
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

    // --- Avaliação rápida: pega 1000 amostras aleatórias e compara com sin ---
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
