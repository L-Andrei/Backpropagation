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

// ==========================================
// 1. Funções Auxiliares
// ==========================================

// Inicialização (Com semente fixa 42)
template <typename T>
void randomize(Matrix<T>& m, double min_val, double max_val) {
    std::mt19937 gen(42); 
    std::uniform_real_distribution<T> dis(min_val, max_val);
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            m(i, j) = dis(gen);
        }
    }
}

// === Função para somar Bias (Broadcast) ===
template <typename T>
void add_bias_broadcast(Matrix<T>& M, const Matrix<T>& B) {
    assert(M.rows() == B.rows());
    for (size_t i = 0; i < M.rows(); ++i) {
        T bias_val = B(i, 0); 
        for (size_t j = 0; j < M.cols(); ++j) {
            M(i, j) += bias_val;
        }
    }
}

// Soma Gradientes do Batch
template <typename T>
Matrix<T> sum_batch_gradients(const Matrix<T>& S) {
    Matrix<T> dB(S.rows(), 1);
    for (size_t i = 0; i < S.rows(); ++i) {
        T sum = 0.0;
        for (size_t j = 0; j < S.cols(); ++j) {
            sum += S(i, j);
        }
        dB(i, 0) = sum;
    }
    return dB;
}

// Funções de Ativação
double logsig(double x) { return 1.0 / (1.0 + std::exp(-x)); }
double dlogsig(double x) { double s = logsig(x); return (1.0 - s) * s; }

double act_tanh(double x) { return std::tanh(x); }
double d_act_tanh(double x) { double t = std::tanh(x); return 1.0 - t * t; }

double act_linear(double x) { return x; }
double d_act_linear(double x) { return 1.0; }

template <typename T>
T sum_squared_error(const Matrix<T>& diff) {
    T sum = 0.0;
    for(size_t i=0; i<diff.rows(); ++i) {
        for(size_t j=0; j<diff.cols(); ++j) {
            sum += diff(i,j) * diff(i,j);
        }
    }
    return sum;
}

// ==========================================
// MAIN: MLP 4 Camadas (Operador *)
// ==========================================
int main() {
    const int SIZE = 512; 
    
    const int N_INPUT = SIZE; 
    const int N_L1    = SIZE; 
    const int N_L2    = SIZE; 
    const int N_L3    = SIZE; 
    const int N_L4    = SIZE; 
    
    const int BATCH_SIZE = SIZE; 
    
    const int EPOCHS = 1000;
    const double LR = 0.05; 

    std::cout << "=== REPLICA MLP.M (OPERADOR *) ===" << std::endl;

    // --- Matrizes de Peso ---
    Matrix<double> W1(N_L1, N_INPUT);
    Matrix<double> W2(N_L2, N_L1);
    Matrix<double> W3(N_L3, N_L2);
    Matrix<double> W4(N_L4, N_L3);

    // --- Vetores de Bias (Nx1) ---
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

    // --- Dados ---
    Matrix<double> P(N_INPUT, BATCH_SIZE); 
    Matrix<double> T(N_L4, BATCH_SIZE);    

    randomize(P, -1.0, 1.0); 

    // Calcula o Target
    for(size_t i=0; i<P.rows(); ++i) {
        for(size_t j=0; j<P.cols(); ++j) {
            double valor_real = P(i,j) * 3.1415926535; 
            T(i,j) = std::sin(valor_real);
        }
    }

    std::cout << "Iniciando Treinamento..." << std::endl;
    auto start_total = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto start_epoch = std::chrono::high_resolution_clock::now();

        // -------------------------------------------------
        // 1. FEED FORWARD (Com Operador *)
        // -------------------------------------------------
        
        // Layer 1
        Matrix<double> N1 = W1 * P; 
        add_bias_broadcast(N1, B1); 
        Matrix<double> A1 = N1.apply(act_tanh);

        // Layer 2
        Matrix<double> N2 = W2 * A1; 
        add_bias_broadcast(N2, B2);
        Matrix<double> A2 = N2.apply(act_tanh);

        // Layer 3
        Matrix<double> N3 = W3 * A2; 
        add_bias_broadcast(N3, B3);
        Matrix<double> A3 = N3.apply(act_tanh);

        // Layer 4
        Matrix<double> N4 = W4 * A3; 
        add_bias_broadcast(N4, B4);
        Matrix<double> A4 = N4.apply(act_linear);

        // -------------------------------------------------
        // 2. CALCULO DO ERRO
        // -------------------------------------------------
        Matrix<double> E = A4 - T; 
        double sse = sum_squared_error(E);

        // -------------------------------------------------
        // 3. BACKPROPAGATION (Com Operador *)
        // -------------------------------------------------
        
        // S4
        Matrix<double> dN4 = N4.apply(d_act_linear); 
        Matrix<double> S4 = E.hadamard(dN4); 

        // S3
        Matrix<double> W4_T = W4.T();
        Matrix<double> error_prop_3 = W4_T * S4; 
        Matrix<double> dN3 = N3.apply(d_act_tanh);
        Matrix<double> S3 = error_prop_3.hadamard(dN3);

        // S2
        Matrix<double> W3_T = W3.T();
        Matrix<double> error_prop_2 = W3_T * S3; 
        Matrix<double> dN2 = N2.apply(d_act_tanh);
        Matrix<double> S2 = error_prop_2.hadamard(dN2);

        // S1
        Matrix<double> W2_T = W2.T();
        Matrix<double> error_prop_1 = W2_T * S2; 
        Matrix<double> dN1 = N1.apply(d_act_tanh);
        Matrix<double> S1 = error_prop_1.hadamard(dN1);

        // -------------------------------------------------
        // 4. UPDATE (Com Operador *)
        // -------------------------------------------------
        double alpha = LR / BATCH_SIZE; 

        // Update L4
        Matrix<double> A3_T = A3.T();
        Matrix<double> dW4 = S4 * A3_T; 
        Matrix<double> dB4 = sum_batch_gradients(S4); 
        W4 = W4 - dW4 * alpha;
        B4 = B4 - dB4 * alpha;

        // Update L3
        Matrix<double> A2_T = A2.T();
        Matrix<double> dW3 = S3 * A2_T;
        Matrix<double> dB3 = sum_batch_gradients(S3);
        W3 = W3 - dW3 * alpha;
        B3 = B3 - dB3 * alpha;

        // Update L2
        Matrix<double> A1_T = A1.T();
        Matrix<double> dW2 = S2 * A1_T;
        Matrix<double> dB2 = sum_batch_gradients(S2);
        W2 = W2 - dW2 * alpha;
        B2 = B2 - dB2 * alpha;

        // Update L1
        Matrix<double> P_T = P.T();
        Matrix<double> dW1 = S1 * P_T; 
        Matrix<double> dB1 = sum_batch_gradients(S1);
        W1 = W1 - dW1 * alpha;
        B1 = B1 - dB1 * alpha;

        // Log
        auto end_epoch = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_epoch - start_epoch;
        
        std::cout << "Epoca " << epoch+1 
                  << " | SSE: " << sse 
                  << " | Tempo: " << elapsed.count() << "s" << std::endl;
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> final_time = end_total - start_total;
    std::cout << "Tempo Total: " << final_time.count() << "s" << std::endl;

    return 0;
}