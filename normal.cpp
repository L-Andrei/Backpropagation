#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>

// Seus arquivos de cabeçalho
#include <ifnum/linearAlgebra/matrix.hpp>
#include <ifnum/linearAlgebra/indexGenerator.hpp>

using namespace ifnum::linearAlgebra;

// ==========================================
// Utils
// ==========================================

template <typename T>
void randomize(Matrix<T>& m, double min_val, double max_val) {
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<T> dis(min_val, max_val);

    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            m(i, j) = dis(gen);
        }
    }
}

template <typename T>
T sum_all_elements(const Matrix<T>& m) {
    T total_sum = 0.0;
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            total_sum += m(i, j);
        }
    }
    return total_sum;
}

// Funções de Ativação
double act_tanh(double x) { return std::tanh(x); }
double d_act_tanh(double x) { double t = std::tanh(x); return 1.0 - t * t; }
double act_linear(double x) { return x; }
double d_act_linear(double x) { return 1.0; }

// ==========================================
// MLP - Operator Overloading Benchmark (A * B)
// ==========================================
int main() {
    // ---------------------------------------------------------
    // Configurações (Mesma Carga do Teste Otimizado)
    // ---------------------------------------------------------
    const int N_IN  = 2048; 
    const int N_H1  = 2048; 
    const int N_H2  = 2048; 
    const int N_OUT = 1;    

    const int BATCH_SIZE = 256; 
    // ATENÇÃO: Reduzido para 5 épocas pois a multiplicação padrão é lenta
    const int EPOCHS = 5;      
    const double LR = 0.0005;   

    const double N_ELEMENTS = static_cast<double>(N_OUT * BATCH_SIZE);

    std::cout << "=== BENCHMARK: OPERATOR * (STANDARD) ===" << std::endl;
    std::cout << "Matrizes: " << N_H1 << "x" << N_IN << std::endl;
    std::cout << "Usando sobrecarga: A * B" << std::endl;

    // Pesos (W)
    Matrix<double> W1(N_H1, N_IN);
    Matrix<double> W2(N_H2, N_H1);
    Matrix<double> W3(N_OUT, N_H2);

    randomize(W1, -0.02, 0.02);
    randomize(W2, -0.02, 0.02);
    randomize(W3, -0.02, 0.02);

    // Dados
    Matrix<double> X(N_IN, BATCH_SIZE); 
    Matrix<double> Y(N_OUT, BATCH_SIZE);
    
    randomize(X, -3.14159, 3.14159);

    // Target: Seno da primeira linha
    for (size_t b = 0; b < BATCH_SIZE; ++b) {
        Y(0, b) = std::sin(X(0, b));
    }

    std::cout << "Iniciando loop (Aguarde, pode ser lento)..." << std::endl;

    auto start_total = std::chrono::high_resolution_clock::now();
    double final_mse = 0.0;
    Matrix<double> A3(N_OUT, BATCH_SIZE);

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        
        // --- FEED FORWARD (Usa operator*) ---
        
        // Z1 = W1 * X
        Matrix<double> Z1 = W1 * X; 
        Matrix<double> A1 = Z1.apply(act_tanh);

        // Z2 = W2 * A1
        Matrix<double> Z2 = W2 * A1;
        Matrix<double> A2 = Z2.apply(act_tanh);

        // Z3 = W3 * A2
        Matrix<double> Z3 = W3 * A2;
        A3 = Z3.apply(act_linear); 

        // --- CALCULO DO ERRO ---
        Matrix<double> E = A3 - Y;
        Matrix<double> E_squared = E.hadamard(E); 
        double SSE = sum_all_elements(E_squared);
        double MSE = SSE / N_ELEMENTS;
        final_mse = MSE;

        // --- BACKPROPAGATION (Usa operator*) ---
        
        // Delta 3
        Matrix<double> dZ3 = Z3.apply(d_act_linear); 
        Matrix<double> delta3 = E.hadamard(dZ3);

        // dW3 = delta3 * A2_T
        Matrix<double> A2_T = A2.T();
        Matrix<double> dW3 = delta3 * A2_T;

        // Delta 2 (Retropropagando erro) -> E2 = W3_T * delta3
        Matrix<double> W3_T = W3.T();
        Matrix<double> E2 = W3_T * delta3;
        
        Matrix<double> dZ2 = Z2.apply(d_act_tanh);
        Matrix<double> delta2 = E2.hadamard(dZ2);

        // dW2 = delta2 * A1_T
        Matrix<double> A1_T = A1.T();
        Matrix<double> dW2 = delta2 * A1_T;

        // Delta 1 -> E1 = W2_T * delta2
        Matrix<double> W2_T = W2.T();
        Matrix<double> E1 = W2_T * delta2;

        Matrix<double> dZ1 = Z1.apply(d_act_tanh);
        Matrix<double> delta1 = E1.hadamard(dZ1);

        // dW1 = delta1 * X_T
        Matrix<double> X_T = X.T();
        Matrix<double> dW1 = delta1 * X_T;

        // --- UPDATE ---
        double alpha = LR / BATCH_SIZE; 
        
        W1 = W1 - dW1 * alpha; 
        W2 = W2 - dW2 * alpha;
        W3 = W3 - dW3 * alpha;

        // Monitoramento
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - start_total;
        
        std::cout << "Epoca " << std::setw(3) << epoch + 1 
                  << " | MSE: " << std::fixed << std::setprecision(6) << MSE 
                  << " | Tempo decorrido: " << elapsed.count() << "s" << std::endl;
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> final_time = end_total - start_total;
    
    std::cout << "\n=== FIM DO TESTE (OPERATOR *) ===" << std::endl;
    std::cout << "Tempo Total: " << final_time.count() << "s" << std::endl;
    std::cout << "Tempo Medio por Epoca: " << final_time.count() / EPOCHS << "s" << std::endl;

    return 0;
}