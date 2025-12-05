#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>

// Seus arquivos de cabeçalho
#include "ifnum/linearAlgebra/matrix.hpp"
#include "ifnum/linearAlgebra/indexGenerator.hpp"

using namespace ifnum::linearAlgebra;

// ==========================================
// Utils (Adaptados para Matrix)
// ==========================================

// Transposta
template <typename T>
Matrix<T> transpose(const Matrix<T>& m) {
    Matrix<T> t(m.cols(), m.rows());
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            t(j, i) = m(i, j);
        }
    }
    return t;
}

// Inicialização
template <typename T>
void randomize(Matrix<T>& m, double range) {
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<T> dis(-range, range);

    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            m(i, j) = dis(gen);
        }
    }
}

// NOVO: Função para somar todos os elementos da matriz
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
// MLP High Performance (Só Matrix)
// ==========================================
int main() {
    // ---------------------------------------------------------
    // 1. Definição de Escala (Matrizes Grandes)
    // ---------------------------------------------------------
    const int BATCH_SIZE = 512;  
    const int EPOCHS = 50;       
    const double LR = 0.001;     

    const int N_IN  = 1024;
    const int N_H1  = 2048;
    const int N_H2  = 2048;
    const int N_OUT = 128;

    const double N_ELEMENTS = static_cast<double>(N_OUT * BATCH_SIZE);

    std::cout << "=== CONFIGURACAO HPC ===" << std::endl;
    std::cout << "Matrizes Peso: " << N_H1 << "x" << N_IN << " (e similares)" << std::endl;
    std::cout << "Batch Size: " << BATCH_SIZE << " | Total de elementos por batch: " << N_ELEMENTS << std::endl;
    
    // ---------------------------------------------------------
    // 2. Alocação (Tudo é Matrix)
    // ---------------------------------------------------------
    
    // Pesos (W)
    Matrix<double> W1(N_H1, N_IN);
    Matrix<double> W2(N_H2, N_H1);
    Matrix<double> W3(N_OUT, N_H2);

    randomize(W1, 0.05);
    randomize(W2, 0.05);
    randomize(W3, 0.05);

    // Dados de Treino (Input X e Target Y)
    Matrix<double> X(N_IN, BATCH_SIZE); 
    Matrix<double> Y(N_OUT, BATCH_SIZE);
    
    randomize(X, 1.0); 
    randomize(Y, 1.0); 

    // Variáveis para o loop
    double final_mse = 0.0;

    // ---------------------------------------------------------
    // 3. Training Loop
    // ---------------------------------------------------------
    std::cout << "Iniciando Loop de Treinamento..." << std::endl;
    auto start_total = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        
        // --- FEED FORWARD ---
        Matrix<double> Z1 = blockMultiply(W1, X);
        Matrix<double> A1 = Z1.apply(act_tanh);

        Matrix<double> Z2 = blockMultiply(W2, A1);
        Matrix<double> A2 = Z2.apply(act_tanh);

        Matrix<double> Z3 = blockMultiply(W3, A2);
        Matrix<double> A3 = Z3.apply(act_linear); 

        // --- CALCULO DO ERRO (PARA LOG) ---
        // Erro = Target - Output
        Matrix<double> E = Y - A3;

        // MSE = Sum((Y-A3)^2) / N_ELEMENTS
        Matrix<double> E_squared = E.hadamard(E); 
        double SSE = sum_all_elements(E_squared);
        double MSE = SSE / N_ELEMENTS;
        final_mse = MSE; // Guarda o MSE para o log final

        // --- BACKPROPAGATION ---
        
        // Delta 3 (Saída)
        Matrix<double> dZ3 = Z3.apply(d_act_linear); 
        Matrix<double> delta3 = E.hadamard(dZ3);

        // dW3 = delta3 * A2_transposto
        Matrix<double> A2_T = transpose(A2);
        Matrix<double> dW3 = blockMultiply(delta3, A2_T);

        // Delta 2
        Matrix<double> W3_T = transpose(W3);
        Matrix<double> E2 = blockMultiply(W3_T, delta3);
        
        Matrix<double> dZ2 = Z2.apply(d_act_tanh);
        Matrix<double> delta2 = E2.hadamard(dZ2);

        // dW2 = delta2 * A1_transposto
        Matrix<double> A1_T = transpose(A1);
        Matrix<double> dW2 = blockMultiply(delta2, A1_T);

        // Delta 1
        Matrix<double> W2_T = transpose(W2);
        Matrix<double> E1 = blockMultiply(W2_T, delta2);

        Matrix<double> dZ1 = Z1.apply(d_act_tanh);
        Matrix<double> delta1 = E1.hadamard(dZ1);

        // dW1 = delta1 * X_transposto
        Matrix<double> X_T = transpose(X);
        Matrix<double> dW1 = blockMultiply(delta1, X_T);

        // --- ATUALIZAÇÃO DE PESOS (Gradient Descent) ---
        double alpha = LR / BATCH_SIZE; 
        
        W1 += dW1 * alpha; 
        W2 += dW2 * alpha;
        W3 += dW3 * alpha;

        // --- MONITORAMENTO ---
        if ((epoch + 1) % 10 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = now - start_total;
            
            std::cout << std::fixed << std::setprecision(6)
                      << "Epoca " << epoch + 1 
                      << " | Tempo acumulado: " << elapsed.count() << "s" 
                      << " | Erro da Epoca (MSE): " << MSE << std::endl;
        }
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> final_time = end_total - start_total;
    
    // --- ERRO FINAL ---
    std::cout << "\n=== FIM ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6)
              << "Tempo Total: " << final_time.count() << "s" << std::endl;
    std::cout << "Erro Final (MSE do último batch): " << final_mse << std::endl;

    return 0;
}