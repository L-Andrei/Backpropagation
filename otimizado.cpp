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

// Transposta (Mantida)
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

// Inicialização Aleatória Uniforme
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

// Soma de elementos (para MSE)
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
// Tanh é boa para normalizar entre -1 e 1 nas camadas ocultas
double act_tanh(double x) { return std::tanh(x); }
double d_act_tanh(double x) { double t = std::tanh(x); return 1.0 - t * t; }

// Linear na saída para permitir prever o valor exato do seno
double act_linear(double x) { return x; }
double d_act_linear(double x) { return 1.0; }

// ==========================================
// MLP High Performance - Sine Approximator
// ==========================================
int main() {
    // ---------------------------------------------------------
    // 1. Definição de Escala (Matrizes Grandes)
    // ---------------------------------------------------------
    // Mantendo dimensões grandes para estressar a multiplicação de matrizes
    const int N_IN  = 2048; 
    const int N_H1  = 2048; // Camada oculta gigante
    const int N_H2  = 2048; // Camada oculta gigante
    const int N_OUT = 1;    // Apenas 1 saída final (O valor do seno)

    const int BATCH_SIZE = 256; // Batch size ajustado para não estourar RAM se for muito grande
    const int EPOCHS = 100;     // Mais épocas para dar tempo de convergir
    const double LR = 0.0005;   // Learning rate reduzido devido ao tamanho massivo das somas

    const double N_ELEMENTS = static_cast<double>(N_OUT * BATCH_SIZE);

    std::cout << "=== CONFIGURACAO HPC: SINE APPROXIMATION ===" << std::endl;
    std::cout << "Matrizes de Peso: " << N_H1 << "x" << N_IN << " (Stress Test)" << std::endl;
    std::cout << "Objetivo: Aprender y = sin(x[0]) ignorando x[1..2047]" << std::endl;

    // Pesos (W)
    Matrix<double> W1(N_H1, N_IN);
    Matrix<double> W2(N_H2, N_H1);
    Matrix<double> W3(N_OUT, N_H2);

    // Inicialização Xavier/He simples (valores pequenos para não saturar Tanh)
    randomize(W1, -0.02, 0.02);
    randomize(W2, -0.02, 0.02);
    randomize(W3, -0.02, 0.02);

    // ---------------------------------------------------------
    // 2. Geração de Dados (Input X e Target Y)
    // ---------------------------------------------------------
    Matrix<double> X(N_IN, BATCH_SIZE); 
    Matrix<double> Y(N_OUT, BATCH_SIZE);
    
    // Gera X aleatório entre -PI e PI
    randomize(X, -3.14159, 3.14159);

    // Gera o Target Y baseada na função Seno APENAS da primeira linha de X.
    // A rede terá que aprender a zerar os pesos das outras 2047 entradas.
    for (size_t b = 0; b < BATCH_SIZE; ++b) {
        double val = X(0, b); // Pega o primeiro elemento da coluna b
        Y(0, b) = std::sin(val);
    }

    std::cout << "Dados gerados. Iniciando treinamento..." << std::endl;

    // ---------------------------------------------------------
    // 3. Training Loop
    // ---------------------------------------------------------
    auto start_total = std::chrono::high_resolution_clock::now();
    double final_mse = 0.0;
    Matrix<double> A3(N_OUT, BATCH_SIZE); // Declarado fora para usar no print final

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        
        // --- FEED FORWARD ---
        // Aqui acontece a mágica do HPC: Multiplicação de matrizes 2048x2048
        Matrix<double> Z1 = blockMultiply(W1, X);
        Matrix<double> A1 = Z1.apply(act_tanh);

        Matrix<double> Z2 = blockMultiply(W2, A1);
        Matrix<double> A2 = Z2.apply(act_tanh);

        Matrix<double> Z3 = blockMultiply(W3, A2);
        A3 = Z3.apply(act_linear); 

        // --- CALCULO DO ERRO ---
        Matrix<double> E = A3 - Y; // Pred - Target (ou Target - Pred, ajustando sinal do gradiente)
        
        // MSE
        Matrix<double> E_squared = E.hadamard(E); 
        double SSE = sum_all_elements(E_squared);
        double MSE = SSE / N_ELEMENTS;
        final_mse = MSE;

        // --- BACKPROPAGATION ---
        // Gradiente: dMSE/dW = (Pred - Target) * dActivation * Input^T
        
        // Delta 3
        Matrix<double> dZ3 = Z3.apply(d_act_linear); 
        Matrix<double> delta3 = E.hadamard(dZ3);

        Matrix<double> A2_T = A2.T();
        Matrix<double> dW3 = blockMultiply(delta3, A2_T);

        // Delta 2
        Matrix<double> W3_T = W3.T();
        Matrix<double> E2 = blockMultiply(W3_T, delta3);
        
        Matrix<double> dZ2 = Z2.apply(d_act_tanh);
        Matrix<double> delta2 = E2.hadamard(dZ2);

        Matrix<double> A1_T = A1.T();
        Matrix<double> dW2 = blockMultiply(delta2, A1_T);

        // Delta 1
        Matrix<double> W2_T = W2.T();
        Matrix<double> E1 = blockMultiply(W2_T, delta2);

        Matrix<double> dZ1 = Z1.apply(d_act_tanh);
        Matrix<double> delta1 = E1.hadamard(dZ1);

        Matrix<double> X_T = X.T();
        Matrix<double> dW1 = blockMultiply(delta1, X_T);

        // --- ATUALIZAÇÃO (Gradient Descent) ---
        // Sinal negativo pois calculamos E = Pred - Target
        double alpha = LR / BATCH_SIZE; 
        
        W1 = W1 - dW1 * alpha; 
        W2 = W2 - dW2 * alpha;
        W3 = W3 - dW3 * alpha;

        if (epoch % 10 == 0 || epoch == EPOCHS - 1) {
            std::cout << "Epoca " << std::setw(3) << epoch + 1 
                      << " | MSE: " << std::fixed << std::setprecision(6) << MSE << std::endl;
        }
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> final_time = end_total - start_total;
    
    // --- RESULTADOS ---
    std::cout << "\n=== FIM DO TREINAMENTO ===" << std::endl;
    std::cout << "Tempo Total: " << final_time.count() << "s" << std::endl;
    std::cout << "MSE Final: " << final_mse << std::endl;

    // Teste de Sanidade: Comparar Target vs Predito para o primeiro elemento do batch
    std::cout << "\n--- Comparacao Visual (Amostra do Batch 0) ---" << std::endl;
    double input_val = X(0, 0);
    double target_val = Y(0, 0); // que é sin(input_val)
    double pred_val = A3(0, 0);
    
    std::cout << "Entrada (X[0]): " << input_val << std::endl;
    std::cout << "Esperado (Sin) : " << target_val << std::endl;
    std::cout << "Predito (Rede) : " << pred_val << std::endl;
    std::cout << "Diferenca      : " << std::abs(target_val - pred_val) << std::endl;

    return 0;
}