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
// Utils (Adaptados para Matrix)
// ==========================================

// Transposta (se precisar)
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
    static std::random_device rd;
    static std::mt19937 gen(rd()); 
    std::uniform_real_distribution<T> dis(-range, range);

    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            m(i, j) = dis(gen);
        }
    }
}

// Somar todos os elementos
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

    const int N_IN  = 2048;   // mantido >= 2048
    const int N_H1  = 2048;
    const int N_H2  = 2048;
    const int N_OUT = 1;      // saída única (aprox. sin)

    const double N_ELEMENTS = static_cast<double>(N_OUT * BATCH_SIZE);

    std::cout << "=== CONFIGURACAO HPC ===" << std::endl;
    std::cout << "Matrizes Peso: " << N_H1 << "x" << N_IN << " (e similares)" << std::endl;
    std::cout << "Batch Size: " << BATCH_SIZE << " | Total de elementos por batch: " << N_ELEMENTS << std::endl;

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
    
    // Vamos preparar os dados: cada coluna tem um "angle" em [-pi, pi] na primeira posição.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);
    std::uniform_real_distribution<double> noise_dist(-0.1, 0.1);

    // Preencher X e Y
    for (int b = 0; b < BATCH_SIZE; ++b) {
        double angle = angle_dist(gen);
        // primeira entrada da coluna recebe o ângulo; o resto é ruído pequeno
        X(0, b) = angle;
        for (int r = 1; r < N_IN; ++r) {
            X(r, b) = noise_dist(gen); // pode ser zero também, pequeno ruído para evitar sobreajuste trivial
        }
        Y(0, b) = std::sin(angle); // target é sin(angle)
    }

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
        Matrix<double> A3 = Z3.apply(act_linear); // A3 tem dimensão 1 x BATCH_SIZE

        // --- CALCULO DO ERRO (PARA LOG) ---
        Matrix<double> E = Y - A3;
        Matrix<double> E_squared = E.hadamard(E); 
        double SSE = sum_all_elements(E_squared);
        double MSE = SSE / N_ELEMENTS;
        final_mse = MSE; // Guarda o MSE para o log final

        // --- BACKPROPAGATION ---
        Matrix<double> dZ3 = Z3.apply(d_act_linear); 
        Matrix<double> delta3 = E.hadamard(dZ3);

        Matrix<double> A2_T = A2.T();
        Matrix<double> dW3 = blockMultiply(delta3, A2_T);

        Matrix<double> W3_T = W3.T();
        Matrix<double> E2 = blockMultiply(W3_T, delta3);

        Matrix<double> dZ2 = Z2.apply(d_act_tanh);
        Matrix<double> delta2 = E2.hadamard(dZ2);

        Matrix<double> A1_T = A1.T();
        Matrix<double> dW2 = blockMultiply(delta2, A1_T);

        Matrix<double> W2_T = W2.T();
        Matrix<double> E1 = blockMultiply(W2_T, delta2);

        Matrix<double> dZ1 = Z1.apply(d_act_tanh);
        Matrix<double> delta1 = E1.hadamard(dZ1);

        Matrix<double> X_T = X.T();
        Matrix<double> dW1 = blockMultiply(delta1, X_T);

        // --- ATUALIZAÇÃO DE PESOS (Gradient Descent) ---
        double alpha = LR / static_cast<double>(BATCH_SIZE);
        
        W1 += dW1 * alpha; 
        W2 += dW2 * alpha;
        W3 += dW3 * alpha;

        // --- MONITORAMENTO ---
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = now - start_total;
        
        std::cout << std::fixed << std::setprecision(6)
                  << "Epoca " << epoch + 1 
                  << " | Tempo acumulado: " << elapsed.count() << "s" 
                  << " | Erro da Epoca (MSE): " << MSE << std::endl;
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> final_time = end_total - start_total;
    
    // --- ERRO FINAL ---
    std::cout << "\n=== FIM ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6)
              << "Tempo Total: " << final_time.count() << "s" << std::endl;
    std::cout << "Erro Final (MSE do último batch): " << final_mse << std::endl;

    // --- AVALIAÇÃO SIMPLES: predição para um ângulo de teste ---
    double test_angle = 0.7; // exemplo qualquer
    Matrix<double> x_test(N_IN, 1);
    x_test(0, 0) = test_angle;
    for (int r = 1; r < N_IN; ++r) x_test(r, 0) = 0.0;

    Matrix<double> z1_t = blockMultiply(W1, x_test);
    Matrix<double> a1_t = z1_t.apply(act_tanh);

    Matrix<double> z2_t = blockMultiply(W2, a1_t);
    Matrix<double> a2_t = z2_t.apply(act_tanh);

    Matrix<double> z3_t = blockMultiply(W3, a2_t);
    Matrix<double> a3_t = z3_t.apply(act_linear);

    double predicted = a3_t(0, 0);
    double expected = std::sin(test_angle);

    std::cout << std::fixed << std::setprecision(8)
              << "Teste -> angle: " << test_angle
              << " | Predito: " << predicted
              << " | Real(sin): " << expected << std::endl;

    return 0;
}
