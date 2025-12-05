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

// Transposta Otimizada
template <typename T>
Matrix<T> transpose(const Matrix<T>& m) {
    Matrix<T> t(m.cols(), m.rows());
    // Para HPC extremo, aqui também poderíamos usar blocking, 
    // mas vamos confiar no compilador para este loop simples
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            t(j, i) = m(i, j);
        }
    }
    return t;
}

// Inicialização (Preenche a Matrix diretamente)
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

// Funções de Ativação (Compatíveis com Matrix::apply)
double act_tanh(double x) { return std::tanh(x); }
// Derivada: 1 - tanh²(x). 
// Nota: O mlp.m calcula d = 1 - x² assumindo que x já é o output da tanh.
// Aqui faremos o cálculo completo para garantir precisão numérica.
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
    // Batch Size: Processamos 512 amostras simultaneamente para usar blockMultiply
    const int BATCH_SIZE = 512;  
    const int EPOCHS = 50;       
    const double LR = 0.001;     

    // Topologia da Rede
    const int N_IN  = 1024; // Entrada de Alta Dimensão
    const int N_H1  = 2048; // Camada Oculta Grande
    const int N_H2  = 2048; // Camada Oculta Grande
    const int N_OUT = 128;  // Saída Múltipla

    std::cout << "=== CONFIGURACAO HPC ===" << std::endl;
    std::cout << "Matrizes Peso: " << N_H1 << "x" << N_IN << " (e similares)" << std::endl;
    std::cout << "Matrizes Dados (Batch): " << N_IN << "x" << BATCH_SIZE << std::endl;
    
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
    // Usamos Matrix diretamente, nada de std::vector
    Matrix<double> X(N_IN, BATCH_SIZE); 
    Matrix<double> Y(N_OUT, BATCH_SIZE);
    
    randomize(X, 1.0); // Dados sintéticos
    randomize(Y, 1.0); // Targets sintéticos

    // ---------------------------------------------------------
    // 3. Training Loop
    // ---------------------------------------------------------
    std::cout << "Iniciando Loop de Treinamento..." << std::endl;
    auto start_total = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        
        // --- FEED FORWARD ---
        
        // Camada 1: Z1 = W1 * X
        // blockMultiply usa cache blocking para essa matriz grande
        Matrix<double> Z1 = blockMultiply(W1, X);
        Matrix<double> A1 = Z1.apply(act_tanh);

        // Camada 2: Z2 = W2 * A1
        Matrix<double> Z2 = blockMultiply(W2, A1);
        Matrix<double> A2 = Z2.apply(act_tanh);

        // Camada 3 (Output): Z3 = W3 * A2
        Matrix<double> Z3 = blockMultiply(W3, A2);
        Matrix<double> A3 = Z3.apply(act_linear); 

        // --- BACKPROPAGATION ---
        // Erro = Target - Output
        Matrix<double> E = Y - A3;

        // Calcular Gradientes
        
        // Delta 3 (Saída)
        // d_act_linear é 1.0, então delta3 = E
        Matrix<double> dZ3 = Z3.apply(d_act_linear); 
        Matrix<double> delta3 = E.hadamard(dZ3);

        // dW3 = delta3 * A2_transposto
        Matrix<double> A2_T = transpose(A2);
        Matrix<double> dW3 = blockMultiply(delta3, A2_T);

        // Delta 2
        // Erro propagado: W3_transposto * delta3
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
        // Normalização pelo batch size (média do gradiente)
        double alpha = LR / BATCH_SIZE; 
        
        // Operações += definidas em matrix.hpp
        // Matrix * Escalar -> Matrix
        W1 += dW1 * alpha; 
        W2 += dW2 * alpha;
        W3 += dW3 * alpha;

        // --- Monitoramento ---
        if ((epoch + 1) % 10 == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = now - start_total;
            std::cout << "Epoca " << epoch + 1 
                      << " | Tempo acumulado: " << elapsed.count() << "s" 
                      << " | Exemplo Output[0,0]: " << A3(0,0) << std::endl;
        }
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> final_time = end_total - start_total;
    
    std::cout << "=== FIM ===" << std::endl;
    std::cout << "Tempo Total: " << final_time.count() << "s" << std::endl;

    return 0;
}