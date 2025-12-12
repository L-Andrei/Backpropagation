#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>

// Seus arquivos de cabeçalho (assumo mesma API que você usou antes)
#include <ifnum/linearAlgebra/matrix.hpp>
#include <ifnum/linearAlgebra/indexGenerator.hpp>

using namespace ifnum::linearAlgebra;

// ========================= Utils =========================

// Inicialização Glorot (Xavier) - normal
void xavier_init(Matrix<double>& M, int fan_in, int fan_out) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    double stddev = std::sqrt(2.0 / (fan_in + fan_out));
    std::normal_distribution<double> dist(0.0, stddev);
    for (size_t i = 0; i < M.rows(); ++i)
        for (size_t j = 0; j < M.cols(); ++j)
            M(i, j) = dist(gen);
}

// Preenche com ruído gaussiano (útil para dados ou pequenas perturbações)
void fill_noise(Matrix<double>& M, double sigma) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, sigma);
    for (size_t i = 0; i < M.rows(); ++i)
        for (size_t j = 0; j < M.cols(); ++j)
            M(i, j) = dist(gen);
}

template <typename T>
T sum_all_elements(const Matrix<T>& m) {
    T s = 0.0;
    for (size_t i = 0; i < m.rows(); ++i)
        for (size_t j = 0; j < m.cols(); ++j)
            s += m(i, j);
    return s;
}

// Ativações
double relu(double x) { return x > 0.0 ? x : 0.0; }
double d_relu(double x) { return x > 0.0 ? 1.0 : 0.0; }
double linear(double x) { return x; }
double d_linear(double) { return 1.0; }

// Clip elementar: usado para limitar atualizações
double clip_double(double v, double limit) {
    if (v > limit) return limit;
    if (v < -limit) return -limit;
    return v;
}

// ========================= MLP otimizado =========================
int main() {
    // ----------------- hiperparâmetros -----------------
    const int BATCH_SIZE = 256;   // menor que antes para convergência mais estável
    const int EPOCHS = 400;       // mais epochs
    const double LR = 0.01;       // passo inicial maior
    const double GRAD_CLIP = 1.0; // limite simples para atualizações (por elemento)

    // arquitetura (mantendo camadas grandes no início)
    const int N_IN  = 2048;
    const int N_H1  = 1024;
    const int N_H2  = 512;
    const int N_H3  = 256;
    const int N_H4  = 128;
    const int N_H5  = 64;
    const int N_OUT = 1;

    const double N_ELEMENTS = static_cast<double>(N_OUT * BATCH_SIZE);

    std::cout << "=== MLP Otimizado: 2048 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 1 ===\n";
    std::cout << "Batch=" << BATCH_SIZE << " | Epochs=" << EPOCHS << " | LR=" << LR << "\n";

    // ----------------- pesos -----------------
    Matrix<double> W1(N_H1, N_IN);
    Matrix<double> W2(N_H2, N_H1);
    Matrix<double> W3(N_H3, N_H2);
    Matrix<double> W4(N_H4, N_H3);
    Matrix<double> W5(N_H5, N_H4);
    Matrix<double> W6(N_OUT, N_H5);

    // Xavier init
    xavier_init(W1, N_IN, N_H1);
    xavier_init(W2, N_H1, N_H2);
    xavier_init(W3, N_H2, N_H3);
    xavier_init(W4, N_H3, N_H4);
    xavier_init(W5, N_H4, N_H5);
    xavier_init(W6, N_H5, N_OUT);

    // ----------------- dados -----------------
    Matrix<double> X(N_IN, BATCH_SIZE);
    Matrix<double> Y(N_OUT, BATCH_SIZE);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> angle_dist(-M_PI, M_PI);
    std::normal_distribution<double> tiny_noise(0.0, 0.005); // ruído pequeno

    // Preenchendo X e Y:
    // Normalizamos angle para [-1, 1] dividindo por PI e replicamos o sinal por todas as features
    for (int b = 0; b < BATCH_SIZE; ++b) {
        double angle = angle_dist(gen);
        double scaled = angle / M_PI; // agora em [-1, 1]
        for (int r = 0; r < N_IN; ++r) {
            // espalhe o sinal por todas as features (facilita aprendizado com grandes matrizes)
            X(r, b) = scaled + tiny_noise(gen);
        }
        Y(0, b) = std::sin(angle); // target sem escala (entre -1 e 1)
    }

    double final_mse = 0.0;

    std::cout << "Treinamento iniciado...\n";
    auto start_total = std::chrono::high_resolution_clock::now();

    // ----------------- loop de treinamento -----------------
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {

        // -------- forward --------
        Matrix<double> Z1 = W1 * X;
        Matrix<double> A1 = Z1.apply(relu);

        Matrix<double> Z2 = W2 * A1;
        Matrix<double> A2 = Z2.apply(relu);

        Matrix<double> Z3 = W3 * A2;
        Matrix<double> A3 = Z3.apply(relu);

        Matrix<double> Z4 = W4 * A3;
        Matrix<double> A4 = Z4.apply(relu);

        Matrix<double> Z5 = W5 * A4;
        Matrix<double> A5 = Z5.apply(relu);

        Matrix<double> Z6 = W6 * A5;
        Matrix<double> A6 = Z6.apply(linear); // saída 1 x BATCH_SIZE

        // -------- erro e métricas --------
        Matrix<double> E = Y - A6;
        Matrix<double> E2 = E.hadamard(E);
        double SSE = sum_all_elements(E2);
        double MSE = SSE / N_ELEMENTS;
        final_mse = MSE;

        // -------- backprop (ReLU derivada simples aplicada sobre Z) --------
        // camada saída
        Matrix<double> dZ6 = Z6.apply(d_linear);
        Matrix<double> delta6 = E.hadamard(dZ6);               // 1 x BATCH

        Matrix<double> dW6 = delta6 * A5.T();                  // (1 x BATCH) * (BATCH x 64) => 1 x 64

        // camada 5
        Matrix<double> tmp5 = (W6.T() * delta6);               // 64 x BATCH
        Matrix<double> dZ5 = Z5.apply(d_relu);
        Matrix<double> delta5 = tmp5.hadamard(dZ5);
        Matrix<double> dW5 = delta5 * A4.T();                  // 64 x 128

        // camada 4
        Matrix<double> tmp4 = (W5.T() * delta5);               // 128 x BATCH
        Matrix<double> dZ4 = Z4.apply(d_relu);
        Matrix<double> delta4 = tmp4.hadamard(dZ4);
        Matrix<double> dW4 = delta4 * A3.T();                  // 128 x 256

        // camada 3
        Matrix<double> tmp3 = (W4.T() * delta4);               // 256 x BATCH
        Matrix<double> dZ3 = Z3.apply(d_relu);
        Matrix<double> delta3 = tmp3.hadamard(dZ3);
        Matrix<double> dW3 = delta3 * A2.T();                  // 256 x 512

        // camada 2
        Matrix<double> tmp2 = (W3.T() * delta3);               // 512 x BATCH
        Matrix<double> dZ2 = Z2.apply(d_relu);
        Matrix<double> delta2 = tmp2.hadamard(dZ2);
        Matrix<double> dW2 = delta2 * A1.T();                  // 512 x 1024

        // camada 1
        Matrix<double> tmp1 = (W2.T() * delta2);               // 1024 x BATCH
        Matrix<double> dZ1 = Z1.apply(d_relu);
        Matrix<double> delta1 = tmp1.hadamard(dZ1);
        Matrix<double> dW1 = delta1 * X.T();                   // 1024 x 2048

        // -------- aplicar clipping simples nas atualizações (por elemento) --------
        auto clip_matrix = [&](Matrix<double>& M, double limit) {
            for (size_t i = 0; i < M.rows(); ++i)
                for (size_t j = 0; j < M.cols(); ++j)
                    M(i, j) = clip_double(M(i, j), limit);
        };

        clip_matrix(dW1, GRAD_CLIP);
        clip_matrix(dW2, GRAD_CLIP);
        clip_matrix(dW3, GRAD_CLIP);
        clip_matrix(dW4, GRAD_CLIP);
        clip_matrix(dW5, GRAD_CLIP);
        clip_matrix(dW6, GRAD_CLIP);

        // -------- atualização de pesos (SGD simples) --------
        double alpha = LR / static_cast<double>(BATCH_SIZE); // dividir por batch para normalizar passo
        W1 += dW1 * alpha;
        W2 += dW2 * alpha;
        W3 += dW3 * alpha;
        W4 += dW4 * alpha;
        W5 += dW5 * alpha;
        W6 += dW6 * alpha;

        // -------- logging --------
        if ((epoch % 10) == 0 || epoch == EPOCHS - 1) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_total).count();
            std::cout << std::fixed << std::setprecision(6)
                      << "Epoch " << epoch + 1
                      << " | elapsed: " << elapsed << " s"
                      << " | MSE: " << MSE << std::endl;
        }

        // (Opcional) poderia re-gerar X,Y por batch aqui se você preferir dataset infinito;
        // mantive dados fixos para estabilidade do experimento.
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(end_total - start_total).count();

    std::cout << "\n=== FIM ===\n";
    std::cout << "Tempo total: " << total_time << " s\n";
    std::cout << "Erro final (MSE): " << final_mse << std::endl;

    // ----------------- avaliação simples -----------------
    double test_angle = 0.7;
    Matrix<double> x_test(N_IN, 1);
    double scaled = test_angle / M_PI;
    for (int r = 0; r < N_IN; ++r) x_test(r, 0) = scaled; // sem ruído para teste

    Matrix<double> z1_t = W1 * x_test;
    Matrix<double> a1_t = z1_t.apply(relu);

    Matrix<double> z2_t = W2 * a1_t;
    Matrix<double> a2_t = z2_t.apply(relu);

    Matrix<double> z3_t = W3 * a2_t;
    Matrix<double> a3_t = z3_t.apply(relu);

    Matrix<double> z4_t = W4 * a3_t;
    Matrix<double> a4_t = z4_t.apply(relu);

    Matrix<double> z5_t = W5 * a4_t;
    Matrix<double> a5_t = z5_t.apply(relu);

    Matrix<double> z6_t = W6 * a5_t;
    Matrix<double> out_t = z6_t.apply(linear);

    double predicted = out_t(0, 0);
    double expected  = std::sin(test_angle);

    std::cout << std::fixed << std::setprecision(8)
              << "Teste -> angle: " << test_angle
              << " | Predito: " << predicted
              << " | Real(sin): " << expected << std::endl;

    return 0;
}
