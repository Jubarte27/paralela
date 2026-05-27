# Relatório de Desempenho: Algoritmo Genético para Otimização de Hiperparâmetros

## 1. Arquitetura do Sistema (Sequencial vs. Paralelo)
O projeto consiste em um Algoritmo Genético (AG) adaptado para rodar o controle evolutivo em C/C++ e o treinamento/avaliação das redes neurais em Python (utilizando CPU). 

No modelo paralelo:
* Cada indivíduo e cada treinamento de rede utiliza 1 thread.
* O processo Python permanece aberto mantendo os datasets em memória (carregar os datasets consumia quase metade do tempo de execução, alteração que foi aplicada retroativamente ao sequencial).

## 2. Decisões de Paralelização e Resultados de Desempenho

### `parallel for`:

#### Gerar população, Crossover e Mutação
Apresentaram pequeno ganho de desempenho. O tempo de execução é completamente dominado pela avaliação dos indivíduos. Além disso, a quantidade de indivíduos por geração é pequena quando comparada ao número de threads (~1 por thread).

#### Enviar requisições de treinamento ao Python:
Foi responsável pelo maior ganho de desempenho. Como os cálculos de treinamento na CPU são lentos, computar múltiplos indivíduos simultaneamente trouxe uma vantagem muito superior a simplesmente tentar usar mais threads em um único treinamento.

### `task`:
#### Tratar retorno do Python

Gerou ganho mínimo de desempenho. O tratamento sequencial é rápido o suficiente frente ao tempo de avaliação e a população pequena não engarrafa o buffer de resposta. Esse mecanismo seria mais importante caso o volume de dados transmitidos fosse muito superior.

## 3. Espaço de Busca (População)
Os hiperparâmetros otimizados pelo AG:
* **Função de ativação:** `relu`, `tanh`, `sigmoid`.
* **Camadas Ocultas (Dense):** De 1 a 3 camadas.
* **Neurônios (1ª camada):** [32, 128] para 1 camada; [16, 64] para 2 camadas; [8, 32] para 3 camadas.
* **Progressão das camadas:** Mantendo o mesmo número de neurônios ou dividindo por 2 a cada camada.
* **Taxa de aprendizado:** Escala de $2^{-4.0}$ a $2^{-1.0}$.
* **Decaimento da taxa de aprendizado:** [0.0, 1.0].
* **Otimizador:** `adam` ou `adamw`.
* **Tamanho do batch:** 32, 64 ou 128.

## 4. Conjuntos de Dados (Datasets)
Os testes utilizaram o dataset **Fashion MNIST** configurado em dois tamanhos de entrada:
* **Completo:** 70.000 imagens.
* **Pequeno:** Amostra reduzida de 10.000 imagens para viabilizar os testes no tempo disponível.

Cifar 10, Cifar 100 e o MNIST original foram descartados por restrições de tempo (idealmente a avaliação seria em GPU).*

## Github com todo o projeto:
https://github.com/Jubarte27/paralela
