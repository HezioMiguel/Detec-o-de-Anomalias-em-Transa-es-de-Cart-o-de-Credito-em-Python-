# Detec-o-de-Anomalias-em-Transa-es-de-Cart-o-de-Credito-em-Python-
Pipeline de Detecção de Fraude em Cartões de Crédito
Este repositório contém um pipeline estruturado de Machine Learning para identificação de transações fraudulentas em cartões de crédito. O projeto aborda o desbalanceamento severo de classes (fraudes representam aproximadamente 0,17% do dataset original) utilizando métricas apropriadas para o contexto do problema.

Desafio de Negócio e Métricas
Em sistemas de prevenção a fraude, a otimização deste modelo foi direcionada para a métrica de Recall. No cenário bancário, o custo associado a um Falso Negativo (permitir que uma fraude ocorra) é consideravelmente maior do que o impacto operacional de um Falso Positivo (bloqueio preventivo de uma transação legítima).

Estrutura do Pipeline
1. Ingestão e Preparação de Dados
Carga do dataset via URL pública do Google Storage.

Divisão estratificada (stratify=y) no particionamento para preservar a proporção da classe minoritária nos conjuntos de treino e teste.

2. Modelo Baseline (Regressão Logística)
Padronização de escala com StandardScaler integrada em Pipeline para evitar vazamento de dados (data leakage).

Ajuste de pesos de classe (class_weight="balanced") e utilização do algoritmo SAGA para convergência em alta dimensionalidade.

3. Modelo Avançado (XGBoost)
Configuração do parâmetro scale_pos_weight para penalizar erros na classe de fraude.

Uso do método de histograma (tree_method="hist") para ganho de performance computacional.

Monitoramento do logloss durante o treinamento em tempo real.

4. Otimização de Hiperparâmetros
GridSearchCV aplicado com foco em Recall.

Execução em paralelo (n_jobs=-1) para otimizar o tempo de validação cruzada.

5. Explicabilidade (SHAP)
Aplicação do SHAP para auditoria das decisões do modelo XGBoost.

Isolamento do impacto matemático das variáveis nos casos classificados como fraude para fins de conformidade e governança.

Arquivos Gerados
graficos_baseline.png: Curvas ROC e Precision-Recall do modelo inicial.

relatorio_previsoes.csv: Resultados do conjunto de teste com as predições do XGBoost.

relatorio_fraudes.txt: Dossiê em formato texto contendo os três principais fatores de decisão para cada bloqueio realizado.

Dependências
Python 3.x

Pandas

NumPy

Matplotlib

Scikit-Learn

XGBoost

SHAP

Como Executar
Instale as bibliotecas necessárias:

Bash
pip install pandas numpy matplotlib scikit-learn xgboost shap
Execute o script principal:

Bash
python detector_fraudes.py
