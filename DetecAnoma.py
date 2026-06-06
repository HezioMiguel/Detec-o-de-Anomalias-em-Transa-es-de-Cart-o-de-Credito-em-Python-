import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier
import shap



# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS

print(" Carregando a base de dados de cartões de crédito...")
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(url)

# Amostragem de 10% para testes de código se frac = 0.1. 
# NOTA PARA PRODUÇÃO: Comente a linha abaixo para treinar com a base completa (284 mil linhas).
df = df.sample(frac=0.1, random_state=42)
print(f" Base carregada. Trabalhando com {len(df)} registros de transações.\n")

# Dropando a variável 'Time', pois requer Feature Engineering para ser útil
X = df.drop(["Class", "Time"], axis=1)
y = df["Class"]

# Divisão Estratificada devido ao extremo desbalanceamento das classes (fraudes são raras)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# 2. MODELO BASELINE: REGRESSÃO LOGÍSTICA (MOTOR SAGA)

print(" Treinando Modelo Baseline (Regressão Logística)...")

pipeline_lr = Pipeline([
    ("scaler", StandardScaler()),
    # Adicionamos solver="saga" e verbose=1
    ("model", LogisticRegression(
        solver="saga", 
        max_iter=30000, 
        class_weight="balanced", 
        verbose=1
    ))
])

pipeline_lr.fit(X_train, y_train)

y_probs_lr = pipeline_lr.predict_proba(X_test)[:, 1]
 
# 3. AVALIAÇÃO DO BASELINE (CURVAS DE DESEMPENHO)

plt.figure(figsize=(12, 5))

# Curva ROC
plt.subplot(1, 2, 1)
fpr, tpr, _ = roc_curve(y_test, y_probs_lr)
plt.plot(fpr, tpr, color='blue', label=f'AUC = {roc_auc_score(y_test, y_probs_lr):.4f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray') 
plt.title("ROC Curve - Baseline")
plt.xlabel("Taxa de Falso Positivo")
plt.ylabel("Taxa de Verdadeiro Positivo")
plt.legend()

plt.subplot(1, 2, 2)
precision, recall, _ = precision_recall_curve(y_test, y_probs_lr)
plt.plot(recall, precision, color='red')
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")


plt.tight_layout()

plt.savefig("graficos_baseline.png", bbox_inches='tight', dpi=300)
plt.close() # Fecha a figura invisível da memória para não pesar o PC
print(" Gráficos do Baseline salvos na imagem 'graficos_baseline.png'!")

# 4. MODELO AVANÇADO: XGBOOST (COM MONITORAMENTO EM TEMPO REAL)

print("\n Treinando Modelo Avançado (XGBoost)...")

xgb = XGBClassifier(
    scale_pos_weight=10, 
    eval_metric="logloss",
    tree_method="hist",  
    random_state=42,
    n_estimators=100 
)

conjunto_de_avaliacao = [(X_train, y_train), (X_test, y_test)]

xgb.fit(
    X_train, y_train, 
    eval_set=conjunto_de_avaliacao, 
    verbose=10 
)

y_pred_xgb = xgb.predict(X_test)

print("\n--- Relatório de Classificação (XGBoost) ---")
print(classification_report(y_test, y_pred_xgb))

# 5. OTIMIZAÇÃO DE HIPERPARÂMETROS (GRID SEARCH)

print("\n Iniciando Otimização de Hiperparâmetros (GridSearchCV)...")
param_grid = {
    "max_depth": [3, 5],
    "n_estimators": [50, 100]
}

grid = GridSearchCV(
    estimator=XGBClassifier(eval_metric="logloss", tree_method="hist", random_state=42),
    param_grid=param_grid,
    scoring="recall", 
    cv=3,
    verbose=3, 
    n_jobs=-1 
)

grid.fit(X_train, y_train)
print(f"Melhores parâmetros encontrados: {grid.best_params_}")

# 6. EXPORTAÇÃO DOS RESULTADOS (CSV DE PREVISÕES)

print("\nExportando relatório de previsões para CSV...")
tabela_resultados = X_test.copy()
tabela_resultados["Fraude_Real"] = y_test 
tabela_resultados["Previsao_XGBoost"] = y_pred_xgb 
tabela_resultados.to_csv("relatorio_previsoes.csv", index=False)
print("Arquivo 'relatorio_previsoes.csv' gerado com sucesso!")


# 7 . EXPLICABILIDADE (SHAP) E RELATÓRIO INTELIGENTE

print("\n Caçando fraudes para análise detalhada do SHAP...")

# 1. Encontramos EXATAMENTE em quais posições (índices) o robô previu fraude (1)
indices_fraude = np.where(y_pred_xgb == 1)[0]

if len(indices_fraude) == 0:
    print("Nenhuma fraude foi identificada nesta rodada de testes.")
else:
    limite_analise = min(5, len(indices_fraude))
    indices_para_explicar = indices_fraude[:limite_analise]
    
    fraudes_detectadas = X_test.iloc[indices_para_explicar]
    
    explainer = shap.Explainer(xgb)
    shap_values_fraudes = explainer(fraudes_detectadas)
    
    print(" Escrevendo relatório de texto para humanos...")
    with open("relatorio_fraudes.txt", "w", encoding="utf-8") as arquivo:
        arquivo.write("=== RELATÓRIO DE TRANSAÇÕES BLOQUEADAS ===\n\n")
        
        for i in range(limite_analise):
            pesos = shap_values_fraudes.values[i]
            pesos_com_nomes = pd.Series(pesos, index=X_test.columns)
            
            top_3_motivos = pesos_com_nomes.sort_values(ascending=False).head(3)
            
            valor_compra = fraudes_detectadas.iloc[i]["Amount"]
            
            arquivo.write(f" ALERTA NA TRANSAÇÃO (Índice original: {indices_para_explicar[i]})\n")
            arquivo.write(f"Valor da tentativa de compra: R$ {valor_compra:.2f}\n")
            arquivo.write("Decisão: BLOQUEADA (Risco de Fraude)\n")
            arquivo.write("Principais motivos que levaram a Inteligência Artificial a bloquear:\n")
            
            for nome_variavel, peso in top_3_motivos.items():
                arquivo.write(f"  -> A variável '{nome_variavel}' aumentou a suspeita em {peso:.2f} pontos.\n")
            
            arquivo.write("-" * 50 + "\n\n")
            
    print("Relatório de texto 'relatorio_fraudes.txt' criado com sucesso com dados REAIS!")
