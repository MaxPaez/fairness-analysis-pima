"""
Demonstração das Funções de Fairness com Dataset Pima Diabetes
Testa métricas de fairness e gera visualizações na pasta outputs/
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# Importar SDK de fairness
from fairness_metrics import AdjustedIntersectionalNetBenefit, ablni_score
from fairness_visualization import (
    FairnessVisualizer, 
    FairnessComparator,
    generate_fairness_report_html
)

# Configurar estilo de plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

# Criar pasta de outputs
os.makedirs('outputs', exist_ok=True)

print("=" * 80)
print("ANÁLISE DE FAIRNESS - DATASET PIMA DIABETES")
print("=" * 80)

# ============================================================================
# 1. CARREGAR E PREPARAR O DATASET PIMA DIABETES
# ============================================================================
print("\n[1] Carregando dataset Pima Diabetes...")

try:
    # Tentar carregar via OpenML
    diabetes = fetch_openml('diabetes', version=1, as_frame=True, parser='auto')
    X = diabetes.data
    y = diabetes.target
    
    # Converter target para binário
    if y.dtype == 'object' or y.dtype.name == 'category':
        # Se for categórico, mapear os valores
        unique_vals = y.unique()
        if len(unique_vals) == 2:
            # Pegar o valor positivo (tested_positive ou 1)
            positive_val = [v for v in unique_vals if 'positive' in str(v).lower() or v == '1']
            if positive_val:
                y = (y == positive_val[0]).astype(int)
            else:
                y = (y == unique_vals[1]).astype(int)
        else:
            y = pd.Series(y).map({'tested_positive': 1, 'tested_negative': 0})
    else:
        y = y.astype(int)
    
    print("[OK] Dataset carregado via OpenML")
    
except Exception as e:
    print(f"[AVISO] Nao foi possivel carregar via OpenML: {e}")
    print("  Criando dataset simulado baseado em Pima Diabetes...")
    
    # Criar dataset simulado se falhar
    np.random.seed(42)
    n_samples = 768
    
    X = pd.DataFrame({
        'preg': np.random.poisson(3, n_samples),
        'plas': np.random.normal(120, 30, n_samples),
        'pres': np.random.normal(70, 12, n_samples),
        'skin': np.random.normal(20, 15, n_samples),
        'insu': np.random.normal(80, 110, n_samples),
        'mass': np.random.normal(32, 7, n_samples),
        'pedi': np.random.gamma(2, 0.2, n_samples),
        'age': np.random.gamma(6, 5, n_samples)
    })
    
    # Criar target com relação aos features
    risk_score = (
        0.1 * X['preg'] +
        0.02 * (X['plas'] - 120) +
        0.1 * (X['mass'] - 32) +
        0.5 * X['pedi'] +
        0.05 * X['age']
    )
    prob = 1 / (1 + np.exp(-risk_score + 2))
    y = (np.random.random(n_samples) < prob).astype(int)
    
    print("[OK] Dataset simulado criado")

print(f"  - Tamanho: {len(X)} amostras")
print(f"  - Features: {X.shape[1]} variaveis")
print(f"  - Prevalencia de diabetes: {y.mean():.1%}")

# ============================================================================
# 2. CRIAR ATRIBUTOS SENSÍVEIS SIMULADOS
# ============================================================================
print("\n[2] Criando atributos sensíveis...")

# Como o dataset Pima original não tem atributos demográficos explícitos,
# vamos criar grupos baseados em características clínicas
np.random.seed(42)

# Dividir por idade
age_col = 'age' if 'age' in X.columns else X.columns[-1]
age_median = X[age_col].median()

# Dividir por IMC (mass)
bmi_col = 'mass' if 'mass' in X.columns else X.columns[5]
bmi_median = X[bmi_col].median()

sensitive_attrs = pd.DataFrame({
    'age_group': np.where(X[age_col] >= age_median, 'older', 'younger'),
    'bmi_group': np.where(X[bmi_col] >= bmi_median, 'high_bmi', 'normal_bmi')
})

print("[OK] Atributos sensiveis criados:")
print(f"  - age_group: {sensitive_attrs['age_group'].value_counts().to_dict()}")
print(f"  - bmi_group: {sensitive_attrs['bmi_group'].value_counts().to_dict()}")
print(f"  - Grupos interseccionais: {len(sensitive_attrs.drop_duplicates())}")

# ============================================================================
# 3. DIVIDIR DADOS E TREINAR MODELOS
# ============================================================================
print("\n[3] Dividindo dados e treinando modelos...")

X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive_attrs, test_size=0.3, random_state=42, stratify=y
)

print(f"  - Treino: {len(X_train)} amostras")
print(f"  - Teste: {len(X_test)} amostras")

# Normalizar features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Dicionário para armazenar modelos
models = {}

# Modelo 1: Logistic Regression
print("\n  Treinando Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
models['Logistic Regression'] = lr.predict_proba(X_test_scaled)[:, 1]
print(f"    AUC-ROC: {roc_auc_score(y_test, models['Logistic Regression']):.3f}")

# Modelo 2: Random Forest
print("  Treinando Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
rf.fit(X_train_scaled, y_train)
models['Random Forest'] = rf.predict_proba(X_test_scaled)[:, 1]
print(f"    AUC-ROC: {roc_auc_score(y_test, models['Random Forest']):.3f}")

# Modelo 3: Gradient Boosting
print("  Treinando Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=3)
gb.fit(X_train_scaled, y_train)
models['Gradient Boosting'] = gb.predict_proba(X_test_scaled)[:, 1]
print(f"    AUC-ROC: {roc_auc_score(y_test, models['Gradient Boosting']):.3f}")

print(f"\n[OK] {len(models)} modelos treinados com sucesso")

# ============================================================================
# 4. ANÁLISE DE FAIRNESS DETALHADA - RANDOM FOREST
# ============================================================================
print("\n[4] Análise de Fairness Detalhada - Random Forest")
print("-" * 80)

ablni_rf = AdjustedIntersectionalNetBenefit(
    threshold=0.35,  # Threshold clínico
    prevalence_weighted=True,
    bootstrap_iterations=500,
    confidence_level=0.95,
    random_state=42
)

score_rf = ablni_rf.fit(y_test, models['Random Forest'], sens_test)

print(f"\nScore ABLNI: {score_rf:.4f}")
if ablni_rf.confidence_interval_:
    ci = ablni_rf.confidence_interval_
    print(f"IC 95%: ({ci[0]:.4f}, {ci[1]:.4f})")

# Gerar relatório detalhado
print("\n" + ablni_rf.get_summary_report())

# Plot básico
print("\nGerando visualização de resultados por subgrupo...")
ablni_rf.plot_subgroup_results(
    figsize=(14, 6),
    save_path='outputs/01_subgroup_results_rf.png'
)

# ============================================================================
# 5. VISUALIZAÇÕES AVANÇADAS
# ============================================================================
print("\n[5] Gerando visualizações avançadas...")

visualizer = FairnessVisualizer(ablni_rf)

# Dashboard completo
print("  - Dashboard completo...")
visualizer.plot_comprehensive_dashboard(
    figsize=(16, 12),
    save_path='outputs/02_comprehensive_dashboard.png'
)

# Curvas de calibração
print("  - Curvas de calibração...")
visualizer.plot_calibration_curves(
    y_test, models['Random Forest'], sens_test,
    n_bins=10,
    figsize=(14, 6),
    save_path='outputs/03_calibration_curves.png'
)

# Curvas de decisão
print("  - Curvas de decisão...")
visualizer.plot_decision_curves(
    y_test, models['Random Forest'], sens_test,
    threshold_range=(0.1, 0.6),
    n_thresholds=50,
    figsize=(14, 6),
    save_path='outputs/04_decision_curves.png'
)

# ============================================================================
# 6. COMPARAÇÃO ENTRE MODELOS
# ============================================================================
print("\n[6] Comparando fairness entre modelos...")
print("-" * 80)

comparator = FairnessComparator()

for model_name, predictions in models.items():
    print(f"  Avaliando {model_name}...")
    comparator.add_model(
        name=model_name,
        y_true=y_test,
        y_pred_proba=predictions,
        sensitive_attrs=sens_test,
        ablni_kwargs={
            'threshold': 0.35,
            'bootstrap_iterations': 300,
            'random_state': 42
        }
    )

# Plot de comparação
print("\nGerando visualização de comparação...")
comparator.plot_comparison(
    figsize=(16, 10),
    save_path='outputs/05_model_comparison.png'
)

# Tabela de comparação
comparison_table = comparator.get_comparison_table()
print("\n" + "=" * 80)
print("TABELA DE COMPARAÇÃO DE MODELOS")
print("=" * 80)
print(comparison_table.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

# Salvar tabela
comparison_table.to_csv('outputs/model_comparison.csv', index=False)
print("\n[OK] Tabela salva em: outputs/model_comparison.csv")

# ============================================================================
# 7. ANÁLISE DE SENSIBILIDADE - DIFERENTES THRESHOLDS
# ============================================================================
print("\n[7] Análise de Sensibilidade - Diferentes Thresholds...")
print("-" * 80)

thresholds_to_test = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
threshold_results = []

for thresh in thresholds_to_test:
    ablni_thresh = AdjustedIntersectionalNetBenefit(
        threshold=thresh,
        bootstrap_iterations=0,
        random_state=42
    )
    score = ablni_thresh.fit(y_test, models['Random Forest'], sens_test)
    
    threshold_results.append({
        'threshold': thresh,
        'ablni_score': score,
        'min_net_benefit': ablni_thresh.subgroup_results_['net_benefit'].min(),
        'max_net_benefit': ablni_thresh.subgroup_results_['net_benefit'].max(),
        'range': ablni_thresh.subgroup_results_['net_benefit'].max() - 
                ablni_thresh.subgroup_results_['net_benefit'].min()
    })

threshold_df = pd.DataFrame(threshold_results)
print("\nFairness em diferentes thresholds clínicos:")
print(threshold_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

# Plot de sensibilidade
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(threshold_df['threshold'], threshold_df['ablni_score'], 
            'o-', linewidth=2, markersize=8, color='#1f77b4')
axes[0].set_xlabel('Threshold de Decisão')
axes[0].set_ylabel('Score ABLNI')
axes[0].set_title('Fairness vs Threshold')
axes[0].grid(alpha=0.3)
axes[0].axhline(0.8, color='orange', linestyle='--', label='Aceitável', alpha=0.7)
axes[0].legend()

axes[1].plot(threshold_df['threshold'], threshold_df['min_net_benefit'], 
            'o-', linewidth=2, markersize=8, label='Mínimo', color='red')
axes[1].plot(threshold_df['threshold'], threshold_df['max_net_benefit'], 
            'o-', linewidth=2, markersize=8, label='Máximo', color='green')
axes[1].fill_between(threshold_df['threshold'], 
                     threshold_df['min_net_benefit'],
                     threshold_df['max_net_benefit'], 
                     alpha=0.3)
axes[1].set_xlabel('Threshold de Decisão')
axes[1].set_ylabel('Net Benefit')
axes[1].set_title('Range de Net Benefit')
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(threshold_df['threshold'], threshold_df['range'], 
            'o-', linewidth=2, markersize=8, color='purple')
axes[2].set_xlabel('Threshold de Decisão')
axes[2].set_ylabel('Range de Net Benefit')
axes[2].set_title('Magnitude da Disparidade')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/06_threshold_sensitivity.png', dpi=300, bbox_inches='tight')
print("[OK] Grafico salvo em: outputs/06_threshold_sensitivity.png")
plt.show()

# ============================================================================
# 8. SALVAR RESULTADOS DETALHADOS
# ============================================================================
print("\n[8] Salvando resultados detalhados...")

# Salvar resultados detalhados dos subgrupos
ablni_rf.subgroup_results_.to_csv('outputs/subgroup_detailed_results.csv', index=False)
print("[OK] Resultados detalhados: outputs/subgroup_detailed_results.csv")

# Salvar análise de threshold
threshold_df.to_csv('outputs/threshold_sensitivity.csv', index=False)
print("[OK] Analise de threshold: outputs/threshold_sensitivity.csv")

# Gerar relatório HTML
generate_fairness_report_html(
    ablni_rf, 
    output_path='outputs/fairness_report.html'
)
print("[OK] Relatorio HTML: outputs/fairness_report.html")

# Gerar relatório texto
with open('outputs/fairness_report.txt', 'w', encoding='utf-8') as f:
    f.write(ablni_rf.get_summary_report())
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("COMPARACAO DE MODELOS\n")
    f.write("=" * 80 + "\n\n")
    f.write(comparison_table.to_string(index=False))
print("[OK] Relatorio texto: outputs/fairness_report.txt")

# ============================================================================
# 9. ANÁLISE INTERSECCIONAL DETALHADA
# ============================================================================
print("\n[9] Análise Interseccional Detalhada...")
print("-" * 80)

# Heatmap de net benefit por grupo interseccional
results_pivot = ablni_rf.subgroup_results_.copy()

# Separar os grupos interseccionais
results_pivot[['age', 'bmi']] = results_pivot['subgroup'].str.split('_', n=1, expand=True)
pivot_table = results_pivot.pivot_table(
    values='net_benefit', 
    index='age', 
    columns='bmi',
    aggfunc='mean'
)

plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt='.4f', cmap='RdYlGn', 
           center=pivot_table.mean().mean(), cbar_kws={'label': 'Net Benefit'})
plt.title('Net Benefit por Grupos Interseccionais\n(Random Forest)')
plt.xlabel('Grupo BMI')
plt.ylabel('Grupo Idade')
plt.tight_layout()
plt.savefig('outputs/07_intersectional_heatmap.png', dpi=300, bbox_inches='tight')
print("[OK] Heatmap interseccional: outputs/07_intersectional_heatmap.png")
plt.show()

# ============================================================================
# 10. RESUMO FINAL E RECOMENDAÇÕES
# ============================================================================
print("\n" + "=" * 80)
print("RESUMO FINAL DA ANÁLISE")
print("=" * 80)

print("\n[METRICAS PRINCIPAIS]")
print(f"  - Score ABLNI (Random Forest): {score_rf:.4f}")
print(f"  - Numero de subgrupos analisados: {len(ablni_rf.subgroup_results_)}")
print(f"  - Range de Net Benefit: {ablni_rf.subgroup_results_['net_benefit'].min():.4f} a "
      f"{ablni_rf.subgroup_results_['net_benefit'].max():.4f}")

print("\n[ARQUIVOS GERADOS]")
outputs_files = [
    "01_subgroup_results_rf.png - Resultados por subgrupo",
    "02_comprehensive_dashboard.png - Dashboard completo",
    "03_calibration_curves.png - Curvas de calibração",
    "04_decision_curves.png - Curvas de decisão",
    "05_model_comparison.png - Comparação de modelos",
    "06_threshold_sensitivity.png - Análise de sensibilidade",
    "07_intersectional_heatmap.png - Heatmap interseccional",
    "model_comparison.csv - Tabela de comparação",
    "subgroup_detailed_results.csv - Resultados detalhados",
    "threshold_sensitivity.csv - Análise de threshold",
    "fairness_report.html - Relatório HTML interativo",
    "fairness_report.txt - Relatório em texto"
]

for file in outputs_files:
    print(f"  [OK] {file}")

print("\n[RECOMENDACOES]")

# Subgrupo com pior performance
worst_subgroup = ablni_rf.subgroup_results_.loc[
    ablni_rf.subgroup_results_['net_benefit'].idxmin()
]
print(f"  1. Atenção especial ao subgrupo '{worst_subgroup['subgroup']}'")
print(f"     (Net Benefit: {worst_subgroup['net_benefit']:.4f})")

best_model = comparison_table.loc[comparison_table['ABLNI_Score'].idxmax()]
print(f"\n  2. Melhor modelo em termos de fairness: {best_model['Model']}")
print(f"     (ABLNI Score: {best_model['ABLNI_Score']:.4f})")

if score_rf < 0.8:
    print("\n  3. Considerar tecnicas de mitigacao de vies:")
    print("     - Reweighting de amostras")
    print("     - Adversarial debiasing")
    print("     - Modelos especificos por subgrupo")

print("\n  4. Monitoramento continuo de fairness em producao")
print("  5. Validacao com stakeholders das populacoes afetadas")

print("\n" + "=" * 80)
print("ANALISE CONCLUIDA COM SUCESSO!")
print("Todos os resultados foram salvos na pasta: outputs/")
print("=" * 80)

