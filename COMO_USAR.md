# Guia Rápido - Análise de Fairness com Pima Diabetes

## ✅ Configuração Concluída

Seu projeto de análise de fairness está pronto para uso!

## 📁 Estrutura do Projeto

```
fairness/
├── fairness_metrics.py          # Implementação ABLNI (métrica de fairness)
├── fairness_visualization.py    # Ferramentas de visualização
├── demo_pima_diabetes.py        # Script de demonstração principal
├── test_fairness_metrics.py     # Testes unitários
├── requirements.txt             # Dependências Python
├── README.md                    # Documentação completa
├── COMO_USAR.md                 # Este arquivo
└── outputs/                     # Resultados gerados ✓
    ├── *.png                    # 7 visualizações
    ├── *.csv                    # 3 tabelas de dados
    ├── fairness_report.html     # Relatório interativo
    └── fairness_report.txt      # Relatório texto
```

## 🚀 Como Executar

### 1. Primeira vez (instalar dependências)

```bash
pip install -r requirements.txt
```

### 2. Executar demonstração

```bash
python demo_pima_diabetes.py
```

Este comando irá:
- ✓ Carregar o dataset Pima Diabetes automaticamente
- ✓ Criar grupos sensíveis (idade e IMC)
- ✓ Treinar 3 modelos de ML (Logistic Regression, Random Forest, Gradient Boosting)
- ✓ Calcular métricas de fairness (ABLNI)
- ✓ Gerar 7 visualizações em PNG
- ✓ Criar relatórios em HTML e texto
- ✓ Salvar todos os dados em CSV

## 📊 Outputs Gerados (pasta outputs/)

### Visualizações (PNG)

1. **01_subgroup_results_rf.png**
   - Net Benefit por subgrupo
   - Métricas de performance (TPR, TNR, PPV, NPV)

2. **02_comprehensive_dashboard.png**
   - Dashboard completo com 5 visualizações
   - Net benefit, gauge de fairness, heatmap, etc.

3. **03_calibration_curves.png**
   - Curvas de calibração por subgrupo
   - Distribuição de probabilidades preditas

4. **04_decision_curves.png**
   - Curvas de decisão (net benefit vs threshold)
   - Análise de desvio da mediana

5. **05_model_comparison.png**
   - Comparação de fairness entre modelos
   - ABLNI scores, ranges, tradeoffs

6. **06_threshold_sensitivity.png**
   - Como fairness varia com diferentes thresholds
   - Análise de sensibilidade clínica

7. **07_intersectional_heatmap.png**
   - Heatmap de grupos interseccionais
   - Net benefit por idade × IMC

### Dados (CSV)

1. **model_comparison.csv**
   - Comparação quantitativa dos 3 modelos
   - ABLNI scores, intervalos de confiança, estatísticas

2. **subgroup_detailed_results.csv**
   - Métricas detalhadas por cada subgrupo
   - TP, FP, TN, FN, TPR, TNR, PPV, NPV, Net Benefit

3. **threshold_sensitivity.csv**
   - Análise em diferentes thresholds
   - Impacto no fairness e net benefit

### Relatórios

1. **fairness_report.html**
   - Relatório interativo em HTML
   - Tabelas formatadas, interpretação automática
   - Abrir no navegador para visualizar

2. **fairness_report.txt**
   - Relatório em texto simples
   - Todas as métricas e interpretações

## 📖 Entendendo os Resultados

### Score ABLNI

O **ABLNI (Adjusted Intersectional Net Benefit)** mede equidade:

- **0.9 - 1.0**: 🟢 EXCELENTE - Modelo muito equitativo
- **0.8 - 0.9**: 🟡 BOM - Equidade aceitável
- **0.7 - 0.8**: 🟠 MODERADO - Atenção necessária
- **< 0.7**: 🔴 CRÍTICO - Intervenção urgente

### Net Benefit

- Mede utilidade clínica da predição
- Considera trade-off entre verdadeiros e falsos positivos
- Valores maiores = melhor performance
- Valores negativos = pior que não intervir

### Grupos Interseccionais

No demo com Pima Diabetes:
- **younger_normal_bmi**: Jovens com IMC normal
- **younger_high_bmi**: Jovens com IMC alto
- **older_normal_bmi**: Idosos com IMC normal
- **older_high_bmi**: Idosos com IMC alto

## 💻 Uso Programático

### Exemplo Simples

```python
from fairness_metrics import AdjustedIntersectionalNetBenefit
import pandas as pd

# Seus dados
y_true = [0, 1, 0, 1, ...]
y_pred_proba = [0.2, 0.8, 0.1, 0.9, ...]
sensitive_attrs = pd.DataFrame({
    'age': ['young', 'old', 'young', 'old', ...],
    'gender': ['F', 'M', 'F', 'M', ...]
})

# Calcular fairness
ablni = AdjustedIntersectionalNetBenefit(
    threshold=0.5,              # Limiar de decisão
    bootstrap_iterations=1000   # Para IC 95%
)

score = ablni.fit(y_true, y_pred_proba, sensitive_attrs)

print(f"ABLNI Score: {score:.3f}")
print(ablni.get_summary_report())

# Visualizar
ablni.plot_subgroup_results(save_path='my_analysis.png')
```

### Comparar Modelos

```python
from fairness_visualization import FairnessComparator

comparator = FairnessComparator()
comparator.add_model('Model A', y_true, y_pred_a, sensitive_attrs)
comparator.add_model('Model B', y_true, y_pred_b, sensitive_attrs)

comparator.plot_comparison(save_path='comparison.png')
table = comparator.get_comparison_table()
print(table)
```

## 🔧 Personalização

### Modificar o Script Demo

Edite `demo_pima_diabetes.py` para:

1. **Usar seu próprio dataset**:
```python
# Substituir a seção de carregamento (linha ~40)
X = pd.read_csv('seu_dataset.csv')
y = X['target']
X = X.drop('target', axis=1)
```

2. **Definir seus atributos sensíveis**:
```python
sensitive_attrs = pd.DataFrame({
    'gender': X['gender'],
    'race': X['race'],
    'age_group': pd.cut(X['age'], bins=[0, 30, 60, 100])
})
```

3. **Ajustar threshold clínico**:
```python
ablni = AdjustedIntersectionalNetBenefit(
    threshold=0.4,  # Seu threshold específico
    harm_to_benefit_ratio=1.5  # Peso de falsos positivos
)
```

## 🧪 Executar Testes

Verificar se tudo funciona corretamente:

```bash
pytest test_fairness_metrics.py -v
```

## 🆘 Resolução de Problemas

### Erro: ModuleNotFoundError

```bash
pip install -r requirements.txt
```

### Erro: Dataset não carrega

O script tem fallback automático para dataset simulado.
Verifique sua conexão com internet.

### Gráficos não aparecem

- No Windows: Verifique se matplotlib está instalado
- Em servidor: Use `plt.savefig()` sem `plt.show()`

### Caracteres estranhos no terminal

Isso é normal no Windows. Os arquivos salvos estão corretos.

## 📚 Recursos Adicionais

- **README.md**: Documentação completa
- **fairness_metrics.py**: Docstrings detalhadas
- **Exemplos de uso**: Ver seção STEP 10 em `fairness.py`

## 🤝 Suporte

- Issues: Criar issue no repositório
- Dúvidas: Contatar equipe LABDAPS
- Sugestões: Pull requests bem-vindos!

---

**Última atualização**: 2025-10-02
**Versão**: 1.0.0

