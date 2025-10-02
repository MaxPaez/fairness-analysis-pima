# Análise de Fairness com Dataset Pima Diabetes

Este projeto demonstra a aplicação de métricas de fairness em modelos de Machine Learning usando o dataset Pima Diabetes.

## 📋 Requisitos

```bash
pip install -r requirements.txt
```

## 🚀 Como Usar

### Executar Demonstração Completa

```bash
python demo_pima_diabetes.py
```

Este script irá:
1. Carregar o dataset Pima Diabetes
2. Criar atributos sensíveis (grupos de idade e IMC)
3. Treinar múltiplos modelos (Logistic Regression, Random Forest, Gradient Boosting)
4. Calcular métricas de fairness (ABLNI)
5. Gerar visualizações na pasta `outputs/`
6. Criar relatórios em HTML e texto

## 📊 Outputs Gerados

Todos os arquivos são salvos na pasta `outputs/`:

### Visualizações (PNG)
- `01_subgroup_results_rf.png` - Resultados por subgrupo
- `02_comprehensive_dashboard.png` - Dashboard completo de fairness
- `03_calibration_curves.png` - Curvas de calibração por subgrupo
- `04_decision_curves.png` - Curvas de decisão (net benefit)
- `05_model_comparison.png` - Comparação entre modelos
- `06_threshold_sensitivity.png` - Análise de sensibilidade ao threshold
- `07_intersectional_heatmap.png` - Heatmap de grupos interseccionais

### Dados (CSV)
- `model_comparison.csv` - Comparação quantitativa de modelos
- `subgroup_detailed_results.csv` - Métricas detalhadas por subgrupo
- `threshold_sensitivity.csv` - Análise de diferentes thresholds

### Relatórios
- `fairness_report.html` - Relatório interativo em HTML
- `fairness_report.txt` - Relatório em texto simples

## 🔍 Estrutura do Projeto

```
fairness/
├── fairness_metrics.py          # Implementação da métrica ABLNI
├── fairness_visualization.py    # Ferramentas de visualização
├── demo_pima_diabetes.py        # Script de demonstração
├── test_fairness_metrics.py     # Testes unitários
├── requirements.txt             # Dependências
├── README.md                    # Este arquivo
└── outputs/                     # Pasta com resultados (gerada automaticamente)
```

## 📖 Sobre a Métrica ABLNI

**ABLNI (Adjusted Intersectional Net Benefit)** é uma métrica de fairness que:

- Avalia equidade entre subgrupos interseccionais
- Considera o net benefit clínico de cada subgrupo
- Pondera falsos positivos e verdadeiros positivos
- Retorna um score de 0 a 1 (valores maiores = mais equidade)

### Interpretação do Score ABLNI

- **≥ 0.90**: EXCELENTE - Alta equidade
- **0.80-0.89**: BOM - Equidade aceitável
- **0.70-0.79**: MODERADO - Disparidades notáveis
- **< 0.70**: CRÍTICO - Intervenção necessária

## 💻 Uso Programático

### Exemplo Básico

```python
from fairness_metrics import AdjustedIntersectionalNetBenefit
import pandas as pd

# Seus dados
y_true = [0, 1, 0, 1, ...]
y_pred_proba = [0.2, 0.8, 0.3, 0.7, ...]
sensitive_attrs = pd.DataFrame({
    'age_group': ['young', 'old', ...],
    'gender': ['F', 'M', ...]
})

# Calcular fairness
ablni = AdjustedIntersectionalNetBenefit(threshold=0.5)
score = ablni.fit(y_true, y_pred_proba, sensitive_attrs)

print(f"ABLNI Score: {score:.3f}")
print(ablni.get_summary_report())
ablni.plot_subgroup_results()
```

### Comparar Múltiplos Modelos

```python
from fairness_visualizations import FairnessComparator

comparator = FairnessComparator()
comparator.add_model('Model A', y_true, y_pred_a, sensitive_attrs)
comparator.add_model('Model B', y_true, y_pred_b, sensitive_attrs)
comparator.plot_comparison()

# Obter tabela de comparação
comparison_table = comparator.get_comparison_table()
print(comparison_table)
```

## 🧪 Executar Testes

```bash
pytest test_fairness_metrics.py -v
```

## 📚 Dataset Pima Diabetes

O dataset Pima Diabetes contém informações de 768 pacientes mulheres de herança Pima:

- **Features**: Gravidez, glucose, pressão arterial, BMI, idade, etc.
- **Target**: Presença de diabetes (binário)
- **Uso**: Previsão de diabetes tipo 2

Neste projeto, criamos grupos sensíveis baseados em:
- **Idade**: younger vs older (mediana)
- **IMC**: normal_bmi vs high_bmi (mediana)

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests.

## 📄 Licença

MIT License

## 📧 Contato

Para dúvidas ou sugestões, entre em contato com a equipe LABDAPS.

---

**Nota**: Este é um projeto educacional para demonstrar análises de fairness em modelos de ML clínicos.

