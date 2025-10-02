"""
Fairness Metrics SDK - Core Implementation
Adjusted Intersectional Net Benefit (ABLNI) Metric
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional, Dict, List, Tuple
from scipy import stats
import warnings


class AdjustedIntersectionalNetBenefit:
    """
    Adjusted Intersectional Net Benefit (ABLNI) - Uma métrica de fairness
    que avalia equidade em predições de modelos de ML considerando 
    subgrupos interseccionais.
    
    Parameters
    ----------
    threshold : float ou 'optimal'
        Limiar de decisão para converter probabilidades em predições binárias.
        Se 'optimal', calcula o limiar ótimo para cada subgrupo.
    harm_to_benefit_ratio : float, dict ou 'auto'
        Razão de custo entre falsos positivos e verdadeiros positivos.
        Se dict, mapeia subgrupos para suas razões específicas.
        Se 'auto', calcula automaticamente do threshold.
    min_subgroup_size : int
        Tamanho mínimo de subgrupo para inclusão na análise.
    prevalence_weighted : bool
        Se True, pondera métricas pela prevalência do outcome.
    bootstrap_iterations : int
        Número de iterações bootstrap para intervalos de confiança.
    confidence_level : float
        Nível de confiança para intervalos (0-1).
    random_state : int, opcional
        Seed para reprodutibilidade.
    """
    
    def __init__(
        self,
        threshold: Union[float, str] = 0.5,
        harm_to_benefit_ratio: Union[float, Dict, str] = 'auto',
        min_subgroup_size: int = 50,
        prevalence_weighted: bool = True,
        bootstrap_iterations: int = 1000,
        confidence_level: float = 0.95,
        random_state: Optional[int] = None
    ):
        self.threshold = threshold
        self.harm_to_benefit_ratio = harm_to_benefit_ratio
        self.min_subgroup_size = min_subgroup_size
        self.prevalence_weighted = prevalence_weighted
        self.bootstrap_iterations = bootstrap_iterations
        self.confidence_level = confidence_level
        self.random_state = random_state
        
        # Atributos definidos após fit
        self.overall_score_ = None
        self.subgroup_results_ = None
        self.confidence_interval_ = None
        
    def fit(
        self, 
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        sensitive_attrs: Union[pd.DataFrame, pd.Series, np.ndarray]
    ) -> float:
        """
        Calcula a métrica ABLNI.
        
        Parameters
        ----------
        y_true : array-like
            Labels verdadeiros (0 ou 1).
        y_pred_proba : array-like
            Probabilidades preditas.
        sensitive_attrs : DataFrame, Series ou array
            Atributos sensíveis para definir subgrupos.
            
        Returns
        -------
        float
            Score ABLNI (0-1, valores maiores indicam mais equidade).
        """
        # Validação de entrada
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)
        
        if len(y_true) != len(y_pred_proba):
            raise ValueError("y_true e y_pred_proba devem ter o mesmo tamanho")
        
        # Converter sensitive_attrs para DataFrame se necessário
        if isinstance(sensitive_attrs, pd.Series):
            sensitive_attrs = sensitive_attrs.to_frame()
        elif isinstance(sensitive_attrs, np.ndarray):
            if sensitive_attrs.ndim == 1:
                sensitive_attrs = pd.DataFrame({'group': sensitive_attrs})
            else:
                sensitive_attrs = pd.DataFrame(sensitive_attrs)
        elif not isinstance(sensitive_attrs, pd.DataFrame):
            raise ValueError("sensitive_attrs deve ser DataFrame, Series ou array")
        
        if len(sensitive_attrs) != len(y_true):
            raise ValueError("sensitive_attrs deve ter o mesmo tamanho que y_true")
        
        # Criar labels de subgrupos interseccionais
        subgroup_labels = sensitive_attrs.astype(str).agg('_'.join, axis=1)
        
        # Calcular métricas para cada subgrupo
        subgroup_results = []
        unique_subgroups = subgroup_labels.unique()
        
        for subgroup in unique_subgroups:
            mask = subgroup_labels == subgroup
            n_subgroup = mask.sum()
            
            # Verificar tamanho mínimo
            if n_subgroup < self.min_subgroup_size:
                warnings.warn(
                    f"Subgrupo '{subgroup}' tem apenas {n_subgroup} amostras "
                    f"(mínimo: {self.min_subgroup_size}). Será excluído.",
                    UserWarning
                )
                continue
            
            # Extrair dados do subgrupo
            y_true_sub = y_true[mask]
            y_pred_proba_sub = y_pred_proba[mask]
            
            # Determinar threshold
            if self.threshold == 'optimal':
                threshold_sub = self._find_optimal_threshold(y_true_sub, y_pred_proba_sub)
            else:
                threshold_sub = self.threshold
            
            # Converter para predições binárias
            y_pred_binary = (y_pred_proba_sub >= threshold_sub).astype(int)
            
            # Calcular confusion matrix
            tp = np.sum((y_pred_binary == 1) & (y_true_sub == 1))
            tn = np.sum((y_pred_binary == 0) & (y_true_sub == 0))
            fp = np.sum((y_pred_binary == 1) & (y_true_sub == 0))
            fn = np.sum((y_pred_binary == 0) & (y_true_sub == 1))
            
            # Calcular métricas
            prevalence = y_true_sub.mean()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
            
            # Calcular weight (harm-to-benefit ratio)
            if isinstance(self.harm_to_benefit_ratio, dict):
                weight = self.harm_to_benefit_ratio.get(subgroup, 1.0)
            elif self.harm_to_benefit_ratio == 'auto':
                weight = (1 - threshold_sub) / threshold_sub if threshold_sub > 0 else 1.0
            else:
                weight = float(self.harm_to_benefit_ratio)
            
            # Calcular Net Benefit
            net_benefit = self._compute_net_benefit(y_true_sub, y_pred_binary, weight)
            
            subgroup_results.append({
                'subgroup': subgroup,
                'net_benefit': net_benefit,
                'threshold': threshold_sub,
                'weight': weight,
                'n': n_subgroup,
                'prevalence': prevalence,
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tpr': tpr,
                'tnr': tnr,
                'ppv': ppv,
                'npv': npv
            })
        
        if len(subgroup_results) == 0:
            raise ValueError("Nenhum subgrupo atende aos critérios mínimos")
        
        # Converter para DataFrame
        self.subgroup_results_ = pd.DataFrame(subgroup_results)
        
        # Calcular score ABLNI global
        self.overall_score_ = self._compute_ablni_score(self.subgroup_results_)
        
        # Bootstrap para intervalos de confiança
        if self.bootstrap_iterations > 0:
            self.confidence_interval_ = self._bootstrap_ci(
                y_true, y_pred_proba, subgroup_labels
            )
        else:
            self.confidence_interval_ = None
        
        return self.overall_score_
    
    def _compute_net_benefit(
        self, 
        y_true: np.ndarray, 
        y_pred_binary: np.ndarray, 
        weight: float
    ) -> float:
        """Calcula Net Benefit."""
        n = len(y_true)
        if n == 0:
            return 0.0
        
        tp = np.sum((y_pred_binary == 1) & (y_true == 1))
        fp = np.sum((y_pred_binary == 1) & (y_true == 0))
        
        net_benefit = (tp / n) - (fp / n) * weight
        return net_benefit
    
    def _find_optimal_threshold(
        self, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> float:
        """Encontra o threshold ótimo usando Youden's J statistic."""
        thresholds = np.linspace(0.1, 0.9, 50)
        best_threshold = 0.5
        best_j = -1
        
        for thresh in thresholds:
            y_pred_binary = (y_pred_proba >= thresh).astype(int)
            
            tp = np.sum((y_pred_binary == 1) & (y_true == 1))
            tn = np.sum((y_pred_binary == 0) & (y_true == 0))
            fp = np.sum((y_pred_binary == 1) & (y_true == 0))
            fn = np.sum((y_pred_binary == 0) & (y_true == 1))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            j = tpr + tnr - 1  # Youden's J
            
            if j > best_j:
                best_j = j
                best_threshold = thresh
        
        return best_threshold
    
    def _compute_ablni_score(self, results_df: pd.DataFrame) -> float:
        """
        Calcula o score ABLNI geral baseado nos net benefits dos subgrupos.
        
        ABLNI = min(NB) / max(NB) se todos NB > 0
        """
        net_benefits = results_df['net_benefit'].values
        
        if len(net_benefits) == 0:
            return np.nan
        
        min_nb = net_benefits.min()
        max_nb = net_benefits.max()
        
        if max_nb <= 0:
            # Se todos são negativos ou zero, retorna 0
            return 0.0
        
        if min_nb < 0:
            # Se o mínimo é negativo, ajusta a escala
            score = max(0, (min_nb + max_nb) / (2 * max_nb))
        else:
            # Razão entre min e max
            score = min_nb / max_nb if max_nb > 0 else 0.0
        
        return float(score)
    
    def _bootstrap_ci(
        self, 
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        subgroup_labels: pd.Series
    ) -> Tuple[float, float]:
        """Calcula intervalo de confiança via bootstrap."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        bootstrap_scores = []
        n = len(y_true)
        
        for _ in range(self.bootstrap_iterations):
            # Amostra com reposição
            indices = np.random.choice(n, size=n, replace=True)
            
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred_proba[indices]
            subgroup_boot = subgroup_labels.iloc[indices].reset_index(drop=True)
            
            try:
                # Criar instância temporária sem bootstrap
                ablni_boot = AdjustedIntersectionalNetBenefit(
                    threshold=self.threshold,
                    harm_to_benefit_ratio=self.harm_to_benefit_ratio,
                    min_subgroup_size=max(10, self.min_subgroup_size // 2),
                    bootstrap_iterations=0,
                    random_state=None
                )
                
                score_boot = ablni_boot.fit(
                    y_true_boot, 
                    y_pred_boot, 
                    pd.DataFrame({'group': subgroup_boot})
                )
                
                if not np.isnan(score_boot):
                    bootstrap_scores.append(score_boot)
            except:
                continue
        
        if len(bootstrap_scores) > 0:
            alpha = 1 - self.confidence_level
            ci_lower = np.percentile(bootstrap_scores, alpha/2 * 100)
            ci_upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
            return (ci_lower, ci_upper)
        else:
            return (np.nan, np.nan)
    
    def get_summary_report(self) -> str:
        """Gera relatório resumido em texto."""
        if self.subgroup_results_ is None:
            return "Métrica não foi ajustada ainda. Use fit() primeiro."
        
        report = []
        report.append("=" * 70)
        report.append("ANÁLISE DE FAIRNESS - ABLNI")
        report.append("=" * 70)
        report.append(f"\nScore ABLNI Geral: {self.overall_score_:.4f}")
        
        if self.confidence_interval_ is not None and not np.isnan(self.confidence_interval_[0]):
            ci = self.confidence_interval_
            report.append(f"Intervalo de Confiança {self.confidence_level*100:.0f}%: "
                         f"({ci[0]:.4f}, {ci[1]:.4f})")
        
        report.append(f"\nNúmero de Subgrupos Analisados: {len(self.subgroup_results_)}")
        
        report.append("\n" + "-" * 70)
        report.append("RESULTADOS POR SUBGRUPO")
        report.append("-" * 70)
        
        for _, row in self.subgroup_results_.iterrows():
            report.append(f"\nSubgrupo: {row['subgroup']}")
            report.append(f"  Tamanho: {row['n']}")
            report.append(f"  Prevalência: {row['prevalence']:.1%}")
            report.append(f"  Net Benefit: {row['net_benefit']:.4f}")
            report.append(f"  Sensibilidade (TPR): {row['tpr']:.3f}")
            report.append(f"  Especificidade (TNR): {row['tnr']:.3f}")
            report.append(f"  Precision (PPV): {row['ppv']:.3f}")
        
        report.append("\n" + "=" * 70)
        
        # Interpretação
        score = self.overall_score_
        report.append("\nINTERPRETACAO:")
        if score >= 0.9:
            report.append("[OK] EXCELENTE - Alta equidade entre subgrupos")
        elif score >= 0.8:
            report.append("[OK] BOM - Equidade aceitavel com pequenas disparidades")
        elif score >= 0.7:
            report.append("[AVISO] MODERADO - Disparidades notaveis, atencao necessaria")
        else:
            report.append("[CRITICO] Disparidades significativas requerem intervencao")
        
        report.append("=" * 70)
        
        return "\n".join(report)
    
    def plot_subgroup_results(
        self, 
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ):
        """
        Plota resultados por subgrupo.
        
        Parameters
        ----------
        figsize : tuple
            Tamanho da figura.
        save_path : str, opcional
            Caminho para salvar a figura.
        """
        if self.subgroup_results_ is None:
            raise ValueError("Métrica não foi ajustada ainda. Use fit() primeiro.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        df = self.subgroup_results_.sort_values('net_benefit')
        
        # Plot 1: Net Benefit por subgrupo
        colors = ['#d62728' if i == 0 else '#2ca02c' if i == len(df)-1 
                 else '#1f77b4' for i in range(len(df))]
        
        ax1.barh(range(len(df)), df['net_benefit'], color=colors, alpha=0.7)
        ax1.set_yticks(range(len(df)))
        ax1.set_yticklabels(df['subgroup'])
        ax1.set_xlabel('Net Benefit')
        ax1.set_title('Net Benefit por Subgrupo')
        ax1.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(axis='x', alpha=0.3)
        
        # Plot 2: Métricas de performance
        metrics = df[['tpr', 'tnr', 'ppv', 'npv']].values.T
        x = np.arange(len(df))
        width = 0.2
        
        ax2.bar(x - 1.5*width, metrics[0], width, label='Sensibilidade', alpha=0.8)
        ax2.bar(x - 0.5*width, metrics[1], width, label='Especificidade', alpha=0.8)
        ax2.bar(x + 0.5*width, metrics[2], width, label='Precision', alpha=0.8)
        ax2.bar(x + 1.5*width, metrics[3], width, label='NPV', alpha=0.8)
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['subgroup'], rotation=45, ha='right')
        ax2.set_ylabel('Score')
        ax2.set_title('Métricas de Performance')
        ax2.legend()
        ax2.set_ylim([0, 1])
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figura salva em: {save_path}")
        
        plt.show()


def ablni_score(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    sensitive_attrs: Union[pd.DataFrame, pd.Series, np.ndarray],
    threshold: Union[float, str] = 0.5,
    **kwargs
) -> float:
    """
    Função de conveniência para calcular ABLNI score rapidamente.
    
    Parameters
    ----------
    y_true : array-like
        Labels verdadeiros.
    y_pred_proba : array-like
        Probabilidades preditas.
    sensitive_attrs : DataFrame, Series ou array
        Atributos sensíveis.
    threshold : float ou 'optimal'
        Limiar de decisão.
    **kwargs
        Argumentos adicionais para AdjustedIntersectionalNetBenefit.
        
    Returns
    -------
    float
        Score ABLNI.
    """
    ablni = AdjustedIntersectionalNetBenefit(threshold=threshold, **kwargs)
    return ablni.fit(y_true, y_pred_proba, sensitive_attrs)

