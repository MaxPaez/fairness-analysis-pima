"""
Unit Tests for Fairness Metrics SDK
Run with: pytest test_fairness_metrics.py -v
"""

import pytest
import numpy as np
import pandas as pd
from fairness_metrics import AdjustedIntersectionalNetBenefit, ablni_score


class TestABLNI:
    """Test suite for ABLNI metric"""
    
    @pytest.fixture
    def simple_data(self):
        """Create simple test data"""
        np.random.seed(42)
        n = 500
        y_true = np.random.binomial(1, 0.3, n)
        y_pred_proba = np.random.beta(2, 3, n)
        sensitive_attrs = pd.Series(np.random.choice(['A', 'B'], n))
        return y_true, y_pred_proba, sensitive_attrs
    
    @pytest.fixture
    def intersectional_data(self):
        """Create intersectional test data"""
        np.random.seed(42)
        n = 1000
        y_true = np.random.binomial(1, 0.25, n)
        y_pred_proba = np.random.beta(2, 4, n)
        sensitive_attrs = pd.DataFrame({
            'sex': np.random.choice(['M', 'F'], n),
            'race': np.random.choice(['White', 'Black', 'Asian'], n)
        })
        return y_true, y_pred_proba, sensitive_attrs
    
    def test_basic_computation(self, simple_data):
        """Test basic ABLNI computation"""
        y_true, y_pred_proba, sensitive_attrs = simple_data
        
        ablni = AdjustedIntersectionalNetBenefit(
            threshold=0.5,
            bootstrap_iterations=0
        )
        score = ablni.fit(y_true, y_pred_proba, sensitive_attrs)
        
        assert isinstance(score, (float, np.floating))
        assert 0 <= score <= 1 or score < 0  # Can be negative if min NB is negative
        assert ablni.subgroup_results_ is not None
        assert len(ablni.subgroup_results_) == 2  # Two subgroups: A and B
    
    def test_intersectional_groups(self, intersectional_data):
        """Test intersectional group creation"""
        y_true, y_pred_proba, sensitive_attrs = intersectional_data
        
        ablni = AdjustedIntersectionalNetBenefit(
            threshold=0.3,
            min_subgroup_size=30,
            bootstrap_iterations=0
        )
        score = ablni.fit(y_true, y_pred_proba, sensitive_attrs)
        
        # Should have multiple intersectional groups (M_White, M_Black, etc.)
        assert len(ablni.subgroup_results_) >= 4
        
        # Check that subgroup names contain both attributes
        for subgroup in ablni.subgroup_results_['subgroup']:
            assert '_' in subgroup
    
    def test_threshold_options(self, simple_data):
        """Test different threshold options"""
        y_true, y_pred_proba, sensitive_attrs = simple_data
        
        # Test fixed threshold
        ablni_fixed = AdjustedIntersectionalNetBenefit(
            threshold=0.4,
            bootstrap_iterations=0
        )
        score_fixed = ablni_fixed.fit(y_true, y_pred_proba, sensitive_attrs)
        
        # Test optimal threshold
        ablni_optimal = AdjustedIntersectionalNetBenefit(
            threshold='optimal',
            bootstrap_iterations=0
        )
        score_optimal = ablni_optimal.fit(y_true, y_pred_proba, sensitive_attrs)
        
        # Both should produce valid scores
        assert not np.isnan(score_fixed)
        assert not np.isnan(score_optimal)
        
        # Optimal thresholds should be different per subgroup
        thresholds = ablni_optimal.subgroup_results_['threshold'].values
        assert len(np.unique(thresholds)) > 1 or len(thresholds) == 1
    
    def test_minimum_subgroup_size(self):
        """Test minimum subgroup size enforcement"""
        np.random.seed(42)
        n = 100
        y_true = np.random.binomial(1, 0.3, n)
        y_pred_proba = np.random.beta(2, 3, n)
        
        # Create imbalanced groups
        sensitive_attrs = pd.Series(['A'] * 80 + ['B'] * 20)
        
        ablni = AdjustedIntersectionalNetBenefit(
            threshold=0.5,
            min_subgroup_size=30,
            bootstrap_iterations=0
        )
        
        with pytest.warns(UserWarning):
            score = ablni.fit(y_true, y_pred_proba, sensitive_attrs)
        
        # Only group A should be included
        assert len(ablni.subgroup_results_) == 1
        assert ablni.subgroup_results_.iloc[0]['subgroup'] == 'A'
    
    def test_bootstrap_confidence_intervals(self, simple_data):
        """Test bootstrap CI computation"""
        y_true, y_pred_proba, sensitive_attrs = simple_data
        
        ablni = AdjustedIntersectionalNetBenefit(
            threshold=0.5,
            bootstrap_iterations=100,
            random_state=42
        )
        score = ablni.fit(y_true, y_pred_proba, sensitive_attrs)
        
        assert ablni.confidence_interval_ is not None
        ci_lower, ci_upper = ablni.confidence_interval_
        assert ci_lower <= score <= ci_upper
    
    def test_custom_harm_ratio(self, simple_data):
        """Test custom harm-to-benefit ratios"""
        y_true, y_pred_proba, sensitive_attrs = simple_data
        
        # Different weights per subgroup
        harm_ratios = {'A': 1.0, 'B': 2.0}
        
        ablni = AdjustedIntersectionalNetBenefit(
            threshold=0.5,
            harm_to_benefit_ratio=harm_ratios,
            bootstrap_iterations=0
        )
        score = ablni.fit(y_true, y_pred_proba, sensitive_attrs)
        
        # Check that weights were applied
        results = ablni.subgroup_results_
        assert results[results['subgroup'] == 'A']['weight'].values[0] == 1.0
        assert results[results['subgroup'] == 'B']['weight'].values[0] == 2.0
    
    def test_input_validation(self):
        """Test input validation"""
        y_true = np.array([0, 1, 0, 1])
        y_pred_proba = np.array([0.2, 0.8, 0.3, 0.7])
        sensitive_attrs = pd.Series(['A', 'A', 'B'])  # Wrong length
        
        ablni = AdjustedIntersectionalNetBenefit(bootstrap_iterations=0)
        
        with pytest.raises(ValueError):
            ablni.fit(y_true, y_pred_proba, sensitive_attrs)
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Perfect predictions
        y_true = np.array([0, 0, 1, 1])
        y_pred_proba = np.array([0.0, 0.0, 1.0, 1.0])
        sensitive_attrs = pd.Series(['A', 'A', 'B', 'B'])
        
        ablni = AdjustedIntersectionalNetBenefit(
            threshold=0.5,
            bootstrap_iterations=0
        )
        score = ablni.fit(y_true, y_pred_proba, sensitive_attrs)
        
        # Should have high fairness score for perfect predictions
        assert score > 0.9
    
    def test_convenience_function(self, simple_data):
        """Test convenience function"""
        y_true, y_pred_proba, sensitive_attrs = simple_data
        
        score = ablni_score(
            y_true, y_pred_proba, sensitive_attrs,
            threshold=0.5,
            bootstrap_iterations=0
        )
        
        assert isinstance(score, (float, np.floating))
    
    def test_subgroup_results_structure(self, simple_data):
        """Test structure of subgroup results DataFrame"""
        y_true, y_pred_proba, sensitive_attrs = simple_data
        
        ablni = AdjustedIntersectionalNetBenefit(
            threshold=0.5,
            bootstrap_iterations=0
        )
        ablni.fit(y_true, y_pred_proba, sensitive_attrs)
        
        results = ablni.subgroup_results_
        
        # Check required columns
        required_cols = [
            'subgroup', 'net_benefit', 'threshold', 'weight',
            'n', 'prevalence', 'tp', 'tn', 'fp', 'fn',
            'tpr', 'tnr', 'ppv', 'npv'
        ]
        
        for col in required_cols:
            assert col in results.columns
        
        # Check data types
        assert results['net_benefit'].dtype in [np.float64, np.float32]
        assert results['n'].dtype in [np.int64, np.int32]
    
    def test_summary_report_generation(self, simple_data):
        """Test summary report generation"""
        y_true, y_pred_proba, sensitive_attrs = simple_data
        
        ablni = AdjustedIntersectionalNetBenefit(
            threshold=0.5,
            bootstrap_iterations=0
        )
        ablni.fit(y_true, y_pred_proba, sensitive_attrs)
        
        report = ablni.get_summary_report()
        
        assert isinstance(report, str)
        assert 'ABLNI' in report
        assert 'Subgroup' in report or 'subgroup' in report


class TestNetBenefitCalculation:
    """Test net benefit calculation specifically"""
    
    def test_perfect_classification(self):
        """Test net benefit with perfect classification"""
        ablni = AdjustedIntersectionalNetBenefit(bootstrap_iterations=0)
        
        # All correct predictions
        y_true = np.array([1, 1, 0, 0])
        y_pred_binary = np.array([1, 1, 0, 0])
        weight = 1.0
        
        nb = ablni._compute_net_benefit(y_true, y_pred_binary, weight)
        
        # NB = (TP/n) - (FP/n) * w = (2/4) - (0/4) * 1 = 0.5
        assert np.isclose(nb, 0.5)
    
    def test_all_wrong_classification(self):
        """Test net benefit with all wrong predictions"""
        ablni = AdjustedIntersectionalNetBenefit(bootstrap_iterations=0)
        
        y_true = np.array([1, 1, 0, 0])
        y_pred_binary = np.array([0, 0, 1, 1])
        weight = 1.0
        
        nb = ablni._compute_net_benefit(y_true, y_pred_binary, weight)
        
        # NB = (TP/n) - (FP/n) * w = (0/4) - (2/4) * 1 = -0.5
        assert np.isclose(nb, -0.5)
    
    def test_different_weights(self):
        """Test net benefit with different harm-to-benefit ratios"""
        ablni = AdjustedIntersectionalNetBenefit(bootstrap_iterations=0)
        
        y_true = np.array([1, 1, 0, 0])
        y_pred_binary = np.array([1, 1, 1, 0])  # 2 TP, 1 FP
        
        # With weight 1.0
        nb1 = ablni._compute_net_benefit(y_true, y_pred_binary, weight=1.0)
        # NB = (2/4) - (1/4) * 1.0 = 0.25
        
        # With weight 2.0 (FPs are twice as bad)
        nb2 = ablni._compute_net_benefit(y_true, y_pred_binary, weight=2.0)
        # NB = (2/4) - (1/4) * 2.0 = 0.0
        
        assert np.isclose(nb1, 0.25)
        assert np.isclose(nb2, 0.0)
        assert nb1 > nb2  # Higher weight decreases net benefit


# Run tests
if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])


# ============================================================================
# INSTALLATION GUIDE
# ============================================================================

"""
# Fairness Metrics SDK - Installation Guide

## Requirements

```bash
pip install numpy pandas scikit-learn scipy matplotlib pytest
```

## Installation

### Option 1: Direct Installation (Recommended for development)

1. Save the SDK code as `fairness_metrics.py`
2. Place it in your project directory or Python path
3. Import and use:

```python
from fairness_metrics import AdjustedIntersectionalNetBenefit, ablni_score
```

### Option 2: Package Installation (For production)

Create a `setup.py`:

```python
from setuptools import setup, find_packages

setup(
    name='fairness-metrics',
    version='0.1.0',
    description='Clinical AI Fairness Metrics SDK',
    author='Your Name',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'scipy>=1.7.0',
        'matplotlib>=3.4.0'
    ],
    python_requires='>=3.8',
)
```

Then install:

```bash
pip install -e .
```

## Quick Start

```python
import numpy as np
import pandas as pd
from fairness_metrics import AdjustedIntersectionalNetBenefit

# Your model predictions
y_true = np.array([...])  # True labels
y_pred_proba = np.array([...])  # Predicted probabilities
sensitive_attrs = pd.DataFrame({
    'sex': [...],
    'race': [...]
})

# Compute fairness
ablni = AdjustedIntersectionalNetBenefit(threshold=0.3)
score = ablni.fit(y_true, y_pred_proba, sensitive_attrs)

# View results
print(f"ABLNI Score: {score:.3f}")
print(ablni.get_summary_report())
ablni.plot_subgroup_results()
```

## Running Tests

```bash
pytest test_fairness_metrics.py -v
```

## Documentation

Full documentation available at: [Add your documentation URL]

## Citation

If you use this SDK in your research, please cite:

```bibtex
@software{fairness_metrics_sdk,
  title={Fairness Metrics SDK for Clinical AI},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/fairness-metrics}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please submit pull requests or open issues on GitHub.

## Support

For questions or issues:
- Email: your.email@example.com
- GitHub Issues: [Your GitHub URL]
"""