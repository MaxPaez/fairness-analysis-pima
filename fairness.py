"""
Complete Demonstration of Fairness Metrics SDK
This script demonstrates all features of the SDK with a realistic clinical scenario.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Import SDK components
from fairness_metrics import AdjustedIntersectionalNetBenefit, ablni_score
from fairness_visualizations import (
    FairnessVisualizer, 
    FairnessComparator,
    simulate_biased_predictions,
    generate_fairness_report_html
)

# Set random seed
np.random.seed(42)

print("="*80)
print("CLINICAL AI FAIRNESS ANALYSIS - COMPLETE DEMONSTRATION")
print("Scenario: Hospital Readmission Risk Prediction Model")
print("="*80)

# ============================================================================
# STEP 1: Create Realistic Clinical Dataset
# ============================================================================
print("\n" + "="*80)
print("STEP 1: Generating Realistic Clinical Dataset")
print("="*80)

n_patients = 3000

# Clinical features
age = np.random.gamma(6, 10, n_patients)  # Age distribution
comorbidities = np.random.poisson(2, n_patients)  # Number of comorbidities
previous_admissions = np.random.poisson(1.5, n_patients)
lab_value_1 = np.random.normal(100, 15, n_patients)  # e.g., glucose
lab_value_2 = np.random.normal(7, 1.5, n_patients)  # e.g., creatinine
length_of_stay = np.random.gamma(2, 3, n_patients)

# Sensitive attributes
sex = np.random.choice(['Male', 'Female'], n_patients, p=[0.52, 0.48])
race = np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], 
                        n_patients, p=[0.60, 0.20, 0.15, 0.05])
insurance = np.random.choice(['Private', 'Medicare', 'Medicaid'], 
                             n_patients, p=[0.50, 0.30, 0.20])

# Create outcome with realistic relationships and health disparities
risk_score = (
    0.02 * age +
    0.3 * comorbidities +
    0.4 * previous_admissions +
    0.01 * (lab_value_1 - 100) +
    0.15 * (lab_value_2 - 7) +
    0.1 * length_of_stay
)

# Add social determinants effects (health disparities)
for i in range(n_patients):
    if race[i] == 'Black':
        risk_score[i] += 0.5  # Healthcare access disparities
    if insurance[i] == 'Medicaid':
        risk_score[i] += 0.3  # Socioeconomic factors
    if sex[i] == 'Female' and age[i] > 65:
        risk_score[i] += 0.2  # Intersectional effect

# Convert to binary outcome (30-day readmission)
risk_prob = 1 / (1 + np.exp(-risk_score + 3))
y_true = (np.random.random(n_patients) < risk_prob).astype(int)

# Create feature matrix
X = np.column_stack([
    age, comorbidities, previous_admissions,
    lab_value_1, lab_value_2, length_of_stay
])

# Create sensitive attributes DataFrame
sensitive_attrs = pd.DataFrame({
    'sex': sex,
    'race': race,
    'insurance': insurance
})

print(f"✓ Generated dataset with {n_patients} patients")
print(f"  - Outcome prevalence: {y_true.mean():.1%}")
print(f"  - Features: 6 clinical variables")
print(f"  - Sensitive attributes: sex, race, insurance")
print(f"  - Potential intersectional groups: {len(sensitive_attrs.drop_duplicates())}")

# ============================================================================
# STEP 2: Train Multiple Models
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Training Multiple Prediction Models")
print("="*80)

# Split data
X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y_true, sensitive_attrs, test_size=0.3, random_state=42, stratify=y_true
)

print(f"Training set: {len(X_train)} patients")
print(f"Test set: {len(X_test)} patients")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train multiple models
models = {}

# Model 1: Logistic Regression
print("\nTraining Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
models['Logistic Regression'] = lr.predict_proba(X_test_scaled)[:, 1]

# Model 2: Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
models['Random Forest'] = rf.predict_proba(X_test_scaled)[:, 1]

# Model 3: Gradient Boosting
print("Training Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train_scaled, y_train)
models['Gradient Boosting'] = gb.predict_proba(X_test_scaled)[:, 1]

# Model 4: Biased model (simulated - excludes sensitive attributes but still biased)
print("Simulating Biased Model...")
models['Biased Model'] = simulate_biased_predictions(
    y_test, sens_test, bias_strength=0.5, random_state=42
)

print("✓ All models trained successfully")

# ============================================================================
# STEP 3: Individual Model Fairness Analysis
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Detailed Fairness Analysis - Random Forest Model")
print("="*80)

# Analyze best performing model in detail
ablni_rf = AdjustedIntersectionalNetBenefit(
    threshold=0.25,  # Clinically meaningful threshold
    prevalence_weighted=True,
    bootstrap_iterations=500,
    confidence_level=0.95,
    random_state=42
)

score_rf = ablni_rf.fit(y_test, models['Random Forest'], sens_test)

print(f"\nOverall ABLNI Score: {score_rf:.4f}")
if ablni_rf.confidence_interval_:
    print(f"95% CI: ({ablni_rf.confidence_interval_[0]:.4f}, {ablni_rf.confidence_interval_[1]:.4f})")

# Generate comprehensive report
print("\n" + "-"*80)
print(ablni_rf.get_summary_report())
print("-"*80)

# Create visualizations
print("\nGenerating visualizations...")

# Basic plot
ablni_rf.plot_subgroup_results(figsize=(14, 6))

# Advanced visualizations
visualizer = FairnessVisualizer(ablni_rf)
visualizer.plot_comprehensive_dashboard(figsize=(16, 12))

# Calibration curves
visualizer.plot_calibration_curves(
    y_test, models['Random Forest'], sens_test,
    n_bins=10, figsize=(14, 6)
)

# Decision curves
visualizer.plot_decision_curves(
    y_test, models['Random Forest'], sens_test,
    threshold_range=(0.1, 0.5), n_thresholds=50,
    figsize=(14, 6)
)

# ============================================================================
# STEP 4: Compare Multiple Models
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Comparing Fairness Across Multiple Models")
print("="*80)

comparator = FairnessComparator()

# Add all models for comparison
for model_name, predictions in models.items():
    print(f"Evaluating {model_name}...")
    comparator.add_model(
        name=model_name,
        y_true=y_test,
        y_pred_proba=predictions,
        sensitive_attrs=sens_test,
        ablni_kwargs={
            'threshold': 0.25,
            'bootstrap_iterations': 300,
            'random_state': 42
        }
    )

# Generate comparison visualizations
print("\nGenerating comparison plots...")
comparator.plot_comparison(figsize=(16, 10))

# Get comparison table
comparison_table = comparator.get_comparison_table()
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)
print(comparison_table.to_string(index=False))

# ============================================================================
# STEP 5: Intersectional Analysis
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Detailed Intersectional Analysis")
print("="*80)

# Focus on sex and race intersection
sens_intersect = sens_test[['sex', 'race']].copy()

ablni_intersect = AdjustedIntersectionalNetBenefit(
    threshold=0.25,
    min_subgroup_size=20,  # Lower threshold to capture smaller intersections
    bootstrap_iterations=500,
    random_state=42
)

score_intersect = ablni_intersect.fit(
    y_test, 
    models['Random Forest'], 
    sens_intersect
)

print(f"\nIntersectional ABLNI Score: {score_intersect:.4f}")
print(f"\nNumber of intersectional groups analyzed: {len(ablni_intersect.subgroup_results_)}")

print("\nIntersectional subgroup performance:")
intersect_results = ablni_intersect.subgroup_results_.sort_values('net_benefit')
print(intersect_results[['subgroup', 'n', 'prevalence', 'net_benefit', 'tpr', 'tnr']].to_string(index=False))

# Visualize
ablni_intersect.plot_subgroup_results(figsize=(14, 8))

# ============================================================================
# STEP 6: Sensitivity Analysis - Different Thresholds
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Sensitivity Analysis Across Thresholds")
print("="*80)

thresholds_to_test = [0.15, 0.20, 0.25, 0.30, 0.35]
threshold_results = []

for thresh in thresholds_to_test:
    ablni_thresh = AdjustedIntersectionalNetBenefit(
        threshold=thresh,
        bootstrap_iterations=0,  # Skip bootstrap for speed
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
print("\nFairness across different clinical thresholds:")
print(threshold_df.to_string(index=False))

# Plot sensitivity
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(threshold_df['threshold'], threshold_df['ablni_score'], 'o-', linewidth=2, markersize=8)
axes[0].set_xlabel('Decision Threshold')
axes[0].set_ylabel('ABLNI Score')
axes[0].set_title('Fairness vs Threshold')
axes[0].grid(alpha=0.3)
axes[0].axhline(0.8, color='orange', linestyle='--', label='Acceptable')
axes[0].legend()

axes[1].plot(threshold_df['threshold'], threshold_df['min_net_benefit'], 'o-', linewidth=2, markersize=8, label='Min')
axes[1].plot(threshold_df['threshold'], threshold_df['max_net_benefit'], 'o-', linewidth=2, markersize=8, label='Max')
axes[1].fill_between(threshold_df['threshold'], threshold_df['min_net_benefit'], 
                      threshold_df['max_net_benefit'], alpha=0.3)
axes[1].set_xlabel('Decision Threshold')
axes[1].set_ylabel('Net Benefit')
axes[1].set_title('Net Benefit Range')
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(threshold_df['threshold'], threshold_df['range'], 'o-', linewidth=2, markersize=8, color='red')
axes[2].set_xlabel('Decision Threshold')
axes[2].set_ylabel('Net Benefit Range')
axes[2].set_title('Disparity Magnitude')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 7: Custom Harm-to-Benefit Ratios
# ============================================================================
print("\n" + "="*80)
print("STEP 7: Context-Specific Harm-to-Benefit Ratios")
print("="*80)

print("\nScenario: Higher cost of false positives for certain subgroups")
print("(e.g., due to invasive follow-up procedures or financial burden)")

# Create subgroup labels
subgroup_labels = sens_test.astype(str).agg('_'.join, axis=1)
unique_subgroups = subgroup_labels.unique()

# Define custom weights
# For example, higher weight for groups with limited healthcare access
custom_weights = {}
for subgroup in unique_subgroups:
    if 'Medicaid' in subgroup or 'Black' in subgroup:
        custom_weights[subgroup] = 1.5  # Higher penalty for false positives
    else:
        custom_weights[subgroup] = 1.0

ablni_custom = AdjustedIntersectionalNetBenefit(
    threshold=0.25,
    harm_to_benefit_ratio=custom_weights,
    bootstrap_iterations=300,
    random_state=42
)

score_custom = ablni_custom.fit(y_test, models['Random Forest'], sens_test)

print(f"\nABLNI with custom weights: {score_custom:.4f}")
print(f"ABLNI with standard weights: {score_rf:.4f}")
print(f"Difference: {score_custom - score_rf:.4f}")

# ============================================================================
# STEP 8: Generate Reports
# ============================================================================
print("\n" + "="*80)
print("STEP 8: Generating Comprehensive Reports")
print("="*80)

# Save results to CSV
print("\nExporting results...")
ablni_rf.subgroup_results_.to_csv('fairness_detailed_results.csv', index=False)
comparison_table.to_csv('model_comparison.csv', index=False)
print("✓ CSV files exported")

# Generate HTML report
generate_fairness_report_html(ablni_rf, 'fairness_analysis_report.html')
print("✓ HTML report generated")

# Generate text report
with open('fairness_analysis_report.txt', 'w') as f:
    f.write(ablni_rf.get_summary_report())
    f.write("\n\n" + "="*80 + "\n")
    f.write("MODEL COMPARISON\n")
    f.write("="*80 + "\n\n")
    f.write(comparison_table.to_string(index=False))
print("✓ Text report generated")

# ============================================================================
# STEP 9: Recommendations
# ============================================================================
print("\n" + "="*80)
print("STEP 9: Actionable Recommendations")
print("="*80)

# Identify worst-performing subgroups
worst_subgroups = ablni_rf.subgroup_results_.nsmallest(3, 'net_benefit')

print("\n🔍 KEY FINDINGS:")
print(f"\n1. Overall Model Fairness: {score_rf:.4f}")
if score_rf >= 0.9:
    print("   → Excellent fairness - model performs consistently across subgroups")
elif score_rf >= 0.8:
    print("   → Good fairness - minor disparities present")
elif score_rf >= 0.7:
    print("   → Moderate fairness - notable disparities warrant attention")
else:
    print("   → Poor fairness - significant disparities require intervention")

print(f"\n2. Most Vulnerable Subgroups:")
for idx, row in worst_subgroups.iterrows():
    print(f"   → {row['subgroup']}: Net Benefit = {row['net_benefit']:.4f} (n={row['n']})")

print(f"\n3. Performance Range:")
nb_range = ablni_rf.subgroup_results_['net_benefit'].max() - ablni_rf.subgroup_results_['net_benefit'].min()
print(f"   → Net benefit ranges from {ablni_rf.subgroup_results_['net_benefit'].min():.4f} to {ablni_rf.subgroup_results_['net_benefit'].max():.4f}")
print(f"   → Total range: {nb_range:.4f}")

print("\n💡 RECOMMENDATIONS:")

recommendations = []

if score_rf < 0.8:
    recommendations.append("1. Implement fairness-aware training methods (e.g., reweighting, adversarial debiasing)")
    recommendations.append("2. Consider separate models for high-disparity subgroups")
    recommendations.append("3. Collect additional data for underperforming subgroups")

if ablni_rf.subgroup_results_['n'].min() < 100:
    recommendations.append("4. Increase sample size for small subgroups to improve reliability")

if nb_range > 0.1:
    recommendations.append("5. Investigate root causes of disparities through clinical review")
    recommendations.append("6. Engage stakeholders from affected communities in model development")

recommendations.append("7. Establish continuous monitoring for fairness drift in production")
recommendations.append("8. Document fairness considerations in model card/fact sheet")
recommendations.append("9. Consider context-specific harm-to-benefit ratios for different populations")

for rec in recommendations:
    print(f"   {rec}")

print("\n⚠️  IMPORTANT CONSIDERATIONS:")
print("   • Fairness metrics quantify disparities but don't automatically solve them")
print("   • Clinical validation with diverse stakeholders is essential")
print("   • Monitor for fairness drift after deployment")
print("   • Consider both individual and group fairness perspectives")
print("   • Balance fairness objectives with clinical utility")

# ============================================================================
# STEP 10: Quick API Usage Examples
# ============================================================================
print("\n" + "="*80)
print("STEP 10: Quick API Usage Examples")
print("="*80)

print("\nExample 1: Minimal usage with convenience function")
print("-" * 50)
print("```python")
print("from fairness_metrics import ablni_score")
print("")
print("score = ablni_score(")
print("    y_true, y_pred_proba, sensitive_attrs,")
print("    threshold=0.3")
print(")")
print("print(f'ABLNI Score: {score:.3f}')")
print("```")

print("\nExample 2: Full analysis with visualization")
print("-" * 50)
print("```python")
print("from fairness_metrics import AdjustedIntersectionalNetBenefit")
print("")
print("ablni = AdjustedIntersectionalNetBenefit(")
print("    threshold=0.25,")
print("    bootstrap_iterations=1000")
print(")")
print("score = ablni.fit(y_true, y_pred_proba, sensitive_attrs)")
print("print(ablni.get_summary_report())")
print("ablni.plot_subgroup_results()")
print("```")

print("\nExample 3: Model comparison")
print("-" * 50)
print("```python")
print("from fairness_visualizations import FairnessComparator")
print("")
print("comparator = FairnessComparator()")
print("comparator.add_model('Model A', y_true, y_pred_a, sensitive_attrs)")
print("comparator.add_model('Model B', y_true, y_pred_b, sensitive_attrs)")
print("comparator.plot_comparison()")
print("```")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

print("\n📊 Generated Outputs:")
print("   • fairness_detailed_results.csv")
print("   • model_comparison.csv")
print("   • fairness_analysis_report.html")
print("   • fairness_analysis_report.txt")
print("   • Multiple visualization plots")

print("\n📈 Models Evaluated:")
for model_name in models.keys():
    model_score = comparator.results[model_name]['score']
    print(f"   • {model_name}: ABLNI = {model_score:.4f}")

print("\n✅ Best Model: " + comparison_table.loc[comparison_table['ABLNI_Score'].idxmax(), 'Model'])
print(f"   ABLNI Score: {comparison_table['ABLNI_Score'].max():.4f}")

print("\n" + "="*80)
print("Thank you for using the Fairness Metrics SDK!")
print("For questions or feedback: [your contact information]")
print("="*80)