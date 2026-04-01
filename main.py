"""
Chest X-Ray Pneumonia Detection — Full Pipeline
=================================================
Runs both Classical ML and Deep Learning pipelines,
then generates a comparative summary report.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = r"c:\Users\baris\Documents\Projects\chest_xray\results"


def run_classical():
    """Run the classical ML pipeline."""
    from classical_pipeline import train_and_evaluate
    return train_and_evaluate()


def run_deep_learning():
    """Run the deep learning pipeline."""
    from deep_learning_pipeline import train_and_evaluate
    return train_and_evaluate()


def generate_comparison_report(classical_results, dl_results):
    """Generate a final comparison report and visualization."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS — Classical vs Deep Learning")
    print("=" * 70)

    # Determine best classical model
    best_classical_name = max(classical_results.keys(),
                              key=lambda k: classical_results[k]['f1_score'])
    best_classical = classical_results[best_classical_name]

    methods = {
        f'Classical: {best_classical_name}': best_classical,
        'Deep Learning: ResNet18': dl_results
    }

    thresholds = {
        f'Classical: {best_classical_name}': 0.80,
        'Deep Learning: ResNet18': 0.90
    }

    metric_names = ['accuracy', 'sensitivity', 'specificity', 'f1_score', 'auc_roc']
    display_names = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score', 'AUC-ROC']

    # Print table
    header = f"{'Metric':<15} | "
    header += " | ".join(f"{name:>28}" for name in methods.keys())
    header += " | Status"
    print(f"\n{header}")
    print("-" * len(header))

    all_pass = True
    for metric, display in zip(metric_names, display_names):
        row = f"{display:<15} | "
        statuses = []
        for name, res in methods.items():
            val = res[metric]
            thresh = thresholds[name]
            status = "PASS" if val >= thresh else "FAIL"
            if val < thresh:
                all_pass = False
            statuses.append(status)
            row += f"{val:>8.4f} (>={thresh:.2f}) [{status}] | "
        print(row)

    # Speed comparison
    print(f"\n{'Speed':<15} | ", end="")
    for name, res in methods.items():
        n = len(res.get('y_test', res.get('y_true', [])))
        t = res['inference_time']
        print(f"  {t:.3f}s total, {t / n * 1000:.2f}ms/image          | ", end="")
    print()

    overall = "ALL METRICS PASS" if all_pass else "SOME METRICS BELOW THRESHOLD"
    print(f"\n  Overall: {overall}")

    # Comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(display_names))
    width = 0.3
    colors = ['#3498db', '#e74c3c']

    for i, (name, res) in enumerate(methods.items()):
        values = [res[m] for m in metric_names]
        bars = ax.bar(x + i * width, values, width, label=name, color=colors[i],
                      edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.axhline(y=0.80, color='blue', linestyle='--', alpha=0.4, label='Classical threshold (0.80)')
    ax.axhline(y=0.90, color='red', linestyle='--', alpha=0.4, label='DL threshold (0.90)')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Classical vs Deep Learning — Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'comparison_chart.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Comparison chart saved to: {os.path.join(RESULTS_DIR, 'comparison_chart.png')}")


def main():
        print("=" * 62)
    print("  Chest X-Ray Pneumonia Detection -- Automated Pipeline")
    print("  Classical ML + Deep Learning with Transfer Learning")
    print("=" * 62 + "\n")

    total_start = time.time()

    # Phase 1: Classical
    print(">> PHASE 1: Classical Machine Learning")
    print("-" * 60)
    classical_results = run_classical()

    # Phase 2: Deep Learning
    print("\n\n>> PHASE 2: Deep Learning (Transfer Learning)")
    print("-" * 60)
    dl_results = run_deep_learning()

    # Phase 3: Comparison
    generate_comparison_report(classical_results, dl_results)

    total_time = time.time() - total_start
    print(f"\n  Total pipeline time: {total_time:.1f}s")
    print("\nDone.")


if __name__ == '__main__':
    main()
