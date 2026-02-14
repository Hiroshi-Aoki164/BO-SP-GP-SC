import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

result_dir = Path('result')

methods = {
    'BO(SP-GPR-SC)': result_dir / 'BO(SP_GPR_SC)' / 'y_all.csv',
    'BO(SP-GPR)': result_dir / 'BO(SP_GPR)' / 'y_all.csv',
    'BO(GPR)': result_dir / 'BO(GPR)' / 'y_all.csv',
    'MLR': result_dir / 'MLR' / 'y_all.csv'
}

y_target = 32

print("=== Wilcoxon signed-rank test for all iterations ===\n")

method_data_all = {}

for method_name, csv_path in methods.items():
    print(f"Loading {method_name}...")

    df = pd.read_csv(csv_path, index_col=0, encoding='shift-jis')

    dt_values = []

    for col in df.columns:
        y_values = df[col].values

        initial_data = y_values[:3]
        best_initial = initial_data[np.argmax(initial_data)]

        search_data = y_values[3:]

        all_data = np.concatenate([[best_initial], search_data])

        dt_per_iteration = []
        for i in range(len(all_data)):
            best_so_far = all_data[:i+1].max()
            dt = y_target - best_so_far
            dt_per_iteration.append(dt)

        dt_values.append(dt_per_iteration)

    method_data_all[method_name] = dt_values
    print(f"  {method_name}: {len(dt_values)} samples")

print("\n" + "="*70)

max_iterations = max(max(len(v) for v in dt_list) for dt_list in method_data_all.values())

method_names = list(method_data_all.keys())
base_method = 'BO(SP-GPR-SC)'

all_results = []

print(f"\nBase method: {base_method}")
print(
    "Alternative hypothesis: DT_average of comparison method > "
    f"DT_average of {base_method} (alternative='greater')\n"
)
print(f"Iterations: 0 to {max_iterations - 1}\n")

for target_iteration in range(0, max_iterations):
    iteration_index = target_iteration

    method_data = {}
    for method_name in method_names:
        values_at_target = [v[iteration_index] for v in method_data_all[method_name]
                           if iteration_index < len(v)]
        method_data[method_name] = values_at_target

    base_data = method_data[base_method]
    for method_name in method_names:
        if method_name == base_method:
            continue

        compare_data = method_data[method_name]

        if len(base_data) != len(compare_data):
            min_len = min(len(base_data), len(compare_data))
            base_data_paired = base_data[:min_len]
            compare_data_paired = compare_data[:min_len]
        else:
            base_data_paired = base_data
            compare_data_paired = compare_data

        diff = np.array(compare_data_paired) - np.array(base_data_paired)

        if np.all(diff == 0):
            stat, p = np.nan, 1.0
        else:
            try:
                stat, p = wilcoxon(compare_data_paired, base_data_paired, alternative='greater')
            except ValueError:
                stat, p = np.nan, 1.0

        all_results.append({
            "Iteration": target_iteration,
            "Comparison method": method_name,
            "Base method": base_method,
            "Sample size": len(compare_data_paired),
            "Statistic": stat,
            "p-value": p,
            "Significant at alpha=0.05": "Yes" if p < 0.05 else "No",
        })

    if target_iteration % 5 == 0 or target_iteration == max_iterations:
        print(f"Completed up to iteration {target_iteration}...")

results_df = pd.DataFrame(all_results)

output_csv_path = result_dir / 'wilcoxon_test_all_iterations.csv'
results_df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
print(f"\nSaved results: {output_csv_path}")

print("\n" + "="*70)
print("Summary of test results (first 10 rows):")
print(results_df.head(10).to_string(index=False))
print("\n...")
print("\nSummary of test results (last 10 rows):")
print(results_df.tail(10).to_string(index=False))
print("="*70)
