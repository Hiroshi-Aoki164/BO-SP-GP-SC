#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from pathlib import Path

result_dir = Path('result')

methods = {
    'BO(SP-GPR-SC)': result_dir / 'BO(SP_GPR_SC)',
    'BO(SP-GPR)': result_dir / 'BO(SP_GPR)',
    'BO(GPR)': result_dir / 'BO(GPR)',
    'MLR': result_dir / 'MLR'
}


results = {}

for method_name, method_dir in methods.items():
    print(f"Processing {method_name}...")

    target_df = pd.read_csv(method_dir / 'y_target.csv', index_col=0)
    initial_df = pd.read_csv(method_dir / 'y_initial.csv', index_col=0)
    experiment_df = pd.read_csv(method_dir / 'y_experiment.csv', index_col=0)

    dt_values = []

    for col in target_df.columns:
        target = target_df[col].values[0]

        initial_data = initial_df[col].values

        best_initial = initial_data[np.argmin(np.abs(initial_data - target))]
        dt_initial = abs(target - best_initial)

        dt_per_iteration = [dt_initial]

        experiment_data = experiment_df[col].values

        current_best = best_initial
        for exp_value in experiment_data:
            if abs(target - exp_value) < abs(target - current_best):
                current_best = exp_value
            dt = abs(target - current_best)
            dt_per_iteration.append(dt)

        dt_values.append(dt_per_iteration)

    max_iterations = max(len(v) for v in dt_values)
    dt_average = []

    for i in range(max_iterations):
        values_at_i = [v[i] for v in dt_values if i < len(v)]
        dt_average.append(np.mean(values_at_i))

    results[method_name] = dt_average
    print(f"  {method_name}: {len(dt_average)} iterations")

output_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
output_csv_path = result_dir / 'convergence_analysis.csv'
output_df.to_csv(output_csv_path, index_label='Iteration')
print(f"\nCSV saved to: {output_csv_path}")

plt.figure(figsize=(8, 7))

colors = {
    'BO(SP-GPR-SC)': 'red',
    'BO(SP-GPR)': 'yellow',
    'BO(GPR)': 'blue',
    'MLR': 'green'
}

for method_name, dt_values in results.items():
    iterations = list(range(len(dt_values)))
    color = colors.get(method_name, 'black')
    plt.plot(iterations, dt_values, label=method_name, linewidth=3.5, color=color)

plt.xlabel('Number of Iterations', fontsize=30)
plt.ylabel('DT$_{average}$', fontsize=30)
plt.xlim(0, 20)
plt.ylim(bottom=0)
plt.xticks(np.arange(0, 21, 5), fontsize=24)
y_max = plt.gca().get_ylim()[1]
plt.yticks(np.arange(0, int(y_max) + 50, 50), fontsize=24)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
plt.grid(True, alpha=0.6)
plt.legend(fontsize=28)
plt.tight_layout()

output_plot_path = result_dir / 'convergence_plot.png'
plt.savefig(output_plot_path, dpi=300)
print(f"Plot saved to: {output_plot_path}")
plt.show()

print("\nCompleted!")
