#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from pathlib import Path

result_dir = Path('result')

methods = {
    'BO(SP-GPR-SC)': result_dir / 'BO(SP_GPR_SC)' / 'y_all.csv',
    'BO(SP-GPR)': result_dir / 'BO(SP_GPR)' / 'y_all.csv',
    'BO(GPR)': result_dir / 'BO(GPR)' / 'y_all.csv',
    'MLR': result_dir / 'MLR' / 'y_all.csv'
}

y_target = 308.129096

results = {}

for method_name, csv_path in methods.items():
    print(f"Processing {method_name}...")

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
plt.ylim(0, 200)
plt.xticks(np.arange(0, 21, 5), fontsize=24)
plt.yticks(np.arange(0, 201, 20), fontsize=24)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
plt.grid(True, alpha=0.6)
plt.tight_layout()

output_plot_path = result_dir / 'convergence_plot.png'
plt.savefig(output_plot_path, dpi=300)
print(f"Plot saved to: {output_plot_path}")
plt.show()

print("\nDone!")
