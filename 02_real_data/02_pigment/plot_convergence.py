#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from pathlib import Path

result_dir = Path('result')

methods = {
    'BO(SP-GPR-SC)': result_dir / 'BO(SP_GPR_SC)' / 'dE_to_target.csv',
    'BO(SP-GPR)': result_dir / 'BO(SP_GPR)' / 'dE_to_target.csv',
    'BO(GPR)': result_dir / 'BO(GPR)' / 'dE_to_target.csv',
    'MLR': result_dir / 'MLR' / 'dE_to_target.csv'
}

results = {}

for method_name, file_path in methods.items():
    print(f"Processing {method_name}...")

    df = pd.read_csv(file_path, index_col=0)

    dE_values = []

    for col in df.columns:
        col_data = df[col].values
        dE_values.append(col_data)

    max_iterations = max(len(v) for v in dE_values)
    dE_average = []

    for i in range(max_iterations):
        values_at_i = [v[i] for v in dE_values if i < len(v)]
        dE_average.append(np.mean(values_at_i))

    results[method_name] = dE_average
    print(f"  {method_name}: {len(dE_average)} iterations")

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

for method_name, dE_values in results.items():
    iterations = list(range(len(dE_values)))
    color = colors.get(method_name, 'black')
    plt.plot(iterations, dE_values, label=method_name, linewidth=3.5, color=color)

plt.xlabel('Number of Iterations', fontsize=30)
plt.ylabel('DT$_{average}$', fontsize=30)
plt.xlim(0, 20)
plt.ylim(bottom=0)
plt.xticks(np.arange(0, 21, 5), fontsize=24)
y_max = plt.gca().get_ylim()[1]
plt.yticks(np.arange(0, int(y_max) + 2, 1), fontsize=24)
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
plt.grid(True, alpha=0.6)
plt.legend(fontsize=28)
plt.tight_layout()

output_plot_path = result_dir / 'convergence_plot.png'
plt.savefig(output_plot_path, dpi=300)
print(f"Plot saved to: {output_plot_path}")
plt.show()

print("\nDone!")
