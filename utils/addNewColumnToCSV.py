import pandas as pd

input = 'hpc_results/results/global_test_metrics.csv'
output = 'hpc_results/results/global_test_metrics_new.csv'

# Assuming your CSV data is stored in a variable named `data.csv`
df = pd.read_csv(input)

# Calculate the DICE metric
df['dice_metric'] = 2 * (df['precision'] * df['recall']) / (df['precision'] + df['recall'])

df.to_csv(output)
# Check the updated DataFrame
print(df)