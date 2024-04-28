import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np

def add_dice():
    # Load the CSV file
    df = pd.read_csv('hpc_results/results/global_test_metrics_omitted_values.csv')

    # Calculate Dice from IoU
    df['dice'] = round((2 * df['iou']) / (1 + df['iou']), 3)

    # Save the updated DataFrame back to a new CSV file
    df.to_csv('hpc_results/results/global_test_metrics_omitted_values+dice.csv', index=False)

    print("Dice column added and file saved as 'updated_file.csv'.")

def prepare_table_for_word(df):
    # df = pd.read_csv('hpc_results/results/global_test_metrics.csv')
    df = pd.read_csv('hpc_results/results/global_test_metrics.csv', usecols=['model_name','time','epochs','fixed_train_size','batch_size','iou','avg_loss','precision','recall','f1_score'])
    # df = df[df['model_name'] == 'unet_baseline']
    columns_to_round = ['iou','avg_loss','precision','recall','f1_score']
    for col in columns_to_round:
        df[col] = df[col].round(3)

    df.reset_index(drop=True, inplace=True)
    df = df.sort_values(by='iou', ascending=False)
    df.to_csv('hpc_results/results/global_test_metrics_omitted_values.csv', index=False)
    print(df.to_string(index=False))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def df_group_one(df):
    df = df.query("fixed_train_size == 5000")
    df_epochs = df.groupby(['model_name', 'epochs' ]).agg({
        'iou': 'mean',  # Assuming 'iou' is a numeric column
        'avg_loss': 'mean',  # Assuming 'avg_loss' is a numeric column
        # 'model_name': 'first'  # For a string column, you might want to take the first one
    })
    return df_epochs


def convert_column_dtypes(df):
    for column in df.columns:
        if column != 'model_name' and column != 'time' and column != 'training_time':
            df[column] = pd.to_numeric(df[column], errors='coerce')
        print(column)
    
    return df
    

# Assuming df is loaded from a CSV
# df = pd.read_csv('path_to_your_file.csv', usecols=['model_name', 'time', 'epochs', 'fixed_train_size', 'batch_size', 'iou', 'avg_loss', 'precision', 'recall', 'f1_score'])
# bar_chart_grouped_by_model_name(df)

df = pd.read_csv('hpc_results/results/global_test_metrics.csv', usecols=['model_name','time','epochs','fixed_train_size','batch_size','iou','avg_loss','precision','recall','f1_score'])

# df = df.drop('training_time')
df = convert_column_dtypes(df)
print(df_group_one(df))

# df = df[(df['epochs'] == 50) & (df['fixed_train_size'] == 5000)]
# print(df[df['epochs'] == 50])

# bar_chart_grouped_by_model_name(df)

# add_dice()