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

# df = pd.read_csv('hpc_results/results/global_test_metrics.csv', usecols=['model_name','time','epochs','fixed_train_size','batch_size','iou','avg_loss','precision','recall','f1_score'])

# # df = df.drop('training_time')
# df = convert_column_dtypes(df)
# print(df_group_one(df))

# df = df[(df['epochs'] == 50) & (df['fixed_train_size'] == 5000)]
# print(df[df['epochs'] == 50])

# bar_chart_grouped_by_model_name(df)

# add_dice()


import pandas as pd
import matplotlib.pyplot as plt

# Creating a DataFrame from the provided CSV data
data = """
model_name,time,epochs,parameter_count,fixed_train_size,fixed_valid_size,fixed_test_size,batch_size,training_time,iou,avg_loss,accuracy,precision,recall,f1_score
unet_baseline,10:16,50,1928417,1000,1000,1000,32,10:16,0.5909,0.7429,0.0251,0.9972,0.6604,0.8489
unet_baseline,10:00,50,1928417,1000,1000,1000,32,10:00,0.5799,0.7341,0.0265,0.9972,0.6717,0.8092
unet_baseline,10:41,50,1928417,1000,1000,1000,32,10:41,0.4153,0.5869,0.031,0.9938,0.4275,0.9357
unet_baseline,10:42,50,1928417,1000,1000,1000,32,10:42,0.5857,0.7387,0.0295,0.9973,0.6885,0.7968
unet_baseline,10:40,50,1928417,1000,1000,1000,32,10:40,0.6358,0.7774,0.0266,0.9977,0.7177,0.8478
unet_baseline,10:36,50,1928417,1000,1000,1000,32,10:36,0.6341,0.7761,0.0195,0.9977,0.7123,0.8525
unet_baseline,10:27,50,1928417,1000,1000,1000,32,10:27,0.5907,0.7427,0.0222,0.9972,0.653,0.861
unet_baseline,10:17,50,1928417,1000,1000,1000,32,10:17,0.5381,0.6997,0.0234,0.9962,0.5542,0.9487
unet_baseline,10:23,50,1928417,1000,1000,1000,32,10:23,0.6486,0.0243,0.9979,0.7544,0.8222,0.7868
unet_baseline,10:25,50,1928417,1000,1000,1000,32,10:25,0.5833,0.0561,0.9976,0.7501,0.7239,0.7368
unet_baseline,10:24,50,1928417,1000,1000,1000,32,10:24,0.5387,0.0383,0.9966,0.6311,0.7861,0.7002
unet_baseline,10:12,50,1928417,1000,1000,1000,32,10:12,0.5418,0.0471,0.9969,0.6792,0.7283,0.7029
unet_baseline,10:12,50,1928417,1000,1000,1000,32,10:12,0.5302,0.0426,0.9966,0.6343,0.7635,0.6929
unet_baseline,10:15,50,1928417,1000,1000,1000,32,10:15,0.5132,0.0433,0.9964,0.609,0.7654,0.6783
unet_baseline,10:08,50,1928417,1000,1000,1000,32,10:08,0.4661,0.0529,0.9958,0.5597,0.7359,0.6358
unet_baseline,10:08,50,1928417,1000,1000,1000,32,10:08,0.4754,0.0461,0.9959,0.571,0.7396,0.6445
unet_baseline,10:04,50,1928417,1000,1000,1000,32,10:04,0.5205,0.0384,0.9964,0.6079,0.7836,0.6846
unet_baseline,10:10,50,1928417,1000,1000,1000,32,10:10,0.51,0.048,0.9965,0.6265,0.7329,0.6755
unet_baseline,10:43,50,1928417,1000,1000,1000,32,10:43,0.5277,0.0514,0.9968,0.6753,0.707,0.6908
unet_baseline,10:43,50,1928417,1000,1000,1000,32,10:43,0.5347,0.0388,0.9967,0.6485,0.753,0.6968
unet_baseline,10:48,50,1928417,1000,1000,1000,32,10:48,0.5497,0.0397,0.9967,0.6403,0.7952,0.7094
unet_baseline,11:11,50,1928417,1000,1000,1000,32,11:11,0.4663,0.0435,0.9956,0.5389,0.7757,0.636
unet_baseline,10:06,50,1928417,1000,1000,1000,32,10:06,0.5119,0.0419,0.9963,0.602,0.7737,0.6772
unet_baseline,10:03,50,1928417,1000,1000,1000,32,10:03,0.5196,0.0419,0.9964,0.6148,0.7703,0.6838
unet_baseline,10:07,50,1928417,1000,1000,1000,32,10:07,0.5318,0.044,0.9967,0.6443,0.7528,0.6943
"""



# Using StringIO to simulate reading from a file
from io import StringIO
df = pd.read_csv(StringIO(data))

mean_iou = df['iou'].mean()
median_iou = df['iou'].median()
variance_iou = df['iou'].var()
std_dev_iou = df['iou'].std()
min_iou = df['iou'].min()
max_iou = df['iou'].max()

# Print the results
print(f"Mean IOU: {mean_iou:.4f}")
print(f"Median IOU: {median_iou:.4f}")
print(f"Variance of IOU: {variance_iou:.4f}")
print(f"Standard Deviation of IOU: {std_dev_iou:.4f}")
print(f"Minimum IOU: {min_iou:.4f}")
print(f"Maximum IOU: {max_iou:.4f}")

# Plotting the IoU values
plt.figure(figsize=(10, 5))
plt.plot(df['iou'], marker='o', linestyle='-')
plt.title('IoU variacija - dataset: 1000, batch_size: 32, epochs: 50')
plt.xlabel('Bandymo numeris')
plt.ylabel('IoU')
plt.grid(True)
plt.xticks(range(len(df)), [f"{i+1}" for i in range(len(df))])  # Adding custom x-axis labels for clarity
plt.show()

