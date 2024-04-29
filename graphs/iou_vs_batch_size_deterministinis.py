import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

y_label = 'Apmokymo laikas (sekundėmis)'
x_label = 'Partijos dydis'
# FONT SIZE PADIDINTI!

# Creating a DataFrame from the provided CSV data
data_1000 = """
model_name,time,epochs,parameter_count,fixed_train_size,fixed_valid_size,fixed_test_size,batch_size,training_time,iou,avg_loss,accuracy,precision,recall,f1_score
unet_baseline,12:09,50,1928417,1000,1000,1000,8,12:09,0.5253,0.0457,0.9966,0.629,0.7611,0.6888
unet_baseline,12:29,50,1928417,1000,1000,1000,16,12:29,0.5304,0.0388,0.9965,0.6156,0.793,0.6931
unet_baseline,11:45,50,1928417,1000,1000,1000,32,11:45,0.4663,0.0435,0.9956,0.5389,0.7757,0.636
unet_baseline,11:34,50,1928417,1000,1000,1000,64,11:34,0.4604,0.0399,0.9952,0.5097,0.8267,0.6306
unet_baseline,11:34,50,1928417,1000,1000,1000,128,11:34,0.4708,0.0464,0.9956,0.5443,0.7771,0.6402
"""

# Using StringIO to simulate reading from a file
df = pd.read_csv(StringIO(data_1000))

df['training_time_minutes'] = df['time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

ax1 = plt.gca()  # Get current axis

# Plot IoU on the primary y-axis
iou_line, = ax1.plot(df['batch_size'], df['iou'], marker='o', linestyle='-', color='b', label='IoU')
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('IoU', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_title('Apmokymo laikas vs Partijos dydis, kai imtis 1000, 50 epochų')

# Create a second y-axis for training time
ax2 = ax1.twinx()
time_line, = ax2.plot(df['batch_size'], df['training_time_minutes'], marker='o', linestyle='-', color='r', label='Apmokymo laikas (minutėmis)')
ax2.set_ylabel('Apmokymo laikas (sekundėmis)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Adding custom x-axis labels for clarity
ax1.set_xticks(df['batch_size'])  # Set x-ticks to batch size values
ax1.set_xticklabels(df['batch_size'])

# Adding a legend to show labels for both lines
lines = [iou_line, time_line]
ax1.legend(lines, [l.get_label() for l in lines])
plt.grid(True)

plt.show()

# # Plotting IoU values
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)  # First subplot for IoU
# plt.plot(df['iou'], marker='o', linestyle='-')
# plt.title('IoU vs Partijos dydis')
# plt.xlabel('Partijos dydis')
# plt.ylabel('IoU')
# plt.grid(True)
# plt.xticks(range(len(df)), labels=df['batch_size'].tolist())  # Custom x-axis labels from batch size

# # Plotting training times
# plt.subplot(1, 2, 2)  # Second subplot for Training Time
# plt.plot(df['training_time_minutes'], marker='o', linestyle='-', color='r')
# plt.title('Apmokymo laikas vs Partijos dydis')
# plt.xlabel('Partijos dydis')
# plt.ylabel('Apmokymo laikas (minutėmis)')
# plt.grid(True)
# plt.xticks(range(len(df)), labels=df['batch_size'].tolist())  # Custom x-axis labels from batch size

# plt.tight_layout()
# plt.show()


# ----------------------5000----------------------------


# Creating a DataFrame from the provided CSV data
data_5000 = """
model_name,time,epochs,parameter_count,fixed_train_size,fixed_valid_size,fixed_test_size,batch_size,training_time,iou,avg_loss,accuracy,precision,recall,f1_score
unet_baseline,20:05,50,1928417,5000,1000,1000,8,20:05,0.4252,0.0372,0.9945,0.471,0.8139,0.5967
unet_baseline,20:23,50,1928417,5000,1000,1000,16,20:23,0.5055,0.0483,0.9965,0.6273,0.7225,0.6715
unet_baseline,19:05,50,1928417,5000,1000,1000,32,19:05,0.5221,0.0467,0.9967,0.6496,0.7267,0.686
unet_baseline,19:02,50,1928417,5000,1000,1000,64,19:02,0.4736,0.0379,0.9958,0.5608,0.7528,0.6428
unet_baseline,18:34,50,1928417,5000,1000,1000,128,18:34,0.4768,0.0383,0.9957,0.5479,0.7861,0.6457
"""

df = pd.read_csv(StringIO(data_5000))


df['training_time_minutes'] = df['time'].apply(lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]))

ax1 = plt.gca()  # Get current axis

# Plot IoU on the primary y-axis
iou_line, = ax1.plot(df['batch_size'], df['iou'], marker='o', linestyle='-', color='b', label='IoU')
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('IoU', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_title('Apmokymo laikas vs Partijos dydis, kai imtis 5000, 50 epochų')

# Create a second y-axis for training time
ax2 = ax1.twinx()
time_line, = ax2.plot(df['batch_size'], df['training_time_minutes'], marker='o', linestyle='-', color='r', label='Apmokymo laikas (minutėmis)')
ax2.set_ylabel('Apmokymo laikas (minutėmis)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Adding custom x-axis labels for clarity
ax1.set_xticks(df['batch_size'])  # Set x-ticks to batch size values
ax1.set_xticklabels(df['batch_size'])

# Adding a legend to show labels for both lines
lines = [iou_line, time_line]
ax1.legend(lines, [l.get_label() for l in lines])
plt.grid(True)

plt.show()

# Plotting IoU values
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)  # First subplot for IoU
# plt.plot(df['iou'], marker='o', linestyle='-')
# plt.title('IoU vs Partijos dydis')
# plt.xlabel('Partijos dydis')
# plt.ylabel('IoU')
# plt.grid(True)
# plt.xticks(range(len(df)), labels=df['batch_size'].tolist())  # Custom x-axis labels from batch size

# # Plotting training times
# plt.subplot(1, 2, 2)  # Second subplot for Training Time
# plt.plot(df['training_time_minutes'], marker='o', linestyle='-', color='r')
# plt.title('Apmokymo laikas vs Partijos dydis')
# plt.xlabel('Partijos dydis')
# plt.ylabel('Apmokymo laikas (minutėmis)')
# plt.grid(True)
# plt.xticks(range(len(df)), labels=df['batch_size'].tolist())  # Custom x-axis labels from batch size

# plt.tight_layout()
# plt.show()


# mean_iou = df['iou'].mean()
# median_iou = df['iou'].median()
# variance_iou = df['iou'].var()
# std_dev_iou = df['iou'].std()
# min_iou = df['iou'].min()
# max_iou = df['iou'].max()

# # Print the results
# print(f"Mean IOU: {mean_iou:.4f}")
# print(f"Median IOU: {median_iou:.4f}")
# print(f"Variance of IOU: {variance_iou:.4f}")
# print(f"Standard Deviation of IOU: {std_dev_iou:.4f}")
# print(f"Minimum IOU: {min_iou:.4f}")
# print(f"Maximum IOU: {max_iou:.4f}")

# # Plotting the IoU values
# plt.figure(figsize=(6, 4))
# plt.plot(df['iou'], marker='o', linestyle='-')
# plt.title('IoU priklausomybė nuo partijos dydžio - 5000 nuotraukų, 50 epochų')
# plt.xlabel('Partijos dydis')
# plt.ylabel('IoU')
# plt.grid(True)
# plt.xticks(range(len(df)), labels=['8', '16', '32', '64', '128'])  # Adding custom x-axis labels for clarity
# plt.show()
