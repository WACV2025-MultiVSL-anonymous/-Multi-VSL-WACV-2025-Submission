import pandas as pd

# Đọc file CSV gốc
df = pd.read_csv('/mnt/disk4/handsign_project/son_data/Experiment/data/label_1_400/test_1_400_three_view_ord1.csv')

# Tạo file test_1_1000_right_ord1.csv với name là cột 'right'
right_df = df[['right', 'label']].copy()
right_df.columns = ['name', 'label']
right_df.to_csv('data/label_1_400/test_1_400_right_ord1.csv', index=False)

# Tạo file test_1_1000_left_ord1.csv với name là cột 'left'
left_df = df[['left', 'label']].copy()
left_df.columns = ['name', 'label']
left_df.to_csv('data/label_1_400/test_1_400_left_ord1.csv', index=False)

print("Files have been created successfully.")
