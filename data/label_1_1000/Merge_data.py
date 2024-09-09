import pandas as pd

def read_and_transform(file_path):
    df = pd.read_csv(file_path)
    rows_list = []

    for index, row in df.iterrows():
        if 'center' in df.columns:
            rows_list.append({'name': row['center'], 'label': row['label']})
        if 'left' in df.columns:
            rows_list.append({'name': row['left'], 'label': row['label']})
        if 'right' in df.columns:
            rows_list.append({'name': row['right'], 'label': row['label']})

    transformed_df = pd.DataFrame(rows_list)
    return transformed_df

def merge_and_process_three_view(files):
    merged_df = pd.DataFrame(columns=['name', 'label'])
    
    for file in files:
        transformed_df = read_and_transform(file)
        merged_df = pd.concat([merged_df, transformed_df], ignore_index=True)
    
    return merged_df

def merge_center(files):
    merged_df = pd.DataFrame(columns=['name', 'label'])
    
    for file in files:
        df = pd.read_csv(file)
        # Giữ nguyên cột 'name' và 'label'
        center_df = df[['name', 'label']]
        merged_df = pd.concat([merged_df, center_df], ignore_index=True)
    
    return merged_df

def check_duplicates_and_sort(df):
    # Loại bỏ các dòng trùng lặp dựa trên cột name
    df.drop_duplicates(subset=['name'], keep='first', inplace=True)
    
    # Sắp xếp dữ liệu theo thứ tự bảng chữ cái của cột name và label
    df.sort_values(by=['name', 'label'], inplace=True)
    
    return df

# Danh sách các file CSV
files_center = ['train_1_1000_center_ord1.csv', 'val_1_1000_center_ord1.csv', 'test_1_1000_center_ord1.csv']
files_three_view = ['train_1_1000_three_view_ord1.csv', 'val_1_1000_three_view_ord1.csv', 'test_1_1000_three_view_ord1.csv']

# Merge các file center theo cách đơn giản
df_center = merge_center(files_center)

# Merge và xử lý các file three_view
df_three_view = merge_and_process_three_view(files_three_view)

# In ra kích thước và 5 dòng đầu của df_center
print("Size of df_center:", df_center.shape)
print("First 5 rows of df_center:")
print(df_center.head())

# In ra kích thước và 5 dòng đầu của df_three_view
print("Size of df_three_view:", df_three_view.shape)
print("First 5 rows of df_three_view:")
print(df_three_view.head())

# Merge cả hai DataFrame lại với nhau
df_combined = pd.concat([df_center, df_three_view], ignore_index=True)

# Kiểm tra trùng lặp và sắp xếp
df_final = check_duplicates_and_sort(df_combined)

# Lưu lại file CSV kết quả
df_final.to_csv('output.csv', index=False)
