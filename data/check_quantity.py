import pandas as pd
import glob

data = pd.read_csv("full_data_1_200.csv")

def compute_quanlity(folder_path):
    # Đọc tất cả các tệp CSV trong thư mục
    all_files = glob.glob(folder_path + "*.csv")

    # Khởi tạo một danh sách rỗng để lưu trữ các DataFrame từ các tệp CSV
    dfs = []

    # Đọc từng tệp CSV và thêm vào danh sách dfs
    for file in all_files:
        df = pd.read_csv(file)
        dfs.append(df)

    # Ghép các DataFrame trong danh sách dfs thành một DataFrame duy nhất
    combined_df = pd.concat(dfs,ignore_index=True)
    print(combined_df.shape)
    if folder_path == "labels/":
        combined_df = combined_df[~combined_df['name'].str.contains('0110')]

    return combined_df.shape[0]

successful_items = compute_quanlity("labels/")
error_items = compute_quanlity("errors/")

print(f"Number of fitems {data.shape[0]}")
print(f"Number of successful items {successful_items}")
print(f"Number of error items {error_items}")

if error_items + successful_items == data.shape[0]:
    print("Enough data")
else:
    print("Missing data")