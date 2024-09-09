import pandas as pd


too_long_video = pd.read_csv("too_long_video.csv",index_col=False)

too_long_video_id = too_long_video["id-2"].values.tolist()



data = pd.read_csv("labels.csv",index_col=False)

full_data = pd.read_csv("full_data_1_200.csv",index_col=False)

print(data.shape)
# remove too long video
filtered_df = data[~data['video_lb_id'].isin(too_long_video_id)]

# remove 0110
filtered_df = filtered_df[~filtered_df['name'].str.contains('0110')]
print(filtered_df.shape)
# remove duplicated video
duplicate_labels = filtered_df[filtered_df.duplicated(subset=['name'])]
duplicate_labels = duplicate_labels.merge(full_data[['id-2', 'id']], left_on='video_lb_id', right_on='id-2', how='left')
# Xóa cột 'col_name' từ DataFrame
duplicate_labels.drop(columns=['id-2'], inplace=True)
duplicate_labels['label'] += 1

unique_df = filtered_df.drop_duplicates(subset=['name'])

print(unique_df.shape)


if duplicate_labels.shape[0] > 0 :
    # remove all related videos
    unique_df = unique_df.loc[~unique_df['name'].str.contains('|'.join(duplicate_labels['name'].values.tolist()))]

print(unique_df.shape)



# save
unique_df.to_csv('labels.csv',index=False)


duplicate_labels.to_csv("duplicate_videos.csv",index=False)