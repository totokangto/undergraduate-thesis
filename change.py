import os

def rename_folders(directory):
    for folder_name in os.listdir(directory):
        old_path = os.path.join(directory, folder_name)
        if os.path.isdir(old_path):
            if folder_name[6:].isdigit():
                number_part = '00' + folder_name[6:] 
                new_folder_name = f"frame{number_part}"
                new_path = os.path.join(directory, new_folder_name)
                os.rename(old_path, new_path)
                print(f'Renamed: {folder_name} -> {new_folder_name}')

target_directory = "/data/khlee01/repos/3DGStream/dataset/flame_steak"
rename_folders(target_directory)

