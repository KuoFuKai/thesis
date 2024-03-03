import os


def replace_dots_in_filenames(directory):
    # 遍历目录中的所有文件和文件夹
    for filename in os.listdir(directory):
        # 检查是否为文件
        if os.path.isfile(os.path.join(directory, filename)):
            # 分割文件名和扩展名
            name_part, extension_part = os.path.splitext(filename)
            # 替换文件名中的点为下划线
            new_name = name_part.replace('.', '_') + extension_part
            # 构建原始文件和新文件的完整路径
            original_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)
            # 重命名文件
            os.rename(original_path, new_path)
            print(f"Renamed '{filename}' to '{new_name}'")


# 指定需要遍历的目录
directory_path = 'C:/Users/kevin/PycharmProjects/Datasets/Tainan_Confucian_Temple/images'
replace_dots_in_filenames(directory_path)
