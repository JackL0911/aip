import os

def merge_text_files(input_folder, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in os.listdir(input_folder):
            if filename.endswith('.txt'):
                file_path = os.path.join(input_folder, filename)
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read().strip()  # 读取内容并去掉前后空白
                    outfile.write(content + '\n')     # 写入到输出文件，后面加换行符

# 示例使用
input_folder = 'aclImdb_v1/aclImdb/train/all'   # 替换为你的文件夹路径
output_file = 'data/reviews_output.txt'  # 输出文件名

merge_text_files(input_folder, output_file)