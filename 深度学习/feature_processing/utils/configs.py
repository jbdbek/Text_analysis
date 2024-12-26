import os
def txt_to_csv_path(input_file):
    # 获取输入文件的目录和文件名
    input_dir, input_filename = os.path.split(input_file)
    # 修改文件路径，输出到out文件夹
    output_dir = './out'  # 输出目录
    # 获取输入文件的相对路径（去掉根目录部分）
    relative_dir = os.path.relpath(input_dir, start='./data')  # 获取相对于 './data' 的路径
    # 拼接输出路径：保持输入的目录结构
    full_output_dir = os.path.join(output_dir, relative_dir)
    # 确保输出目录存在
    os.makedirs(full_output_dir, exist_ok=True)
    # 修改文件扩展名为.csv
    output_filename = os.path.splitext(input_filename)[0] + '.csv'  # 修改文件扩展名为.csv
    # 拼接成完整的输出路径
    output_file = os.path.join(full_output_dir, output_filename)
    return output_file

def csv_to_txt_path(input_file):
    input_dir, input_filename = os.path.split(input_file)
    output_filename = os.path.splitext(input_filename)[0] + '.txt'
    output_file = os.path.join(input_dir, output_filename)
    return output_file

def process_csv_out_path(input_file):
    # 获取输入文件的目录和文件名
    input_dir, input_filename = os.path.split(input_file)
    # 修改文件路径，输出到out文件夹
    output_dir = './out'  # 输出目录
    # 获取输入文件的相对路径（去掉根目录部分）
    relative_dir = os.path.relpath(input_dir, start='./data')  # 获取相对于 './data' 的路径
    print(f"Relative directory: {relative_dir}")  # 调试输出
    # 拼接输出路径：保持输入的目录结构
    full_output_dir = os.path.join(output_dir, relative_dir)
    print(f"Full output directory: {full_output_dir}")  # 调试输出
    # 确保输出目录存在
    os.makedirs(full_output_dir, exist_ok=True)
    # 修改文件扩展名为.csv
    output_filename = os.path.splitext(input_filename)[0] + '.output.csv'  # 修改文件扩展名为.csv
    # 拼接成完整的输出路径
    output_file = os.path.join(full_output_dir, output_filename)
    print(f"Output file path: {output_file}")  # 调试输出
    return output_file
    
def csv_to_npy(input_file):
    # 获取输入文件的目录和文件名
    input_dir, input_filename = os.path.split(input_file)
    # 修改文件扩展名并添加.c2s后缀
    output_filename = os.path.splitext(input_filename)[0] + '.npy'
    # 拼接输出文件的完整路径
    output_file = os.path.join(input_dir, output_filename)
    print(f"Output file path: {output_file}")  # 调试输出
    return output_file


