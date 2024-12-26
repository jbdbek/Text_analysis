import re
import pandas as pd
import os
import csv
from utils.configs import txt_to_csv_path

#读取文件清洗数据，转换为csv文件
def data_processing(input_file, output_file):
    data=[]
    review_id = None
    review_content = []    
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()                                                #去除行首尾空白字符
            match = re.match(r'<review id="(\d+)">', line)                     #检查是否是评论ID行（开始标签）
            if match:
                if review_id and review_content:                               #如果当前有未保存的评论，保存当前评论
                    data.append({'ID': review_id, 'Content': ''.join(review_content).strip()})
                review_id = match.group(1)                                     #提取新的评论ID并开始新评论
                review_content = []                                            #重置评论内容

            elif line.startswith('</review>'):
                if review_id and review_content:                               #评论结束，保存当前评论
                    data.append({'ID': review_id, 'Content': ''.join(review_content).strip()})
                review_id = None
                review_content = []
            else:
                if review_id:                                                  #累积评论内容
                    review_content.append(line)
    if review_id and review_content:                                           #保存最后一个评论（防止文件结束时遗漏）
        data.append({'ID': review_id, 'Content': ''.join(review_content).strip()})
    
    df = pd.DataFrame(data)                                                    #将其他类型的数据结构转换为Pandas DataFrame
    df.to_csv(output_file, index=False, encoding='utf-8')                      #将其保存为 CSV 文件

def labeldata_processing(input_file, output_file):
    data = []
    review_id = None
    review_content = []
    review_label = None
    
    try:                                                                       #尝试以utf-8编码打开文件，如果失败则使用其他编码
        with open(input_file, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()                                            #去除行首尾空白字符
                match = re.match(r'<review id="(\d+)"\s+label="(\d)">', line)  # 检查是否是评论ID行（开始标签）
                if match:
                    if review_id and review_content:                           #如果当前有未保存的评论，保存当前评论
                        data.append({'ID': review_id, 'Content': ''.join(review_content).strip(), 'Label': review_label})
                    review_id = match.group(1)                                 #提取新的评论ID并开始新评论
                    review_label = int(match.group(2))
                    review_content = []                                        #重置评论内容
                elif line.startswith('</review>'):
                    if review_id and review_content:                           #评论结束，保存当前评论
                        data.append({'ID': review_id, 'Content': ''.join(review_content).strip(), 'Label': review_label})
                    review_id = None
                    review_content = []
                    review_label = None
                else:
                    if review_id:                                              #累积评论内容
                        review_content.append(line)
        if review_id and review_content:                                       #保存最后一个评论（防止文件结束时遗漏）
            data.append({'ID': review_id, 'Content': ''.join(review_content).strip(), 'Label': review_label})
    
    except UnicodeDecodeError:                                                #如果文件编码是UTF-8失败，尝试使用其他编码
        print(f"Error decoding file with utf-8, trying ISO-8859-1 encoding...")
        with open(input_file, 'r', encoding='ISO-8859-1') as file:
            for line in file:
                line = line.strip()
                match = re.match(r'<review id="(\d+)"\s+label="(\d)">', line)
                if match:
                    if review_id and review_content:
                        data.append({'ID': review_id, 'Content': ''.join(review_content).strip(), 'Label': review_label})
                    review_id = match.group(1)
                    review_label = int(match.group(2))
                    review_content = []
                elif line.startswith('</review>'):
                    if review_id and review_content:
                        data.append({'ID': review_id, 'Content': ''.join(review_content).strip(), 'Label': review_label})
                    review_id = None
                    review_content = []
                    review_label = None
                else:
                    if review_id:
                        review_content.append(line)
        if review_id and review_content:
            data.append({'ID': review_id, 'Content': ''.join(review_content).strip(), 'Label': review_label})
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding='utf-8')                     #将数据保存为 CSV 文件
    print(f"Data has been saved to {output_file}")

#增加标签0(消极)/1(积极)，得到训练集数据
def merge_datasets(positive_file, negative_file, output_file):
    with open(positive_file, 'r', encoding='utf-8') as pos_file, \
         open(negative_file, 'r', encoding='utf-8') as neg_file, \
         open(output_file, 'w', encoding='utf-8', newline='') as out_file: 
        writer = csv.writer(out_file)
        writer.writerow(['ID', 'Content', 'Label'])                           #写入表头
        
        pos_reader = csv.reader(pos_file)                                     #处理积极评论，标签为1
        next(pos_reader)                                                      #跳过表头
        for row in pos_reader:
            writer.writerow([row[0], row[1], 1])                              #添加标签为 1 
        
        neg_reader = csv.reader(neg_file)                                     #处理消极评论，标签为0
        next(neg_reader)
        for row in neg_reader:
            writer.writerow([row[0], row[1], 0])                              #添加标签为 0      
    print(f"数据集已合并到 {output_file}")

def get_data():
    input_file_1 = './data/train_data/cn_sample_data/sample.negative.txt'
    input_file_2 = './data/train_data/cn_sample_data/sample.positive.txt'
    input_file_3 = './data/train_data/en_sample_data/sample.negative.txt'
    input_file_4 = './data/train_data/en_sample_data/sample.positive.txt'
    input_file_5 = './data/test_data/test.cn.txt'
    input_file_6 = './data/test_data/test.en.txt'
    input_file_7 = './data/test_label_data/test.label.cn.txt'
    input_file_8 = './data/test_label_data/test.label.en.txt'
    output_file_1 = txt_to_csv_path(input_file_1)
    output_file_2 = txt_to_csv_path(input_file_2)
    output_file_3 = txt_to_csv_path(input_file_3)
    output_file_4 = txt_to_csv_path(input_file_4)
    output_file_5 = txt_to_csv_path(input_file_5)
    output_file_6 = txt_to_csv_path(input_file_6)
    output_file_7 = txt_to_csv_path(input_file_7)
    output_file_8 = txt_to_csv_path(input_file_8)
    output_file_9 = './out/train_data/cn_sample_data/sample.csv'
    output_file_10 = './out/train_data/en_sample_data/sample.csv'
    data_processing(input_file_1, output_file_1)
    data_processing(input_file_2, output_file_2)
    data_processing(input_file_3, output_file_3)
    data_processing(input_file_4, output_file_4)
    data_processing(input_file_5, output_file_5)
    data_processing(input_file_6, output_file_6)
    labeldata_processing(input_file_7, output_file_7)
    labeldata_processing(input_file_8, output_file_8)
    merge_datasets(output_file_2, output_file_1, output_file_9)
    merge_datasets(output_file_4, output_file_3, output_file_10)
    print(f'Data has been saved to {output_file_1}')
    print(f'Data has been saved to {output_file_2}')
    print(f'Data has been saved to {output_file_3}')
    print(f'Data has been saved to {output_file_4}')
    print(f'Data has been saved to {output_file_5}')
    print(f'Data has been saved to {output_file_6}')
    print(f'Data has been saved to {output_file_7}')
    print(f'Data has been saved to {output_file_8}')