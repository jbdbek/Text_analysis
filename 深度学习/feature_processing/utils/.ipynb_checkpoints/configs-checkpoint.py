{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "10438563-cf98-46b9-8f23-706a7dace7bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "54b06dbf-47d2-4846-81ad-6358eaa73bfa",
   "metadata": {},
   "outputs": [],
   "source": [
    "def generate_output_path(input_file):\n",
    "    # 获取输入文件的目录和文件名\n",
    "    input_dir, input_filename = os.path.split(input_file)\n",
    "    \n",
    "    # 修改文件路径，输出到out文件夹\n",
    "    output_dir = './out'  # 输出目录\n",
    "\n",
    "    # 获取输入文件的相对路径（去掉根目录部分）\n",
    "    relative_dir = os.path.relpath(input_dir, start='./data')  # 获取相对于 './data' 的路径\n",
    "\n",
    "    # 拼接输出路径：保持输入的目录结构\n",
    "    full_output_dir = os.path.join(output_dir, relative_dir)\n",
    "\n",
    "    # 确保输出目录存在\n",
    "    os.makedirs(full_output_dir, exist_ok=True)\n",
    "\n",
    "    # 修改文件扩展名为.csv\n",
    "    output_filename = os.path.splitext(input_filename)[0] + '.csv'  # 修改文件扩展名为.csv\n",
    "\n",
    "    # 拼接成完整的输出路径\n",
    "    output_file = os.path.join(full_output_dir, output_filename)\n",
    "\n",
    "    return output_file\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5dda2e01-d742-46cc-8da1-df20f34aa065",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.20"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
