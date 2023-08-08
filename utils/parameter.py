# coding=utf-8
# =============================================
# @Time      : 2022-07-20 17:08
# @Author    : DongWei1998
# @FileName  : parameter.py
# @Software  : PyCharm
# =============================================
import os
from easydict import EasyDict
from dotenv import load_dotenv,find_dotenv
import logging.config
import shutil



# 创建路径
def check_directory(path, create=True):
    flag = os.path.exists(path)
    if not flag:
        if create:
            os.makedirs(path)
            flag = True
    return flag


def parser_opt(model):
    load_dotenv(find_dotenv())  # 将.env文件中的变量加载到环境变量中
    args = EasyDict()
    logging.config.fileConfig(os.environ.get("logging_ini"))
    args.logger = logging.getLogger('model_log')
    # 清除模型以及可视化文件
    if model == 'train':
        args.use_stopwords = bool(os.environ.get('use_stopwords'))
        args.custom_dictionary = os.environ.get('custom_dictionary')
        args.stop_words_path = os.environ.get('stop_words_path')
        args.use_words_vector = bool(os.environ.get('use_words_vector'))
        args.text_path = os.path.join(os.environ.get('data_files'),'train_conver.txt')
        args.model_path = os.path.join(os.environ.get('output_dir'),'word2vec.model')
        args.num_epochs = int(os.environ.get('num_epochs'))
        args.locFirs_weight = float(os.environ.get('locFirs_weight'))
        args.locLast_weight = float(os.environ.get('locLast_weight'))
        args.first_num = int(os.environ.get('first_num'))
        args.size = int(os.environ.get('size'))
        args.keyword_type = int(os.environ.get('keyword_type'))
        args.summary_len = int(os.environ.get('summary_len'))
        args.minLen_weight = float(os.environ.get('minLen_weight'))
        # 生成文档向量时，断句用的符号，这个根据给定文章的的符号格式进行调整，下面时中英文版的标点符号
        args.break_points = [",",".","!","?",";","，","。","！","？","；"]
        args.use_MMR = bool(os.environ.get('use_MMR'))
        args.last_num = int(os.environ.get('last_num'))
        args.alpha = float(os.environ.get('alpha'))
        check_directory(args.model_path)
    elif model =='env':
        pass
    elif model == 'server':
        pass
    else:
        raise print('请给定model参数，可选【traian env test】')
    return args


if __name__ == '__main__':
    args = parser_opt('train')
