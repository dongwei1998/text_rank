# coding=utf-8
# =============================================
# @Time      : 2022-08-08 14:16
# @Author    : DongWei1998
# @FileName  : train.py
# @Software  : PyCharm
# =============================================
import os

import jieba
from gensim.models.word2vec import Word2Vec
from utils import parameter
from utils import datahelp
import numpy as np

def train_model(args):
    # 加载数据
    args.logger.info('开始加载数据...')
    x_train = datahelp.get_sentences(args)
    # 模型训练
    if os.path.exists(args.model_path) and os.path.isfile(args.model_path):  # 如果已经有现成model，则load
        # load model
        args.logger.info(f'加载Word2Vec模型 --> {args.model_path}')
        model = Word2Vec.load(args.model_path)
    else:
        args.logger.info(f'创建Word2Vec模型 --> {args.model_path}')
        model = Word2Vec(x_train)

    # 参数跟新
    updata = model.corpus_count
    args.logger.info(f'开始模型训练，总批次：{args.num_epochs}')
    model.train(x_train, total_examples=updata, epochs=args.num_epochs)  # 完成增量训练
    # 模型保存
    args.logger.info(f'模型保存--> {args.model_path}')
    model.save(args.model_path)  # 保存模型


def test_model(args):
    # 模型测试
    word2vec_model = Word2Vec.load(args.model_path)
    test_text = [word for word in jieba.cut('原公诉机关甘肃省榆中县人民检察院')]
    text_voc = get_word_vecter(word2vec_model, test_text)
    print(text_voc)

# 获取 word 向量
def get_word_vecter(word2vec_model,test_text):
    voc_list = []
    for word in test_text:
        try:
            vector_dm = word2vec_model.wv.__getitem__(word)
            voc_list.append(vector_dm)
        except Exception as e:
            # todo 存在oov问题，小数据集明显，随着训练语料的增加，问题将得到解决
            pass
    word_voc = voc_connect(voc_list)
    return word_voc


def voc_connect(word_voc):
    c = np.zeros(len(word_voc[0]), dtype=np.float32)
    n = 0
    for v in word_voc:
        c += v
        n += 1
    word_voc = list(c / n)
    return np.array(word_voc)

if __name__ == '__main__':
    args = parameter.parser_opt(model='train')
    train_model(args)

    # test_model(args)







