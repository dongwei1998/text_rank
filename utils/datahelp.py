# coding=utf-8
# =============================================
# @Time      : 2022-08-08 14:32
# @Author    : DongWei1998
# @FileName  : datahelp.py
# @Software  : PyCharm
# =============================================

import jieba



def read_stopwords(args):
    '''
    读取停用词
    :return: 返回停用词列表
    '''
    jieba.load_userdict(args.custom_dictionary)
    stop_word = []  # 停用词列表，后续进行添加
    with open(args.stop_words_path,'r',encoding="utf-8") as f:
        for line in f:
            # 判断读取的是否为空
            if line:
                stop_word.append(line.strip())  # 去除俩边空格，然后添加到列表
    return stop_word


# 读取原始数据
def get_sentences(args):
    '''
    读取原始数据，并将数据处理成word2vec模型需要的格式
    :return:
    '''
    # 判断是否需要去停用词
    global stop_words
    if args.use_stopwords:
        stop_words = read_stopwords(args)

    # 定义盛饭数据的列表
    sentences = []

    # 判断是否训练词向量
    if args.use_words_vector:
        with open(args.text_path,'r',encoding="utf-8") as f:
            for line in f:
                # 判断是否为空
                if line:
                    # 去除文章前后空格
                    content = line.strip()
                    # 进行分词操作
                    content= jieba.cut(content)
                    content = [x for x in content]
                    # 判断是否去停用词
                    if args.use_stopwords:
                        for word in content:
                            if word in stop_words:
                                content.remove(word)
                    # 如果最终内容不为空，就加入到sentences
                    if content:
                        sentences.append(content)
    else:
        with open(args.text_path,'r',encoding="utf-8") as f:
            for line in f:
                # 判断读取是否为空
                if line:
                    # 去除文章前后空白
                    content =  line.strip()
                    content = [x for x in content]
                    # 判断content 是否为空
                    if content:
                        sentences.append(content)
    return sentences
