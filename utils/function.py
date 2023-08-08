# coding=utf-8
# =============================================
# @Time      : 2022-08-10 10:09
# @Author    : DongWei1998
# @FileName  : function.py
# @Software  : PyCharm
# =============================================
import os
import jieba
from jieba import analyse
import numpy as np

# 定义句子长度的权重
def len_weight_(args,sentence):
    '''
    计算句子长度
    :param sentence: 求解句子长度
    :return: 句子长度权重值
    '''

    # 假设求解句子长度小于我们想要的句子长度
    if len(sentence) <= args.summary_len:
        if len(sentence) / args.summary_len > args.minLen_weight:
            return len(sentence) / args.summary_len
        else:
            return args.minLen_weight
    # 假设长度是大于我们想要的长度
    else:
        #注意此时如果句子长度大于我们想要摘要的句子长度两倍那他的权重就是负数基本不会被进取为摘要
        # return 1-(len(sentence) -GLobalParameters.summary_len)/GLobalparameters.summary_Len
        # 我们采用下面这个策略起码保证句子的长度权重值不会为负数 最低为0.5
        if 1 - (len(sentence) - args.summary_len) / args.summary_len > args.minLen_weight:
            return 1 - (len(sentence) - args.summary_len) / args.summary_len
        else:
            # 如果小于阈值  那么就发回最小的权重值
            return args.minLen_weight


# 根据相似度  关键字权重 以及  句子长度权重 先粗略计算一个摘要句子列表
def get_first_summaries(args,text,stopwords,word2vec_model):
    '''
    :param text: 文档
    :param stopwords:停用词
    :param model: 词向量模型
    :return:按要列表 按照权重从大到小排列[(句子,权重),(句子,权重)]
    '''
    # 获取（位置，句子）列表
    sentences = get_sentences(args,text)

    # 获取句子列表
    sen_lis = [x[1] for x in sentences]

    # 获取文档向量
    docvec = doc_vector(args,text,stopwords,word2vec_model)

    # 获取句子向量列表
    sen_vecs = []
    for i in range(len(sen_lis)):
        # 如果句子是首句
        if i == 0:
            sen_vecs.append(sentence_vector(args,sen_lis[i],stopwords,word2vec_model)*args.locFirs_weight)
        # 如果句子是尾句
        elif i == len(sen_lis)-1:
            sen_vecs.append(sentence_vector(args,sen_lis[i],stopwords,word2vec_model)*args.locLast_weight)
        # 如果是中间的句子
        else:
            sen_vecs.append(sentence_vector(args,sen_lis[i],stopwords,word2vec_model))

    # 计算余弦值列表
    cos_lis = [cos_dist(docvec,x) for x in sen_vecs]

    # 计算关键字权重列表
    # 获取关键字
    keywords = get_keywords(args,text)

    # 计算权重
    keyweights = [keyword_weight(x,keywords) for x in sen_lis]

    # 计算长度权重
    len_weight = [len_weight_(args,x) for x in sen_lis]

    # 根据余弦相似度  关键字 长度权重  计算每个句子最终权重
    final_weights = [cos*keyword*length for cos in cos_lis for keyword in keyweights for length in len_weight]

    # 形成最后的（句子，权重列表）
    final_lis = []
    for sen,weight in zip(sen_lis,final_weights):
        final_lis.append((sen,weight))

    # 将句子大小按照权重大小  从高到低排序
    final_lis = sorted(final_lis,key=lambda x:x[1],reverse=True)

    # 取出第一次摘要的句子个数
    final_lis = final_lis[:args.first_num]

    return final_lis




# 定义生成文档向量的方法
def doc_vector(args,text, stop_words, word2vec_model):
    '''
    计算文档向量，句子向量求平均
    :param text: 需要计算的文档
    :param stop_words:停用词表
    :param model:词向量模型
    :return:文档向量
    '''

    # 获取（位置，句子）列表
    sen_lis = get_sentences(args,text)
    # 提取出句子
    sen_lis = [x[1] for x in sen_lis]
    # 定义一个文档的初始化向量是根据训练词向量的维度来的
    vector = np.zeros(args.size, )
    # 计算文档里包含多少句子
    length = len(sen_lis)
    # 遍历所有句子
    for sentence in sen_lis:
        # 获取句子向量
        sen_vec = sentence_vector(args,sentence, stop_words, word2vec_model)
        # 计算文档向量
        vector += sen_vec


    return vector / length


# 定义生成句子向量的方法

def sentence_vector(args,sentence, stop_words, word2vec_model):
    '''
    根据词向量模型 和句子  生成句子向量
    :param sentence:
    :param stop_words: 停用词列表之所以传入形式是因为可以再程序启动后只需要在外部加载一次
    :param model:词向量模型,之所以通过传参数是为了项目上线时候可以再外部加载一次模型,不能在代码内部加载
    :return:返回句子的向量是np.arrray()格式
    '''

    # 初始化一个句子向量维度100 是根据word2vec模型的参数得来的
    vector = np.zeros(args.size, )

    # 判断用的是不是词向量
    if args.use_words_vector:
        # 当使用了词向量时候，在判断是否需要去停用词
        if args.use_stopwords:
            # 并将句子进行分词
            content = jieba.cut(sentence)
            content = [x for x in content]
            # 计算分词后的长度
            count = 0
            for word in content:
                # 假如该词不在使用词表里
                if word not in stop_words:
                    # 统计用了几个词语进行向量求和
                    count += 1
                    # 将查询后的向量加入进去
                    try:
                        vector += np.array(word2vec_model.wv.__getitem__(word))
                    except Exception as e:
                        # todo 存在oov问题，小数据集明显，随着训练语料的增加，问题将得到解决
                        pass

            # 注意此处一些句子会出现“不！”这样的话，去使用词后是空列表导致count=0导致没办法除所以需要判断count
            if count == 0:
                return np.zeros(args.size, )
            return vector / count
        # 如果不同停用词
        else:
            # 将句子进行分词
            content = jieba.cut(sentence)
            content = [x for x in content]
            # 计算长度
            length = len(content)
            # 遍历该列表
            for word in content:
                try:
                    vector += np.array(word2vec_model.wv.__getitem__(word))
                except Exception as e:
                    # todo 存在oov问题，小数据集明显，随着训练语料的增加，问题将得到解决
                    pass
            return vector / length
    # 如果使用字向量，将不用考虑停用词
    else:
        content = [x for x in sentence]
        # 计算长度
        length = len(content)
        # 遍历求解
        for word in content:
            try:
                vector += np.array(word2vec_model.wv.__getitem__(word))
            except Exception as e:
                # todo 存在oov问题，小数据集明显，随着训练语料的增加，问题将得到解决
                pass

        return vector / length

# 定义获取文档短句列表  和 位置信息的方法
def get_sentences(args,text):
    '''
    将文档切割成句子
    :param text: 需要进行断句的文档
    :return: 返回一个列表，包含句子的信息和位置信息[(1,句子),(2，句子2),()......]
    '''

    # 读取断句符号
    break_points = args.break_points

    # 先将text中的所有断句符号替换成"."
    for point in break_points:
        text = text.replace(point,".")
    # 根据"."进行断句的操作
    sen_lis = text.split(".")

    # 去掉断句后的空字符
    sen_lis = [x for x in sen_lis if x != ""]

    # 将位置信息和句子信息封装在一个列表里
    results = []
    for i in range(len(sen_lis)):
        if i != len(sen_lis)-1:
            results.append((i+1,sen_lis[i]))
        else:
            # 最后一句话的时候  位置-1
            results.append((-1,sen_lis[i]))

    return results

# 定义余弦函数
def cos_dist(vec1,vec2):
    '''
    :param vec1: 向量一
    :param vec2: 向两二
    :return: 返回俩个向量的余弦相似度
    '''

    # a = tf.reduce_sum(tf.multiply(embedding1, embedding2), axis=-1)  # [N,]
    # b = tf.sqrt(tf.reduce_sum(tf.square(embedding1), axis=-1) + 1e-10)  # [N,]
    # c = tf.sqrt(tf.reduce_sum(tf.square(embedding2), axis=-1) + 1e-10)  # [N,]
    # similarity = tf.identity(a / tf.multiply(b, c), 'similarity')  # [N,], 取值范围: (-1,1)
    dist1 = float(np.dot(vec1,vec2)/(np.linalg.norm(vec1+1e-10) * np.linalg.norm(vec2+1e-10)))
    return dist1

# rextRank求解关键字
def get_keywords(args,text):
    '''
    返回关键字列表
    :param text: 需要提取关键字的文档
    :return: 返回关键字列表
    '''

    # 假设是采用textRank算法
    if args.keyword_type == 0:
        textrank = analyse.textrank
        keyword = textrank(text)
        return keyword
    # 采用tfidf求解
    else:
        pass

# 定义求解句子含有关键字个数的权重值
def keyword_weight(sentence,keywords):
    '''
    获取一个句子在一篇文档里的关键字权重值
    :param sentence: 对应的句子
    :param keywords: 关键词列表
    :return: 一个float类型的数字
    '''
    # 计算关键字个数
    count = 0
    for keyword in keywords:
        count += sentence.count(keyword)
    # 如果一个句子中不包含关键字  那么权重就是0
    return count/len(keywords)



# 获取最终摘要
def get_last_summaries(args,text,final_lis,stopwords,model):
    '''
    获取最终的摘要列表
    :param text:
    :param final_lis:
    :param stopwords: 停用词
    :param model: 词向量模型
    :return: 摘要列表
    '''

    # 判断是否用MMR
    if args.use_MMR:
        results = MMR(args,final_lis,stopwords,model)
    else:
        results = final_lis[:args.last_num]
        # 注意此处的results 是以元组为元素的列表  需要将句子取出来
        results = [x[0] for x in results]

    # 为了使句子读起来连贯  我们按照摘要句子在原始文章的位子信息  进行排序
    sentences = get_sentences(args,text)   # [(1,句子1),(2,句子2),(),...]
    # print("句子是"，sentences)

    # 定义摘要列表[(句子，位置),(句子，位置)，...]
    summaries = []
    for summary in results:
        for sentence in sentences:
            if summary == sentence[1]:
                summaries.append((summary,sentence[0]))
    # print("summaries",summaries)

    # 根据位置排序
    summaries = sorted(summaries,key=lambda x:x[1])

    # 获取最终摘要句子  不要位置信息
    summaries = [x[0] for x in summaries]

    return summaries


# 定义MMR算法  保证摘要多样性
def MMR(args,final_lis,stopwords,word2vec_model):
    '''
    根据MMR算法   保证摘要句子多样性
    :param final_lis: 初步摘要（句子，权重）列表
    :param stopwords: 停用词
    :param model: 词向量模型
    :return: 最终摘要的句子列表
    '''

    # 根据final_lis  获取句子列表
    sen_lis = [x[0] for x in final_lis]

    # 权重列表
    weight_lis = [x[1] for x in final_lis]

    # 定义摘要列表
    summary_lis = []

    # 首先挑出权重最大的一句话，它必然是再要列表中的一句
    summary_lis.append(sen_lis[0])

    # 为方便处理 将最终摘要的句子 从预摘要列表里删除掉
    del sen_lis[0]
    del weight_lis[0]

    # 根据要求个数摘要句子
    # 如果只摘要一个句子 直接接结果返回就可以了
    if args.last_num == 1:
        return summary_lis
    # 再要不止一个句子 需要进行计算求解
    else:
        for i in range(len(sen_lis)):
            # 所有候选句子的向量列表
            vec_lis = [sentence_vector(args,x,stopwords,word2vec_model) for x in sen_lis]
            # 已经作为摘要的句子向量
            summary_vec = [sentence_vector(args,x,stopwords,word2vec_model) for x in summary_lis]
            # 定义各个句子的得分情况
            scores = []

            for vec1 in vec_lis:
                # 计数器
                count = 0
                # 初始化句子分数
                score = 0
                for vec2 in summary_vec:
                    score += args.alpha*weight_lis[count]-(1-args.alpha)*cos_dist(vec1,vec2)
                # 求新句子与最终摘要句子的平均相似度
                count += 1
                scores.append(score/len(summary_vec))

            # 根据最大分数的下标  求解对应的句子加入到摘要里面  通过array求解
            scores = np.array(scores)
            index = np.argmax(scores)

            # 将对应的句子加入到摘要列表
            summary_lis.append(sen_lis[index])

            # 将对应句子 和 权重 从预摘要列表里删除
            del sen_lis[index]
            del weight_lis[index]

        # 返回指定的需要的句子个数
        return summary_lis[:args.last_num]