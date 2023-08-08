# coding=utf-8
# =============================================
# @Time      : 2022-08-08 14:17
# @Author    : DongWei1998
# @FileName  : server.py
# @Software  : PyCharm
# =============================================
from utils import parameter,function
from flask import Flask, jsonify, request
from gensim.models.word2vec import Word2Vec
import os

def read_stop_word(args):
    stop_word = []  # 停用词列表，后续进行添加
    with  open(args.stop_words_path, 'r', encoding="utf-8") as f:
        for line in f:
            # 判断读取的是否为空
            if line:
                stop_word.append(line.strip())  # 去除俩边空格，然后添加到列表
    return stop_word


if __name__ == '__main__':
    # APP应用构建
    app = Flask(__name__)
    app.config['JSON_AS_ASCII'] = False
    args = parameter.parser_opt('train')
    # 加载模型
    word2vec_model = Word2Vec.load(args.model_path)
    # 加载停用词列表
    stopwords = read_stop_word(args)

    @app.route('/')
    @app.route('/index')
    def _index():
        return "你好，欢迎使用Flask Web API，抽取式文本摘要!!!"
    @app.route('/get_summary', methods=['POST'])
    def get_summary():
        try:
            infos = request.get_json()
            data_dict = {
                'text': ''
            }
            for k, v in infos.items():
                data_dict[k] = v
            text = data_dict['text']
            # 参数检查
            if text is None:
                return jsonify({
                    'code': 500,
                    'msg': '请给定参数text！！！'
                })
            # 直接调用预测的API
            # 获取初次摘要列表
            final_lis = function.get_first_summaries(args,text, stopwords, word2vec_model)
            # 获取最终摘要列表
            summaries = function.get_last_summaries(args,text, final_lis, stopwords, word2vec_model)
            # 将获得的摘要拼接
            summary = ",".join(summaries)
            return jsonify({
                'code': 200,
                'msg': '成功',
                "summary": summary
            })
        except Exception as e:
            print(e)
            return jsonify({
                'code': 502,
                'msg': '预测数据失败, 异常信息为:{}'.format(e)
            })

    # 启动
    app.run(host='0.0.0.0', port=5100)