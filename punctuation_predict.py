#!/usr/local/bin/python3
# -*-coding:utf-8 -*-
"""
@Date  : 2021/8/18 下午2:51

@Author : zhutingting

@Desc : ==============================================

Blowing in the wind. ===

# ======================================================

@Project : Chinese-Punctuation-Restoration-with-Bert-CNN-RNN

@FileName: punctuation_predict.py

@Software: PyCharm
提供接口预测结果
"""
import json
import os

import numpy as np
import torch
from transformers import BertTokenizer

from model_1_to_1 import (
    BertChineseEmbSlimCNNlstmBert,
)

# 模型路径
BERT_PATH = os.path.join("models", 'bert-base-chinese')
MODEL_PATH = 'models/20210817_200323'
# 测试数据路径
TEST_PATH = 'data/zh_iwslt/test_valid'

PUNCTUATION_ENC = {
    'O': 0,
    '，': 1,
    '。': 2,
    '？': 3
}
PUNCTUATION_DEC = {
    0: 'O',
    1: '，',
    2: '。',
    3: '？'
}
SEGMENT_SIZE = 200
BATCH_SIZE = 200


class PunctuationPredictModel:
    def __init__(self, gpu_id=0):

        self.gpu_id = gpu_id
        if self.gpu_id < 0:
            self.device = torch.device('cpu')
            print("Use CPU")
        else:
            self.device = torch.device('cuda:{}'.format(gpu_id))
            print(f"Use GPU with CUDA:{gpu_id}")
        self.tokenizer = BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)

        with open(os.path.join(MODEL_PATH, 'hyperparameters.json'), 'r') as f:
            self.hyper_parameters = json.load(f)

        output_size = len(PUNCTUATION_ENC)
        dropout = self.hyper_parameters['dropout']

        self.bert_punc = BertChineseEmbSlimCNNlstmBert(SEGMENT_SIZE, output_size, dropout, None).to(self.device)
        self.bert_punc.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'model')))
        self.bert_punc.eval()

    def convert_text_into_model_input(self, text):
        text_str_with_space = " ".join(list(text))
        # text_str_with_space = text
        tokens = self.tokenizer.tokenize(text_str_with_space)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return input_ids, tokens

    def predict(self, text):
        input_ids, tokens = self.convert_text_into_model_input(text)

        with torch.no_grad():
            input_ids = np.array(input_ids)
            input_ids = torch.tensor(input_ids)
            # 扩充维度在batch_size上,否则模型无法使用
            input_ids = input_ids.unsqueeze(0)
            input_ids = input_ids.to(self.device)

            output = self.bert_punc(input_ids)
            y_pred = list(output.argmax(dim=1).cpu().data.numpy().flatten())

            assert len(y_pred) == len(tokens)

            text_with_punc = self.reconstruct_text_with_pred_punc(tokens, y_pred)
            return text_with_punc

    @staticmethod
    def reconstruct_text_with_pred_punc(tokens, y_pred):
        pred_punc_idxes = np.nonzero(y_pred)[0]
        print(f"句子长度{len(tokens)},标点位置{pred_punc_idxes}")
        num_pred_punc = len(pred_punc_idxes)

        final_text = []
        if num_pred_punc != 0:
            final_text.extend(tokens[:pred_punc_idxes[0] + 1])
            final_text.append(PUNCTUATION_DEC.get(y_pred[pred_punc_idxes[0]], ""))
            for index in range(1, num_pred_punc):
                start_of_sub_str = pred_punc_idxes[index - 1] + 1
                end_of_sub_str = pred_punc_idxes[index] + 1
                final_text.extend(tokens[start_of_sub_str:end_of_sub_str])
                pred_punc_id = y_pred[pred_punc_idxes[index]]
                final_text.append(PUNCTUATION_DEC.get(pred_punc_id, ""))
            final_text.extend(tokens[pred_punc_idxes[-1]+1:])
        text_with_punc = "".join(final_text)
        return text_with_punc


if __name__ == "__main__":
    # text = r"首先我要用最快的速度为大家演示一些新技术的基础研究成果正好是一年前微软收购了我们公司而我们为微软带来了这项技术它就是Seadragon"
    text = r"截至8月17日24时据31个省（自治区、直辖市）和新疆生产建设兵团报告现有确诊病例1887例（其中重症病例62例）累计治愈出院病例87977例 累计死亡病例4636例 累计报告确诊病例94500例 现有疑似病例1例累计追踪到密切接触者1153308人 尚在医学观察的密切接触者43156人"
    text_with_punc_gt = r"截至8月17日24时，据31个省（自治区、直辖市）和新疆生产建设兵团报告，现有确诊病例1887例（其中重症病例62例），累计治愈出院病例87977例，累计死亡病例4636例，累计报告确诊病例94500例，现有疑似病例1例。累计追踪到密切接触者1153308人，尚在医学观察的密切接触者43156人。"
    # text = r"境外输入现有确诊病例753例（其中重症病例11例）现有疑似病例1例累计确诊病例7970例累计治愈出院病例7217例无死亡病例"
    # text_with_punc_gt = r"境外输入现有确诊病例753例（其中重症病例11例），现有疑似病例1例。累计确诊病例7970例，累计治愈出院病例7217例，无死亡病例。"
    punctuation_predict_model = PunctuationPredictModel(gpu_id=0)
    text_with_punc = punctuation_predict_model.predict(text)
    print(f"无标点原始文本:\t{text}\n带标点原始文本:\t{text_with_punc_gt}\n标点恢复后文本:\t{text_with_punc}")
