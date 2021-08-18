#!/usr/local/bin/python3
# -*-coding:utf-8 -*-
"""
@Date  : 2021/8/18 下午1:57

@Author : zhutingting

@Desc : ==============================================

Blowing in the wind. ===

# ======================================================

@Project : Chinese-Punctuation-Restoration-with-Bert-CNN-RNN

@FileName: test_1_to_1_iwslt.py

@Software: PyCharm
测试数据,评估指标
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from tqdm import tqdm
from transformers import BertTokenizer

from data_1_to_1 import load_file, preprocess_data, create_data_loader
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
SEGMENT_SIZE = 200
BATCH_SIZE = 200


def predictions(data_loader, bert_punc):
    y_pred = []
    y_true = []
    for inputs, labels in tqdm(data_loader, total=len(data_loader)):
        with torch.no_grad():
            inputs, labels = inputs.cuda(), labels.cuda()
            output = bert_punc(inputs)
            y_pred += list(output.argmax(dim=1).cpu().data.numpy().flatten())
            y_true += list(labels.cpu().data.numpy().flatten())
    return y_pred, y_true


def evaluation(y_pred, y_test):
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[1, 2, 3])
    overall = metrics.precision_recall_fscore_support(
        y_test, y_pred, average='macro', labels=[1, 2, 3])
    result = pd.DataFrame(
        np.array([precision, recall, f1]),
        columns=list(['O', 'COMMA', 'PERIOD', 'QUESTION'])[1:],
        index=['Precision', 'Recall', 'F1']
    )
    result['OVERALL'] = overall[:3]
    return result


def test_model():
    # 加载模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(MODEL_PATH, 'hyperparameters.json'), 'r') as f:
        hyperparameters = json.load(f)
    output_size = len(PUNCTUATION_ENC)
    dropout = hyperparameters['dropout']

    bert_punc = BertChineseEmbSlimCNNlstmBert(SEGMENT_SIZE, output_size, dropout, None).to(device)
    bert_punc.load_state_dict(torch.load(os.path.join(MODEL_PATH, 'model')))
    bert_punc.eval()

    tokenizer = BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)

    # 加载数据
    data_test = load_file(TEST_PATH)
    X_test, y_test = preprocess_data(data_test, tokenizer, PUNCTUATION_ENC, SEGMENT_SIZE)
    data_loader_test = create_data_loader(X_test, y_test, False, BATCH_SIZE)

    # 测试并评估指标
    y_pred_test, y_true_test = predictions(data_loader_test, bert_punc)
    eval_test = evaluation(y_pred_test, y_true_test)
    print(eval_test)


if __name__ == "__main__":
    test_model()
