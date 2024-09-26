# %%
import re

from tqdm import tqdm


import os
import sys
import pickle
import torch

from transformers import  BertModel, BertTokenizer
from utils import CAPACITY, BLOCK_SIZE, DEFAULT_MODEL_NAME
Max = 511



root_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_dir)
from buffer import Buffer, Block

with open('./EURInverted/EUR57K_text_set.pickle', 'rb') as file:
    data = pickle.load(file)
    data_train = data['train']
    data_dev = data['dev']
    data_test = data['test']
with open('./EURInverted/EUR57K_label_set.pickle', 'rb') as file:
    label = pickle.load(file)
    label_train = label['train']
    label_dev = label['dev']
    label_test = label['test']


model = BertModel.from_pretrained('../pretrain/bert_base_uncase/')
tokenizer = BertTokenizer.from_pretrained('../pretrain/bert_base_uncase/')

def calculate_semantic_similarity(text1, text2):
    # 使用tokenizer将文本转换为tokens，并添加特殊标记
    inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
    inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)
    # print(inputs1.data['input_ids'].shape[1])
    # print(inputs2.data['input_ids'].shape[1])

    # 使用BERT模型对文本进行编码
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)

    # 获取文本编码表示
    embeddings1 = outputs1.last_hidden_state[:, 0, :]  # 获取第一个token的表示，即CLS表示
    embeddings2 = outputs2.last_hidden_state[:, 0, :]

    # embeddings1 = outputs1.last_hidden_state.mean(dim=1) # 使用平均池化获取句子表示
    # embeddings2 = outputs2.last_hidden_state.mean(dim=1)

    # 计算文本之间的语义相似度
    similarity = torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1)

    return similarity.item()  # 返回相似度值

def get_text_before_position1(u, text, max_length=20):
    # 从切分点u向前搜索单词，找到文本片段的起始位置
    start_index = max(0, u - max_length)
    # while start_index > 0 and text[start_index] != ' ':
    #     start_index -= 1
    # 获取文本片段
    text_before_u = text[start_index:u]
    return text_before_u

def get_text_after_position2(u, text, max_length=20):
    # 从切分点u向后搜索单词，找到文本片段的结束位置
    end_index = min(len(text), u + max_length)
    # while end_index < len(text) and text[end_index] != ' ':
    #     end_index += 1
    # 获取文本片段
    text_after_u = text[u:end_index]
    return text_after_u

def get_text(u, d, nodes):
    text_before_token = get_text_before_position1(u[0]+1, d, 20)
    text_after_token = get_text_after_position2(u[0]+1, d, 20)

    text_before = tokenizer.convert_tokens_to_string(text_before_token).strip().replace(" ,",",").replace(" .",".").replace(" .",".").replace(" !","!").replace(" ?","?")
    text_after = tokenizer.convert_tokens_to_string(text_after_token).replace(" ,",",").replace(" .",".").replace(" .",".").replace(" !","!").replace(" ?","?")
    return text_before, text_after

def sematic_cost(text, poses):
    new = [list(tup) for tup in poses]
    for i in range(len(poses)-1):
        if i==0:
            continue
        text_before, text_after = get_text(poses[i], text, poses)
        sim = calculate_semantic_similarity(text_before, text_after)
        new[i][1] += sim
    new = [tuple(lst) for lst in new]
    return new

def split_document_into_blocks(d):
    end_tokens = {'\n': 0, '.': 1, '?': 1, '!': 1, ',': 2}
    aa = d[-1]
    # for k, v in list(end_tokens.items()):
    #     end_tokens['Ġ' + k] = v
    break_cost = 8
    poses = [(i, end_tokens[tok]) for i, tok in enumerate(d) if tok in end_tokens]
    poses.insert(0, (-1, 0))
    if poses[-1][0] < len(d) - 1:
        poses.append((len(d) - 1, 0))
    x = 0
    while x < len(poses) - 1:
        if poses[x + 1][0] - poses[x][0] > Max:
            poses.insert(x + 1, (poses[x][0] + Max, break_cost))
        x += 1
    # sematic similarity
    new_cost = sematic_cost(d, poses)
    poses = new_cost
    return poses

def min_cost_split(poses):
    n = len(poses)
    dp = [float('inf')] * n
    split = [-1] * n

    dp[0] = 0
    split[0] = -1

    for i in range(1, n):
        for j in range(i):
            if poses[i][0] - poses[j][0] <= Max:
                cost = dp[j] + poses[i][1]
                if cost < dp[i]:
                    dp[i] = cost
                    split[i] = j

    # Reconstruct the solution
    segments = []
    i = n - 1
    while i > 0:
        segments.append(poses[i][0])
        i = split[i]
    segments.reverse()

    return segments, dp[n - 1]



def clean(data):
    tmp_doc = []
    for words in data.split():
        if ':' in words or '@' in words or len(words) > 60:
            pass
        else:
            c = re.sub(r'[>|-]', '', words)
            # c = words.replace('>', '').replace('-', '')
            if len(c) > 0:
                tmp_doc.append(c)
    tmp_doc = ' '.join(tmp_doc)
    tmp_doc = re.sub(r'\([A-Za-z \.]*[A-Z][A-Za-z \.]*\) ', '', tmp_doc)
    return tmp_doc

def process(dataset, label_set, dataset_name):

    tokenizer = BertTokenizer.from_pretrained('../pretrain/bert_base_uncase/')
    set_txt = []
    for i in tqdm(range(len(dataset))):
        d, l = clean(dataset[i]), label_set[i]
        d = tokenizer.tokenize(d)
        poses = split_document_into_blocks(d)
        segments, min_cost = min_cost_split(poses)

        text = ''
        start = -1
        for e in segments:
            sentence = " ".join(d[start+1 : e+1])

            s = sentence+ str(tokenizer.sep_token)
            text += s
            start = e
        set_txt.append(text)
    return set_txt

trainT = process(data_train, label_train, 'train')
testT = process(data_test, label_test, 'test')
devT = process(data_dev, label_dev, 'dev')
eurInver = {}
eurInver['train'] = trainT
eurInver['test'] = testT
eurInver['dev'] = devT

# with open('../data/EURInverted/EURInver511_text_set_BERT_sim.pickle', 'wb') as f:
#     pickle.dump(eurInver, f)