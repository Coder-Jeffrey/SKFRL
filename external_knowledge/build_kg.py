import requests
import pickle
import re
import jieba
from tqdm import tqdm
import csv
import string

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

# 定义 ConceptNet API 的基本 URL
conceptnet_api_url = "http://api.conceptnet.io"

# 定义一个函数来获取单词的一阶邻居子图
def get_word_neighbors(word):
    url = f"{conceptnet_api_url}/c/en/{word}?offset=0&limit=30&filter=/c/en/start/{word}"
    response = requests.get(url)
    data = response.json()
    edges = data['edges']
    # 仅保留一阶邻居关系
    neighbors = []
    for edge in edges:
        start = edge['start']['label']
        end = edge['end']['label']
        rel = edge['rel']['label']
        if start.lower() == word:
            neighbors.append((start, end, rel))

    return neighbors

def get_word_neighbors_off(word, conceptnet_data):
    neighbors = []
    for row in conceptnet_data:
        if f'/c/en/{word}/' in row[0] and word == row[2].split('/')[3]:
            start = row[2].split('/')[3]
            end = row[3].split('/')[3]
            relation = row[1].split('/')[2]
            # relation_info = eval(row[3])
            neighbors.append((start, relation, end))
            # print('++++++++++++++++++++')
            # print(row[0])
            # print((start, relation, end))
        if len(neighbors) == 10:
            break
    return neighbors

# 读取 EUR 数据
with open('./EUR57K/EUR57K_text_set.pickle', 'rb') as file:
    data = pickle.load(file)
    data_train = data['train']
    data_dev = data['dev']
    data_test = data['test']

# 读取停用词
def read_stopwords():
    with open('./EUR57K/baidu_stopwords.txt', 'r', encoding='utf-8') as file:
         lines = file.readlines()
         stop_list = [k.strip() for k in lines]
    return stop_list

# 读取离线 ConceptNet
conceptnet_data = []
with open('../ConceptNet/EN_assertions.csv', 'r', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for row in csv_reader:
        conceptnet_data.append(row)

all_words = []
for i in tqdm(range(len(data_train))):
    d = clean(data_train[i]).lower()
    dd = [k for k in jieba.cut(d, cut_all=False) if k not in read_stopwords() and k.strip() != '' and k.strip() not in string.punctuation]
    for j in range(len(dd)):
        w = dd[j].lower()
        all_words.append(w)

for i in tqdm(range(len(data_test))):
    d = clean(data_test[i]).lower()
    dd = [k for k in jieba.cut(d, cut_all=False) if k not in read_stopwords() and k.strip() != '' and k.strip() not in string.punctuation]
    for j in range(len(dd)):
        w = dd[j].lower()
        all_words.append(w)

for i in tqdm(range(len(data_dev))):
    d = clean(data_dev[i]).lower()
    dd = [k for k in jieba.cut(d, cut_all=False) if k not in read_stopwords() and k.strip() != '' and k.strip() not in string.punctuation]
    for j in range(len(dd)):
        w = dd[j].lower()
        all_words.append(w)


all_words = list(set(all_words))

all_ere = {}
for i in tqdm(range(len(all_words))):
    w = all_words[i].lower()
    if w in all_ere:
        continue
    neighbors = get_word_neighbors_off(w, conceptnet_data)
    if neighbors == []:
        continue
    all_ere[w] = neighbors

with open('EUR_all_triple.pickle', 'wb') as f:
    pickle.dump(all_ere, f)

print(len(all_ere))
print(len(all_words))


# for i in tqdm(range(len(data_test))):
#     d = clean(data_test[i])
#     dd = [k for k in jieba.cut(d, cut_all=False) if k not in read_stopwords() and k.strip() != '']
#     # cleaned_dd = [item for item in dd if item.strip() != ' ']
#     for j in tqdm(range(len(dd))):
#         w = dd[j].lower()
#         if w in all_ere:
#             continue
#         neighbors = get_word_neighbors_off(w, conceptnet_data)
#         if neighbors==[]:
#             continue
#         all_ere[w] = neighbors
#
# for i in tqdm(range(len(data_dev))):
#     d = clean(data_dev[i])
#     dd = [k for k in jieba.cut(d, cut_all=False) if k not in read_stopwords() and k.strip() != '']
#     # cleaned_dd = [item for item in dd if item.strip() != ' ']
#     for j in tqdm(range(len(dd))):
#         w = dd[j].lower()
#         if w in all_ere:
#             continue
#         neighbors = get_word_neighbors_off(w, conceptnet_data)
#         if neighbors==[]:
#             continue
#         all_ere[w] = neighbors
#
# with open('EUR_all_triple.pickle', 'wb') as f:
#     pickle.dump(all_ere, f)



