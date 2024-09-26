import random
import numpy as np
import pickle
import torch
import math
# from test import GraphSAGE
from tqdm import tqdm
import os
import re
import string

torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.Tensor(in_features, out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        device = self.weight.device
        input = input.to(device)

        support = torch.matmul(input, self.weight)
        device = support.device
        adj = adj.to(device)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


def load_data():
    entity2id = {}
    relation2id = {}
    with open('entity2id.txt', 'r', encoding='utf-8') as f1, open('relation2id.txt', 'r', encoding='utf-8') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            entity2id[line[0]] = line[1]

        for line in lines2:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            relation2id[line[0]] = line[1]

    with open('EUR_all_triple.pickle', 'rb') as f:
        all_triple = pickle.load(f)

    # print(all_triple['self_confidence'])
    # if 'demolition' in all_triple:
    #     print('111')
    # 构建每个entity的邻居
    ent_nei = {}
    for key, value in entity2id.items():
        if key in all_triple:
            nei = all_triple[key]
            temp = []
            for i in range(len(nei)):
                temp.append(entity2id[nei[i][2]])
            ent_nei[entity2id[key]] = temp

    entity_embed = {}
    with open('eur_entity_embeddings.txt', 'r', encoding='utf-8') as f1:
        lines1 = f1.readlines()
        for line in lines1:
            line = line.strip().split('\t')
            if len(line) != 2:
                continue
            entity_embed[line[0]] = line[1]

    all_sage_embed = []
    for key, value in entity_embed.items():
        embed_values = [float(value)
                        for value in entity_embed[key].split(', ')]
        embed_values = np.float32(embed_values)
        all_sage_embed.append(embed_values)
    return entity2id, relation2id, ent_nei, entity_embed, all_sage_embed


def get_knowledge_impru(vocab, PPMI):
    entity2id, relation2id, ent_nei, entity_embed, all_sage_embed = load_data()
    id2entity = {value: key for key, value in entity2id.items()}

    # print()
    wnbor = {}
    for w in vocab:
        wid = entity2id.get(w, None)
        wnbor[w] = {'c_emb': [], 'n_emb': []}
        if wid is None:
            continue
        entity_embed_values = [float(value)
                               for value in entity_embed[wid].split(', ')]
        wnbor[w]['c_emb'] = torch.tensor(
            np.float32(entity_embed_values))

        nbors = ent_nei.get(wid, [])
        if nbors:
            for n in nbors:
                if id2entity[n] in vocab and PPMI[vocab.index(w), vocab.index(id2entity[n])] > 0:
                    print(PPMI[vocab.index(w), vocab.index(id2entity[n])])

                    entity_embed_values = [float(value)
                                           for value in entity_embed[wid].split(', ')]
                    wnbor[w]['n_emb'].append(torch.tensor(
                        np.float32(entity_embed_values)))
    print(wnbor.keys())
    for k, v in wnbor.items():
        print(f'{k}: c emb: ', type(v['c_emb']), 'n emb: ', len(v['n_emb']))
    return wnbor
    # print(w, ent_nei[w])

    # for key, value in tqdm(ent_nei.items()):
    #     entity_embed_values = [float(value)
    #                            for value in entity_embed[key].split(', ')]
    #     center_node_embedding_tensor = torch.tensor(
    #         np.float32(entity_embed_values))

    #     neighbor_node_embedding = []

    #     if len(value) <= 3:
    #         print(value)
    #         for v in value:
    #             neighbor_node_embedding.append(entity_embed[v])
    #     else:
    #         random_integers = random.sample(range(len(value)), 3)
    #         for v in random_integers:
    #             x = value[v]
    #             neighbor_node_embedding.append(entity_embed[value[v]])

    #     neighbor_embeddings = []
    #     for embedding_str in neighbor_node_embedding:
    #         embedding_values = [float(value)
    #                             for value in embedding_str.split(', ')]
    #         neighbor_embeddings.append(embedding_values)
    #     # 转换为float32类型的张量
    #     neighbor_embeddings_tensor = torch.tensor(
    #         neighbor_embeddings, dtype=torch.float32)

    # model = GraphSAGE(input_dim=768)
    # updated_embedding = model(
    #     center_node_embedding_tensor, neighbor_embeddings_tensor)
    # aggregated_embedding = torch.mean(updated_embedding, dim=0)
    # aggregated_embedding = aggregated_embedding.detach().numpy()
    # all_sage_embed[int(key)] = aggregated_embedding
    # with open('../../data/EUR/eur_entity_embeddings_sage.txt', 'w', encoding='utf-8') as file:
    #     line = f"{key}\t{aggregated_embedding}\n"
    #     file.write(line)

    # # 存储嵌入向量为.npy文件
    # os.makedirs('data/booksummaries', exist_ok=True)
    # file_path = 'data/booksummaries/books_entity_embeddings_sage.npy'
    # # np.save(file_path, all_sage_embed)

    # loaded_embeddings = np.load(file_path)

    # # 打印结果
    # print(loaded_embeddings.shape)


def get_knowledge(vocab, PPMI):
    entity2id, relation2id, ent_nei, entity_embed, all_sage_embed = load_data()

    id2entity = {int(value): key for key, value in entity2id.items()}
    t = 0
    for c, neibor in tqdm(ent_nei.items()):
        nodes = []
        nodes.append(c)
        nodes.extend(neibor)
        A = np.zeros((len(nodes), len(nodes)))
        mapid = {int(n): i for i, n in enumerate(nodes)}
        c = int(c)
        if id2entity[c] in vocab:
            i = vocab[id2entity[c]]
            for n in neibor:
                n = int(n)
                if id2entity[n] in vocab:
                    print('!!!')
                    j = vocab[id2entity[n]]
                    print(PPMI[i, j])
                    A[mapid[c], mapid[n]] = PPMI[i, j]
                    A[mapid[n], mapid[c]] = PPMI[i, j]
        else:
            continue

        A = torch.tensor(A, dtype=torch.float32)
        is_first_row_zero = torch.all(A[0] == 0)
        if(is_first_row_zero):
            continue

        node_features = []

        for id in nodes:
            entity_embed_values = [float(value)
                                   for value in entity_embed[id].split(', ')]
            # node_embedding_tensor = torch.tensor(
            #     np.float32(entity_embed_values))
            node_features.append(entity_embed_values)

        # 初始化GCN模型
        # 768，768
        node_features = torch.tensor(node_features, dtype=torch.float32)
        # print('node_features: ', node_features.shape)
        # print('A: ', A.shape, A)

        # A = torch.tensor(A, dtype=torch.float32)
        # 对邻接矩阵归一化
        deg = A.sum(dim=1)  # 计算每个节点的度
        sqrt_deg_inv = torch.pow(deg, -0.5)
        sqrt_deg_inv[sqrt_deg_inv == float('inf')] = 0
        D_inv_sqrt = torch.diag(sqrt_deg_inv)  # 构建度矩阵的逆平方根矩阵
        adj_normalized = torch.mm(torch.mm(D_inv_sqrt, A), D_inv_sqrt)

        gcn_model = GraphConvolution(768, 768)
        # 训练GCN模型
        node_features = node_features.unsqueeze(0)
        adj_normalized = adj_normalized.unsqueeze(0)


        output = gcn_model(node_features, adj_normalized)
        output = output.squeeze(0)

        # print(nodes)
        all_sage_embed[int(c)] = output[0].detach().numpy()
        t += 1
        if t % 50 == 0:
            np.save('out/EUR_entity_embeddings_gcn.npy', all_sage_embed)
    return

def normalize_adj(adj):
    row_sum = torch.tensor(adj.sum(1))
    row_sum[row_sum == 0] = 1e-12  # 避免零除问题
    d_inv_sqrt = torch.pow(row_sum, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
    adj_normalized = torch.bmm(
        torch.bmm(adj, d_mat_inv_sqrt).transpose(1, 2), d_mat_inv_sqrt)
    return adj_normalized

def get_PPMI(words, vocab, window_size=20):
    # 构建词汇共现矩阵（默认简化处理，只考虑共现距离为1的情况）
    vocab_size = len(vocab)
    co_occurrence_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    print('begin to build co_occurrence_matrix')
    for i, word in tqdm(enumerate(words)):
        word_index = vocab[word]
        for j in range(max(0, i - window_size), min(len(words), i + window_size + 1)):
            if i != j:
                context_word_index = vocab[words[j]]
                co_occurrence_matrix[word_index, context_word_index] += 1

    # 计算共现概率和PPMI
    word_prob = co_occurrence_matrix / np.sum(co_occurrence_matrix)
    word_prob[word_prob == 0] = 1e-12  # 避免除以0
    PPMI_matrix = np.log(
        word_prob / np.outer(word_prob.sum(axis=1), word_prob.sum(axis=0)))
    PPMI_matrix = torch.FloatTensor(PPMI_matrix).unsqueeze(0)
    PPMI_matrix = torch.clamp(PPMI_matrix, min=0)
    PPMI_matrix = PPMI_matrix.squeeze(0).numpy()

    np.save('out/PPMI_matrix.npy', PPMI_matrix)
    return PPMI_matrix


def load_copus():
    with open('EUR511_text_set_BERT_sim.pickle', 'rb') as f:
        all_triple = pickle.load(f)
    return all_triple


def clean(data):
    import spacy
    nlp = spacy.load('en_core_web_sm')

    tmp_doc = []
    # 将字符串小写
    data = data.lower()
    for words in data.split():
        if ':' in words or '@' in words or len(words) > 60:
            pass
        else:
            c = re.sub(r'[.,?"()>|-]', '', words)
            # c = words.replace('>', '').replace('-', '')
            if len(c) > 0:
                tmp_doc.append(c)
    tmp_doc = ' '.join(tmp_doc)
    nlp_doc = nlp(tmp_doc)
    # 用户也可以在此进行自定义过滤字符
    # r1 = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    # # 者中规则也过滤不完全
    # r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
    # # \\\可以过滤掉反向单杠和双杠，/可以过滤掉正向单杠和双杠，第一个中括号里放的是英文符号，第二个中括号里放的是中文符号，第二个中括号前不能少|，否则过滤不完全
    # r3 = "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+"
    # # 去掉括号和括号内的所有内容
    # r4 = "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"
    # res = [r1, r2, r3, r4]
    # for r in res:
    #     tmp_doc = re.sub(r, '', tmp_doc)
    # 过滤非英文字符
    # tmp_doc = re.sub(r'[^\x00-\x7F]+', '', tmp_doc)
    # tmp_doc = re.sub(r'\([A-Za-z \.]*[A-Z][A-Za-z \.]*\) ', '', tmp_doc)
    # tmp_doc = re.sub(r1, '', tmp_doc)

    tmp_doc = ' '.join(
        [token.lemma_ for token in nlp_doc if token.is_alpha and token.is_ascii])
    return tmp_doc


def preprocess(text, stopwords):
    words = []
    for t in tqdm(text):
        # ct = clean(t)

        d = clean(t)

        dd = [k for k in d.split() if k not in stopwords and k.strip(
        ) != '' and k.strip() not in string.punctuation+'—’']
        t = [word.replace("[sep", "") for word in dd]
        # t = [word for word in t if word.isascii()]
        words.extend(t)

    return words


if __name__ == '__main__':
    # 输出文件夹
    os.makedirs('out', exist_ok=True)
    # 设置语料库大小设为原来的0.1倍
    copus_size = 1
    # 加载语料库
    copus = load_copus()
    # 加载停用词
    stopwords = set(line.strip() for line in open('baidu_stopwords.txt', encoding='utf-8'))
    # 判断预处理后的文本文件是否存在，如果不存在则生成
    if not os.path.exists('filtered_words.txt'):
        filtered_words = preprocess(copus['train'], stopwords)
        with open('filtered_words.txt', 'w', encoding='utf-8') as f:
            for w in filtered_words:
                f.write(w + '\n')
    # 加载预处理后的文本文件
    filtered_words = list(line.strip() for line in open(
        'filtered_words.txt', encoding='utf-8'))

    if copus_size < 1:
        copus_len = len(filtered_words)
        filtered_words = filtered_words[:int(copus_len*copus_size)]

    vocab = {v: k for k, v in enumerate(set(filtered_words))}
    # 生成PPMI矩阵
    PPMI = get_PPMI(filtered_words, vocab)
    # 生成知识图谱融合的词向量
    impru_worcs = get_knowledge(vocab, PPMI)
