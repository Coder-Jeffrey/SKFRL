import torch
from transformers import BertModel
import torch.nn.functional as F


class BERTClass(torch.nn.Module):
    def __init__(self, dropout_rate, num_labels, label_embeddings, label_adj, kg_embedding, kg_dict):
        super(BERTClass, self).__init__()
        self.bert = BertModel.from_pretrained('./pretrain/bert_base_uncase')
        # self.roberta = RobertaModel.from_pretrained('../roberta-base')
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(768*2, num_labels)

        self.dim = 768
        self.label_embeddings = label_embeddings.cuda()
        self.label_adj = label_adj.cuda()
        self.kg_embedding = kg_embedding
        self.GCN3 = GraphConvolution(self.dim, self.dim)
        self.relu1 = torch.nn.LeakyReLU(0.2)
        self.GCN4 = GraphConvolution(self.dim,self.dim)
        self.kg_dict = kg_dict

    def forward(self, ids, mask, token_type_ids):
        predictions = []
        for i in range(len(ids)):
            # 找到值为 分隔符（RoBerta为2，BERT为102） 的位置索引
            indices = torch.where(ids[i] == 102)[0]
            # 初始化分割后的结果列表
            segments = []
            # 设置起始索引
            start_idx = 0
            # 遍历每个值为 分隔符（RoBerta为2，BERT为102） 的位置索引
            for idx in indices:
                # 截取从起始索引到当前索引处的子序列
                segment = ids[i][start_idx:idx]
                # 将子序列添加到结果列表中
                segments.append(segment)
                # 更新起始索引为当前索引的下一个位置
                start_idx = idx + 1
            # 添加最后一个子序列（从最后一个值为 2 的位置索引到列表末尾）
            segments.append(ids[i][start_idx:])

            # 设置最大长度
            max_len = 511
            # 截断或补齐每个 Tensor
            for i, segment in enumerate(segments):
                if i == 0:
                    segment = segment[1:]
                # 如果长度超过最大长度，则进行截断
                if len(segment) >= max_len:
                    segments[i] = segment[:max_len]
                # 如果长度不足最大长度，则进行补齐
                elif len(segment) < max_len:
                    # 创建需要填充的值的 Tensor，并设置为 PAD（RoBerta为1，BERT为0）
                    padding_tensor = torch.zeros(max_len - len(segment), dtype=torch.int64).cuda()
                    # 使用 torch.cat() 函数将填充的值添加到原始 Tensor 的末尾
                    segments[i] = torch.cat((segment, padding_tensor))
                # 在开头加入[CLS]（RoBerta为0，BERT为101）
                segments[i] = torch.cat((torch.tensor([101]).cuda(), segments[i]))
                # 在末尾加入值为 2
                # segments[i] = torch.cat((segments[i], torch.tensor([2]).cuda()))
            stacked_tensor = torch.stack(segments[:-1])
            mask = self.generate_attention_mask(stacked_tensor)
            token_type_ids = self.generate_token_type_ids(stacked_tensor)

            # 外部知识
            default_mapping_value = 118240
            kg_ids_tensor = torch.zeros_like(stacked_tensor)
            for batch_idx in range(stacked_tensor.size(0)):
                for seq_idx in range(stacked_tensor.size(1)):
                    bert_id = stacked_tensor[batch_idx, seq_idx]
                    kg_id = self.kg_dict.get(bert_id.item(), default_mapping_value)
                    kg_ids_tensor[batch_idx, seq_idx] = kg_id
            kg_ids_tensor = kg_ids_tensor.cpu()
            kg = self.kg_embedding[kg_ids_tensor]
            kg = torch.tensor(kg)
            kg = kg.cuda()

            # token_emb, bert_output = self.roberta(input_ids=stacked_tensor, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
            token_emb, bert_output = self.bert(input_ids=stacked_tensor, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
            drop_output = self.dropout(token_emb)
            drop_output = torch.add(drop_output, kg)
            
            # 注意力机制
            label_embeddings = self.label_embeddings.unsqueeze(0)
            label_embeddings = label_embeddings.expand(drop_output.shape[0], -1, -1)
            label_embeddings = self.GCN3(label_embeddings, self.label_adj)
            label_embeddings = self.relu1(label_embeddings)
            label_embeddings = self.GCN4(label_embeddings, self.label_adj)
            doc_embedding = drop_output
            word_label_att = torch.bmm(doc_embedding, label_embeddings.transpose(1, 2))
            Att_v = torch.max(word_label_att, keepdim=True, dim=-1)[0]
            Att_v_tanh = torch.tanh(Att_v)
            H_enc = Att_v_tanh * doc_embedding

            H_enc = torch.mean(H_enc, dim=1)
            concatenated = torch.cat((H_enc, bert_output), dim=1)
            
            logits = self.classifier(concatenated)
            prediction, _ = torch.max(logits, dim=0, keepdim=True)
            predictions.append(prediction)

        stacked_predictions = torch.cat(predictions, dim=0)
        return stacked_predictions

    def generate_attention_mask(self, input_ids):

        # 创建一个与input_ids相同大小的张量，并填充为1
        attention_mask = torch.ones_like(input_ids)
        # 将输入张量中为1的位置设置为0，以便在注意力机制中屏蔽填充部分
        attention_mask = attention_mask * (input_ids != 0)
        return attention_mask

    def generate_token_type_ids(self, input_ids):
        # 创建一个与输入张量形状相同的全零张量
        token_type_ids = torch.zeros_like(input_ids, dtype=torch.int64)
        return token_type_ids

    def data_selfattention(self, query, key, value, mask): #一般query为解码器的部分  k 和 v为编码器的部分
        d_k = query.size(-1)  # 64=d_k
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # x = scores
        # expanded_mask = mask.unsqueeze(-1).unsqueeze(-1)
        # 然后除以sqrt(d_k)，防止过大的亲密度。
        # new_mask[mask.unsqueeze(2).bool().expand_as(new_mask)] = 1
        # _new_mask = new_mask.permute(0, 2, 1)
        mask_label = torch.ones(key.shape[0],key.shape[1])
        mask_label = mask_label.unsqueeze(-1)
        mask_label = mask_label.float().cuda()

        mask = mask.float()
        mask = mask.unsqueeze(2)
        mask = torch.matmul(mask, mask_label.transpose(-2,-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 将填充的词用0覆盖
            # _scores = _scores.masked_fill(_new_mask == 0, -1e9)
            # new_tensor[:, :, 4] = -float("inf")
            # 使用mask，对已经计算好的scores，按照mask矩阵，填-1e9，
            # 然后在下一步计算softmax的时候，被设置成-1e9的数对应的值~0,被忽视
            p_attn = F.softmax(scores, dim=-2)

        x = torch.matmul(p_attn, value)
        return torch.matmul(p_attn, value)


import math
class GraphConvolution(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(in_features, out_features))
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
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

