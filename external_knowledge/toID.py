import pickle

with open('./EUR57K/EUR_all_triple.pickle', 'rb') as f:
    all_ere = pickle.load(f)

neighbors = []
for key, value in all_ere.items():
    neighbors.append(value)

entities = set()
relations = set()
for neighbor in neighbors:
    for item in neighbor:
        entities.add(item[0])
        relations.add(item[1])
        entities.add(item[2])

entity2id = {entity: idx for idx, entity in enumerate(entities)}
relation2id = {relation: idx for idx, relation in enumerate(relations)}
triplets = [[entity2id[neighbor[0]], entity2id[neighbor[2]], relation2id[neighbor[1]]] for neighbor_list in neighbors for neighbor in neighbor_list]
print()
# 打开文件以写入
with open('entity2id.txt', 'w', encoding='utf-8') as file:
    for key, value in entity2id.items():
        # 写入每行数据，以制表符分隔
        file.write(f'{key}\t{value}\n')

# 打开文件以写入
with open('relation2id.txt', 'w', encoding='utf-8') as file:
    for key, value in relation2id.items():
        # 写入每行数据，以制表符分隔
        file.write(f'{key}\t{value}\n')

with open('triplets.txt', 'w', encoding='utf-8') as file:
    for value in triplets:
        # 写入每行数据，以制表符分隔
        file.write(f'{value[0]}\t{value[1]}\t{value[2]}\n')

