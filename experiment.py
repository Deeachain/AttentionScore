import kashgari
from kashgari.embeddings import BERTEmbedding
import numpy as np


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def scaled_dot_product_attention(Q1, K1, V1, Q2, K2, V2, scope='scaled_dot_product_attention'):
    Q1, K1, V1 = np.asarray(Q1).reshape(1, -1), np.asarray(K1).reshape(1, -1), np.asarray(V1).reshape(1, -1)
    d_k1 = Q1.shape[-1]
    # Q * k
    outputs1 = np.matmul(Q1, np.transpose(K1, [1, 0]))
    outputs1 = np.squeeze(outputs1)
    # scale
    outputs1 /= d_k1 ** 0.5

    Q2, K2, V2 = np.asarray(Q2).reshape(1, -1), np.asarray(K2).reshape(1, -1), np.asarray(V2).reshape(1, -1)
    d_k2 = Q2.shape[-1]
    # Q * k
    outputs2 = np.matmul(Q1, np.transpose(K2, [1, 0]))
    outputs2 = np.squeeze(outputs2)
    # scale
    outputs2 /= d_k2 ** 0.5

    outputs = softmax([outputs1, outputs2])

    # weighted sum (context vectors)
    attention1 = np.multiply(outputs[0], V1)
    attention2 = np.multiply(outputs[0], V2)

    return attention1, attention2



'''获取词向量'''
'''加载bert预训权重,https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
   需要解压文件,将文件夹路径放入BERTEmbedding中'''
bert = BERTEmbedding('/media/ding/Files/ubuntu_study/Datasets/chinese-bert_chinese_wwm_L-12_H-768_A-12',
                     task=kashgari.CLASSIFICATION,
                     sequence_length=7)

sents = 'The quick fox jumped over lazy dog.'
sents = sents.replace('.', ' ')
input = sents.split()

embed_tensor = bert.embed_one(input)       #shape (7, 3072) 7为序列长度


#获取(7, 3072)的q k v
#初始化每个词的q k v为本身,实际网络中q k v是学习得到的向量
'''Attention1'''
index1 = input.index('fox')
index2 = input.index('jumped')
q1, k1, v1 = embed_tensor[index1], embed_tensor[index1], embed_tensor[index1]
q2, k2, v2 = embed_tensor[index2], embed_tensor[index2], embed_tensor[index2]

attention1, attention2 = scaled_dot_product_attention(q1, k1, v1, q2, k2, v2)
print('fox 与自己的attention', attention1[0][index1])
print('fox 与jumped的attention', attention2[0][index2])


'''Attention2'''
index1 = input.index('over')
index2 = input.index('dog')
q1, k1, v1 = embed_tensor[index1], embed_tensor[index1], embed_tensor[index1]
q2, k2, v2 = embed_tensor[index2], embed_tensor[index2], embed_tensor[index2]

attention1, attention2 = scaled_dot_product_attention(q1, k1, v1, q2, k2, v2)
print('over 与自己的attention', attention1[0][index1])
print('over 与dog的attention', attention2[0][index2])


'''
output is

fox 与自己的attention -0.35012457
fox 与jumped的attention 0.016163977
over 与自己的attention 0.07522257
over 与dog的attention 0.37237185
'''