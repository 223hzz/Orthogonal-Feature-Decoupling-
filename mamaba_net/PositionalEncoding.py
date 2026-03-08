import torch
import torch.nn as nn


class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=5000):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)#从0到max_length个向量，并且扩展为一个列向量
        div_term = torch.exp(
            #以自然数为底进行运算
            torch.arange(0, embedding_dim, 2).float()#生成向量长度0到embedding_dim步长为2,此处为一个行向量
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)#计算公式中10000**（2i/d_model)
        )
        #position * div_term的结果是一个max_length x (embedding_dim / 2)的矩阵，所以它的形状与pe矩阵中被选定的偶数列完美匹配。
        pe[:, 0::2] = torch.sin(position * div_term)#一个行向量乘以一个列向量得到一个矩阵：长为max_length，列为embedding_dim/2
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)#扩展一个维度，size=(1, L, d_model)，为了后续与word_embedding相加,意为batch维度下的操作相同，transpose之后意为(L, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #也可写成x = x + self.pe[: x.size(0), :, :]，x.size(0)为向量个数
        x = x + self.pe[: x.size(0), :]#[seq_len, 1, embedding_dim] 的张量可以与 [seq_len, batch_size, embedding_dim] 的张量相加。在这个过程中，第二个维度（1）会被自动扩展以匹配 batch_size
        return x
        #seq_length = x.size(1)
        # 确保位置编码与序列长度匹配
        #x = x + self.pe[:seq_length, :]


# 在 LearnedPositionalEncoding 中，位置编码是一个可学习的参数，通过嵌入层 (nn.Embedding) 实现。这种方法允许模型在训练过程中学习最适合特定任务的位置编码。
# 在代码中，LearnedPositionalEncoding 被初始化时，其最大位置编码长度（max_position_embeddings）和序列长度（seq_length）都被设置为 self.decoder_dim。
# 这意味着学习到的位置编码的长度是基于 decoder_dim 的值来确定的。
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):#self.decoder_dim, self.embedding_dim, self.decoder_dim
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)#一个嵌入层，用于学习位置编码
        #生成max_position_embeddings个张量，每个张量的大小为embedding_dim

        self.seq_length = seq_length

        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),#expand((1, -1)) 将一维张量 [max_position_embeddings] 转换成了二维张量 [1, max_position_embeddings]
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]
            #从预先创建的 position_ids 缓冲区中取出前 self.seq_length 个位置索引
        position_embeddings = self.pe(position_ids)
        #由于position_ids的形状是[1, self.seq_length]，每个位置索引会被映射到一个embedding_dim维的向量，所以position_embeddings的输出形状将是[1, self.seq_length, embedding_dim]
        return x + position_embeddings
    #
    # if position_ids is None:
    #     # 获取当前批次的序列长度
    #     current_seq_length = x.size(1)
    #     position_ids = self.position_ids[:, :current_seq_length]
    #     # 从预先创建的 position_ids 缓冲区中取出与当前序列长度匹配的位置索引
    # position_embeddings = self.pe(position_ids)
    # return x + position_embeddings