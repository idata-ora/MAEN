import argparse
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class RESNET_EXT(nn.Module):
    def __init__(self, RESNET, input_size, output_class):
        super(RESNET_EXT, self).__init__()
        self.feature_extractor = RESNET
        self.fc = nn.Linear(input_size, output_class)

    def forward(self, x):
        device = x.device
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        # print(feats.shape, c.shape)  # torch.Size([32, 512]) torch.Size([32, 4])
        return feats.view(feats.shape[0], -1), c

class Embeddings(nn.Module):
    '''
    对图像进行编码，把图片当做一个句子，把图片分割成块，每一块表示一个单词
    '''

    def __init__(self, config):
        super(Embeddings, self).__init__()
        # 设置可学习的位置编码信息，（1,196+1,786）
        self.position_embeddings = nn.Parameter(torch.zeros(config.hidden_size, 1))
        # 设置可学习的分类信息的维度
        self.classifer_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = nn.Dropout((config.dropout_rate))
        self.map_linA = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size))
        self.map_linB = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size))
        self.instance_lin = nn.Sequential(nn.Linear(10, config.hidden_size-1), nn.ReLU())
    def forward(self, x, InstancePred):
        c = x.shape[0]
        for i in range(len(InstancePred)):
            _, m_indices = torch.sort(InstancePred[i], 0, descending=True)  # sort class scores along the instance dimension, m_indices in shape N x C
            m_feats = torch.index_select(x.view(x.shape[1],-1), dim=0, index=m_indices[0, :])
            # print(m_feats.shape)  # torch.Size([2, 512] torch.Size([2, 512] torch.Size([2, 512] torch.Size([4, 512]
            if i == 0:
                instance_pro = m_feats
            else:
                instance_pro = torch.cat((instance_pro,m_feats),dim=0)  # torch.Size([10, 512]
        # instance_pro = torch.mm(self.position_embeddings, instance_pro)  # # torch.Size([512, 10]
        # instance_pro_para = self.instance_lin(self.position_embeddings)
        instance_pro = self.instance_lin(instance_pro.transpose(0,1))
        # print(instance_pro.shape)
        instance_pro = torch.cat((instance_pro, self.position_embeddings), dim=1)
        # print(instance_pro.shape)  # torch.Size([512, 512])

        bs = x.shape[0]
        cls_tokens = self.classifer_token.expand(bs, 1, 512)
        x = x.flatten(2)
        # print(cls_tokens.shape,x.shape)
        x = torch.cat((cls_tokens, x), dim=1)  # 将分类信息与图片块进行拼接（bs,197,768）
        A = self.map_linA(x).view(x.shape[1],-1)
        B = self.map_linB(x).view(x.shape[1],-1)
        # print(A.shape, B.shape)  # [1, 624, 512]  [1, 624, 512]
        Map = torch.mm(A.transpose(0,1), B)  # [512, 512]
        embeddings = Map + instance_pro  # + instance_pro_para  # 将图片块信息和对其位置信息进行相加(bs,197,768)
        embeddings = self.dropout(embeddings)
        embeddings = embeddings.view(c, embeddings.shape[0], embeddings.shape[1])
        return embeddings

# 构建self-Attention模块
class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention,self).__init__()
        self.vis=vis
        self.num_attention_heads=config.num_heads #12
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)  # 768/12=64
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 12*64=768

        self.query = nn.Linear(config.hidden_size, self.all_head_size) #wm,768->768，Wq矩阵为（768,768）
        self.key = nn.Linear(config.hidden_size, self.all_head_size) #wm,768->768,Wk矩阵为（768,768）
        self.value = nn.Linear(config.hidden_size, self.all_head_size) #wm,768->768,Wv矩阵为（768,768）
        self.out = nn.Linear(config.hidden_size, config.hidden_size)  # wm,768->768
        self.attn_dropout = nn.Dropout(config.attention_dropout_rate)
        self.proj_dropout = nn.Dropout(config.attention_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
        self.num_attention_heads, self.attention_head_size)  # wm,(bs,197)+(12,64)=(bs,197,12,64)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # wm,(bs,12,197,64)

    def forward(self, hidden_states):
        # hidden_states为：(bs,197,768)
        mixed_query_layer = self.query(hidden_states)#wm,768->768
        mixed_key_layer = self.key(hidden_states)#wm,768->768
        mixed_value_layer = self.value(hidden_states)#wm,768->768

        query_layer = self.transpose_for_scores(mixed_query_layer)#wm，(bs,12,197,64)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))#将q向量和k向量进行相乘（bs,12,197,197)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)#将结果除以向量维数的开方
        attention_probs = self.softmax(attention_scores)#将得到的分数进行softmax,得到概率
        weights = attention_probs if self.vis else None#wm,实际上就是权重
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) #将概率与内容向量相乘
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) #wm,(bs,197)+(768,)=(bs,197,768)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights #wm,(bs,197,768),(bs,197,197)

#3.构建前向传播神经网络
#两个全连接神经网络，中间加了激活函数
class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.mlp_dim)#wm,786->3072
        self.fc2 = nn.Linear(config.mlp_dim, config.hidden_size)#wm,3072->786
        self.act_fn = torch.nn.functional.gelu#wm,激活函数
        self.dropout = nn.Dropout(config.dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)#wm,786->3072
        x = self.act_fn(x)#激活函数
        x = self.dropout(x)#wm,丢弃
        x = self.fc2(x)#wm3072->786
        x = self.dropout(x)
        return x

# 4.构建编码器的可重复利用的Block()模块：每一个block包含了self-attention模块和MLP模块
class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size  # wm,768
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)  # wm，层归一化
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h  # 残差结构

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h  # 残差结构
        return x, weights

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.num_layers):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class InstanceFC(nn.Module):
    def __init__(self, in_size, out_size=2):
        super(InstanceFC, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size),nn.ReLU())
    def forward(self, feats):
        x = self.fc(feats)
        return x

class BagMIL(nn.Module):
    def __init__(self, config, FClist, input_size, output_class):
        super(BagMIL, self).__init__()
        self.fcc1 = nn.Conv1d(input_size, input_size, kernel_size=input_size)
        self.fcc2 = nn.Linear(input_size, output_class)
        self.FClist = nn.ModuleList(FClist)#FClist
        self.embed = Embeddings(config)
        self.encoder = Encoder(config, vis=True)
    def forward(self, feats):
        x = feats.view(1,feats.shape[0],-1)
        ERIC = self.FClist[0](x).view(x.shape[1],-1)
        PRIC = self.FClist[1](x).view(x.shape[1],-1)
        Her2IC = self.FClist[2](x).view(x.shape[1],-1)
        IHCIC = self.FClist[3](x).view(x.shape[1],-1)
        InstancePred = [ERIC, PRIC, Her2IC,IHCIC]
        
        out_embedding = self.embed(x, InstancePred)
        out_encoder, _ = self.encoder(out_embedding)
        C = self.fcc1(out_encoder)  # 1 x C x 1
        C = self.fcc2(C.view(x.shape[0],-1))
        return C, InstancePred

    def load_weights(self, weights_path):
        model_dict = self.state_dict()
        pretrained_dict = torch.load(weights_path)  # 这里model_path的后缀是.pth可直接读取
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  # 只保留预训练模型中，自己建的model有的参数
        model_dict.update(pretrained_dict)  # 将预训练的值，更新到自己模型的dict中
        self.load_state_dict(model_dict)  # model加载dict中的数据，更新网络的初始值

def get_config():
    """Returns the ViT-B/16 configuration."""
    config = argparse.ArgumentParser(description='patch features learned by SimCLR')
    config.add_argument('--hidden_size', default=512, type=int)
    config.add_argument('--mlp_dim', default=3072, type=int)
    config.add_argument('--num_heads', default=16, type=int)
    config.add_argument('--num_layers', default=16, type=int)
    config.add_argument('--attention_dropout_rate', default=0.0, type=float)
    config.add_argument('--dropout_rate', default=0.1, type=float)
    config.add_argument('--classifier', default='token', type=str)
    config.add_argument('--representation_size', default=None, type=int)
    args = config.parse_args()
    return args

if __name__ == "__main__":
    config = get_config()
    ER_classifier = InstanceFC(in_size=512, out_size=2)
    PR_classifier = InstanceFC(in_size=512, out_size=2)
    Her2_classifier = InstanceFC(in_size=512, out_size=2)
    IHC_classifier = InstanceFC(in_size=512, out_size=4)
    FClist = [ER_classifier, PR_classifier, Her2_classifier, IHC_classifier]
    model = BagMIL(config, FClist, input_size=512, output_class=4)
    model.eval()
    image = torch.randn(1, 623, 512)   # .cuda()
    with torch.no_grad():
        C, InstancePred = model.forward(image)
    print(C.size())   # torch.Size([2, 1024])
