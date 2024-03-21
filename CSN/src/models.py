from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import flags

FLAGS = flags.FLAGS
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# CasEncoder
class CasEncoder(nn.Module):
    def __init__(self, emb_dim, gru_units):
        super(CasEncoder, self).__init__()
        self.gru = nn.GRU(emb_dim, gru_units, 2, batch_first=True, bidirectional=True)
        #self.pos_emb= nn.Embedding(1000,FLAGS.pos_dim)
        #self.pos_emb= nn.Embedding(1000,FLAGS.pos_dim)
        self.drop=nn.Dropout(0.2)


    def forward(self, input):
        batch_size=input.shape[0]
        max_seq=input.shape[1]
        #pos_feat=self.drop(self.pos_emb(torch.ones(batch_size).long().cuda()).reshape([batch_size,1,FLAGS.pos_dim]))
        
        #if FLAGS.s_weight!='none':
        #    emb_dim=FLAGS.emb_dim+1
        emb_dim=FLAGS.emb_dim
        if 0:
            hidden= torch.zeros(batch_size,max_seq, FLAGS.pos_dim+emb_dim).to(device)
            for i in range(0,max_seq):
                #pos_feat=torch.cat([pos_feat,self.drop(self.pos_emb(torch.ones(batch_size).long().cuda()*(i+1)).reshape([batch_size,1,FLAGS.pos_dim]))],dim=1)
                hidden[:,i,:]=torch.cat([input[:,i,:],self.drop(self.pos_emb(torch.ones(batch_size).long().to(device)*i))],dim=1)
            #_, hn = self.gru(torch.cat([input,pos_feat],dim=2))
            _, hn = self.gru(hidden)
            output = torch.cat([hn[2], hn[3]], dim=1)
            return output
        else:
            _, hn = self.gru(input)
            output = torch.cat([hn[2], hn[3]], dim=1)
            return output


# ContrastiveLoss
class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))  # 超参数 温度
        self.register_buffer("negatives_mask",
                             (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())  # 主对角线为0，其余位置全为1的mask矩阵

    def forward(self, emb_i, emb_j):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / self.temperature)  # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


# MSLE
class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, label, pred):
        return torch.sqrt(self.mse(torch.log(label+1 ), torch.log(pred+1 )))
        
