import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class textCNN(nn.Module):

    def __init__(self, vocab_size, embedding_dim, dropout ,class_num):
        super().__init__()
        self.voacb_size = vocab_size
        self.embedding_dim = embedding_dim
        self.class_num = class_num
        self.dropout =dropout

        self.embed = nn.Embedding(self.voacb_size, self.embedding_dim)
        self.conv1 = nn.Conv2d(1, 100, (3, self.embedding_dim))
        self.conv2 = nn.Conv2d(1, 100, (4, self.embedding_dim))
        self.conv3 = nn.Conv2d(1, 100, (5, self.embedding_dim))

        self.dropout = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(3*100, class_num)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))  # [128, 1, 500, 100]  [128, 100, 498, 1]
        x = x.squeeze(3)  # [128, 100, 498]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # [128, 100]
        return x

    def forward(self, input_data):
        # print(input_data.shape)
        x = self.embed(input_data) # (N, W, D)

        x = x.unsqueeze(1)  # (N, 1, W, D)
        x1 = self.conv_and_pool(x, self.conv1) #(N,Co)
        x2 = self.conv_and_pool(x, self.conv2) #(N,Co)
        x3 = self.conv_and_pool(x, self.conv3) #(N,Co)
        x = torch.cat((x1, x2, x3), 1)  # (N,len(Ks)*Co)

        x = self.dropout(x)
        logit = self.fc1(x)
        return logit

if __name__ == '__main__':
    import config
    # vocab_size, embedding_dim, dropout ,class_num
    net = textCNN(config.VOCAB_SIZE, config.EMBEDDING_SIZE, config.DROPOUT, config.CLASS_NUM)
    print(net)