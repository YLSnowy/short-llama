import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNN(nn.Module):
    def __init__(self, **kwargs):
        super(CNN, self).__init__()

        self.MODEL = kwargs["MODEL"]
        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.FILTERS = kwargs["FILTERS"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        self.IN_CHANNEL = 2

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        self.embedding = nn.Sequential(
            nn.Linear(self.WORD_DIM, self.WORD_DIM),
            nn.LeakyReLU()
        )
        self.embedding2 = nn.Sequential(
            nn.Linear(self.WORD_DIM, self.WORD_DIM),
            nn.LeakyReLU()
        )

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
            setattr(self, f'conv_{i}', conv)

        self.fc = nn.Sequential(
            nn.Linear(sum(self.FILTER_NUM), sum(self.FILTER_NUM)),
            nn.LeakyReLU(),
            nn.Linear(sum(self.FILTER_NUM), sum(self.FILTER_NUM)),
            nn.LeakyReLU(),
            nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
        )

    def get_conv(self, i):
        return getattr(self, f'conv_{i}')

    def forward(self, inp):
        x = self.embedding(inp).view(-1, inp.shape[1], self.WORD_DIM * self.MAX_SENT_LEN)
        # print(x.shape)
        if self.MODEL == "multichannel":
            x2 = self.embedding2(inp).view(-1, inp.shape[1], self.WORD_DIM * self.MAX_SENT_LEN)
            # print(x2.shape)
            x = torch.cat((x, x2), 1)

        conv_results = []
        for i in range(len(self.FILTERS)):
            conv_output = self.get_conv(i)(x)  # 调用 self.get_conv(i) 方法获取属性并在输入 x 上进行卷积操作
            # print(conv_output.shape)
            relu_output = F.relu(conv_output)  # 应用 ReLU 激活函数
            pooled_output = F.max_pool1d(relu_output, self.MAX_SENT_LEN - self.FILTERS[i] + 1)  # 进行最大池化
            # print(pooled_output.shape)
            reshaped_output = pooled_output.view(-1, self.FILTER_NUM[i])  # 重塑输出形状
            conv_results.append(reshaped_output)

        x = torch.cat(conv_results, 1)
        # print(x.shape)
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        x = self.fc(x)

        return x


class CNN2D(nn.Module):
    def __init__(self, **kwargs):
        super(CNN2D, self).__init__()

        self.MODEL = kwargs["MODEL"]
        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.FILTERS = kwargs["FILTERS"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        self.IN_CHANNEL = 2

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        self.embedding = nn.Identity()
        self.embedding2 = nn.Identity()

        for i in range(len(self.FILTERS)):
            conv = nn.Conv2d(self.IN_CHANNEL, self.FILTER_NUM[i], (self.FILTERS[i], self.WORD_DIM * self.FILTERS[i]),
                             stride=(1, self.WORD_DIM))
            setattr(self, f'conv_{i}', conv)

        for i in range(len(self.FILTERS)):
            conv_outdim = 11 - self.FILTERS[i]
            linear1 = nn.Sequential(
                nn.Linear(conv_outdim * conv_outdim, conv_outdim),
                nn.LeakyReLU()
            )
            linear2 = nn.Sequential(
                nn.Linear(conv_outdim, 1),
                nn.LeakyReLU()
            )
            setattr(self, f'linear1_{i}', linear1)
            setattr(self, f'linear2_{i}', linear2)

        for i in range(10):
            fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)
            setattr(self, f'fc_{i}', fc)

    def get_conv(self, i):
        return getattr(self, f'conv_{i}')

    def get_linear1(self, i):
        return getattr(self, f'linear1_{i}')

    def get_linear2(self, i):
        return getattr(self, f'linear2_{i}')

    def get_fc(self, i):
        return getattr(self, f'fc_{i}')

    def forward(self, inp):
        bsz, channel, h, _, _ = inp.shape
        x = self.embedding(inp).view(bsz, channel, h, -1)
        x2 = self.embedding2(inp).view(bsz, channel, h, -1)
        x = torch.cat((x, x2), 1)

        conv_results = []
        for i in range(len(self.FILTERS)):
            conv_output = self.get_conv(i)(x)
            conv_output = F.relu(conv_output)
            conv_output = conv_output.view(bsz, 100, -1)
            print(conv_output.shape)

            conv_output = self.get_linear1(i)(conv_output)
            conv_output = conv_output.view(bsz, 100, -1)
            conv_output = self.get_linear2(i)(conv_output)
            conv_output = conv_output.view(bsz, 100)
            # print(conv_output.shape)

            conv_results.append(conv_output)

        x = torch.cat(conv_results, 1)

        del conv_results
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)

        pre = []
        for i in range(10):
            fc = self.get_fc(i)(x)
            pre.append(fc.unsqueeze(1))
            # print(fc.shape)

        x = torch.cat(pre, 1)

        return x


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()

        V = 32000
        D = 4096
        C = 5
        Ci = 1
        Co = 100
        Ks = [3, 4, 5]

        self.embed = nn.Embedding(V, D)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3)
             for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2)
             for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit


class CNN_Origin(nn.Module):
    def __init__(self, **kwargs):
        super(CNN_Origin, self).__init__()

        self.MODEL = kwargs["MODEL"]
        self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = kwargs["WORD_DIM"]
        self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
        self.CLASS_SIZE = kwargs["CLASS_SIZE"]
        self.FILTERS = kwargs["FILTERS"]
        self.FILTER_NUM = kwargs["FILTER_NUM"]
        self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        self.IN_CHANNEL = 1

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        # one for UNK and one for zero padding
        # self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
        if self.MODEL == "static" or self.MODEL == "non-static" or self.MODEL == "multichannel":
            self.WV_MATRIX = kwargs["WV_MATRIX"]
            self.embedding.weight.data.copy_(self.WV_MATRIX)
            if self.MODEL == "static":
                self.embedding.weight.requires_grad = False
            elif self.MODEL == "multichannel":
                self.embedding2 = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
                self.embedding2.weight.data.copy_(self.WV_MATRIX)
                self.embedding2.weight.requires_grad = False
                self.IN_CHANNEL = 2

        for i in range(len(self.FILTERS)):
            conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
            setattr(self, f'conv_{i}', conv)

        self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)

    def get_conv(self, i):
        return getattr(self, f'conv_{i}')

    def forward(self, inp):
        x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
        if self.MODEL == "multichannel":
            x2 = self.embedding2(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
            x = torch.cat((x, x2), 1)

        conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1)
                .view(-1, self.FILTER_NUM[i])
            for i in range(len(self.FILTERS))]

        x = torch.cat(conv_results, 1)
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        x = self.fc(x)

        return x


class CNN_MaxBSZ(nn.Module):
    def __init__(self, **kwargs):
        super(CNN_MaxBSZ, self).__init__()

        # self.MODEL = kwargs["MODEL"]
        # self.BATCH_SIZE = kwargs["BATCH_SIZE"]
        # self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
        self.WORD_DIM = 4096
        self.VOCAB_SIZE = 32000
        self.CLASS_SIZE = 2
        self.FILTERS = [3, 4, 5]
        self.FILTER_NUM = [100, 100, 100]
        self.DROPOUT_PROB = 0.5
        self.IN_CHANNEL = 2

        assert (len(self.FILTERS) == len(self.FILTER_NUM))

        self.embedding = nn.Identity()
        self.embedding2 = nn.Identity()

        for i in range(len(self.FILTERS)):
            conv = nn.Conv2d(self.IN_CHANNEL, self.FILTER_NUM[i], (self.FILTERS[i], self.WORD_DIM * self.FILTERS[i]),
                             stride=(3, self.WORD_DIM))
            setattr(self, f'conv_{i}', conv)

        for i in range(len(self.FILTERS)):
            conv_outdim = 11 - self.FILTERS[i]
            linear1 = nn.Sequential(
                nn.Linear(10 * conv_outdim, 10),
                nn.LeakyReLU()
            )
            linear2 = nn.Sequential(
                nn.Linear(10, 1),
                nn.LeakyReLU()
            )
            setattr(self, f'linear1_{i}', linear1)
            setattr(self, f'linear2_{i}', linear2)


        self.output_mlp = nn.Sequential(
            nn.Linear(sum(self.FILTER_NUM), sum(self.FILTER_NUM)),
            nn.LeakyReLU(),
            nn.Linear(sum(self.FILTER_NUM), sum(self.FILTER_NUM)//2),
            nn.LeakyReLU(),
            nn.Linear(sum(self.FILTER_NUM)//2, self.CLASS_SIZE*32)
        )

    def get_conv(self, i):
        return getattr(self, f'conv_{i}')

    def get_linear1(self, i):
        return getattr(self, f'linear1_{i}')

    def get_linear2(self, i):
        return getattr(self, f'linear2_{i}')

    def forward(self, inp):
        bsz, channel, h, _ = inp.shape
        x = self.embedding(inp).view(bsz, channel, h, -1)
        x2 = self.embedding2(inp).view(bsz, channel, h, -1)
        x = torch.cat((x, x2), 1)
        # print(x.shape)

        conv_results = []
        for i in range(len(self.FILTERS)):
            conv_output = self.get_conv(i)(x)
            conv_output = F.relu(conv_output)
            conv_output = conv_output.view(bsz, 100, -1)
            conv_output = self.get_linear1(i)(conv_output)
            conv_output = conv_output.view(bsz, 100, -1)
            conv_output = self.get_linear2(i)(conv_output)
            conv_output = conv_output.view(bsz, 100)
            # print(conv_output.shape)
      
            conv_results.append(conv_output)

        x = torch.cat(conv_results, 1)
        # print(x.shape)

        del conv_results
        x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
        x = self.output_mlp(x)
        x = x.view(bsz, 32, self.CLASS_SIZE)

        return x







if __name__ == '__main__':
    # 数据被打包为每32一个batch，一共b个 
    # cache -> (b, 32, 10, 32, 128)
    # input -> (b, 1, 32, 40960)
    # output -> (b, 32, 2) 0-> short 1-> long
    model = CNN_MaxBSZ()

    i = torch.zeros(512, 1, 32, 40960)
    o = model(i)
    print(o.shape)
