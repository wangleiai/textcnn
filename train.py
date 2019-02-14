import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import config
from model import textCNN
from loadData import get_batch_data, get_data_nums
from tensorboardX import SummaryWriter
import numpy as np

def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


net = textCNN(config.VOCAB_SIZE, config.EMBEDDING_SIZE, config.DROPOUT, config.CLASS_NUM)
writer = SummaryWriter(log_dir='logs')

if config.IS_CUDA:
    net = net.cuda()

optimizer = torch.optim.Adam(net.parameters(), config.LEARNING_RATE)

net.train()

batch_data = get_batch_data('data/train.txt', batch_size=config.BATCH_SIZE, word_max_length=config.MAX_LENGTH)
data_nums = get_data_nums('data/train.txt')

val_batch_data = get_batch_data('data/val.txt', batch_size=config.BATCH_SIZE//4, word_max_length=config.MAX_LENGTH)
val_data_nums = get_data_nums('data/val.txt')
steps = 0
for epoch in range(1, config.EPOCHES+1):
    i = 0
    total_loss = 0
    total_acc = 0
    for x, y in batch_data() :
        if i >=data_nums//config.BATCH_SIZE: break

        x = Variable(torch.from_numpy(x))
        y = Variable(torch.from_numpy(np.array(y)))

        if config.IS_CUDA:
            x = x.cuda()
            y = y.cuda()
        logit = net(x.long())

        optimizer.zero_grad()
        loss = F.cross_entropy(logit, y.long())
        loss.backward()
        optimizer.step()
        acc = get_acc(output=logit, label=y.long())

        total_loss += loss.item()
        total_loss += acc
        writer.add_scalar('data/train_acc', acc, steps)
        writer.add_scalar('data/train_loss', loss, steps)
        print('epoch:',epoch ,'train_acc: ', acc, '  train_loss', loss.item())
        i += 1
        steps += 1

    if epoch%2==0:
        val_loss = 0
        val_acc = 0
        i = 0
        for x, y in val_batch_data():
            if i >= val_data_nums // (config.BATCH_SIZE//4): break

            x = Variable(torch.from_numpy(x))
            y = Variable(torch.from_numpy(np.array(y)))

            if config.IS_CUDA:
                x = x.cuda()
                y = y.cuda()
            logit = net(x.long())
            loss = F.cross_entropy(logit, y.long())
            val_loss += loss.item()

            acc = get_acc(logit, y.long())
            val_acc += acc
            i += 1

        val_loss = val_loss/ (val_data_nums // (config.BATCH_SIZE//4))
        val_acc = val_acc/( val_data_nums // (config.BATCH_SIZE//4))
        print('epoch:', epoch, 'train_acc: ', val_acc, '  val_loss', val_loss)
        writer.add_scalar('data/val_acc', val_acc, epoch)
        writer.add_scalar('data/val_loss', val_loss, epoch)
        save_path = 'epoch：{}_val_acc：{:.6f}_val_loss：{:.4f}.pkl'\
            .format(epoch, total_acc/(data_nums//config.BATCH_SIZE), total_loss/(data_nums//config.BATCH_SIZE),
                                                                                                            val_acc,val_loss)
        torch.save(net, save_path)
writer.close()