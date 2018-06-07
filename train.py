# coding=utf-8
# Using '_' as the blank symbol

from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn
 
from ds import MyDataset
from ds import get_train_val_dataset
from crnn import CRNN
from warpctc_pytorch import CTCLoss
import argparse
import os
from operator import mul
from functools import reduce
from os.path import join

import csv

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--gpus', '-g', type=str, default='0', help='which gpu(s) it could use')
parser.add_argument('--data_dir', help='path to dataset directory', default='../train_merged')
parser.add_argument('--label_csv_path', help='path to train.csv', default='../train_all.csv')
parser.add_argument('--modeldir', help='path to save model', default='./saved_models_all/')
parser.add_argument('--modelname', help='model name', default='params-9.pkl')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--train_batchsize', type=int, default=256, help='train batch size')
parser.add_argument('--val_batchsize', type=int, default=1, help='val batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate for Critic, default=1.0')
parser.add_argument('--acc_initial', type=float, default=0, help='initial accuracy on val, default=0')
parser.add_argument('--tr_acc_initial', type=float, default=0, help='initial accuracy on train, default=0')
parser.add_argument('--num_epoch', type=int, default=2500, help='number of epochs to train for')
parser.add_argument('--num_val', type=int, help='ite number of evaluate samples', default=1000)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--nrnn', type=int, default=2, help='the number of layers of rnn(sru)')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout applied between RNN layers(sru), default=0.0')
parser.add_argument('--variational_dropout', type=float, default=0.0,
                    help='variational dropout applied on linear transformation(sru), default=0.0')
parser.add_argument('--ndisplay', type=int, help='ite number of display', default=20)
parser.add_argument('--nsave', type=int, help='ite number of save model', default=1)
parser.add_argument('--finetune', action='store_true', help='Whether to finetune')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is adam)')
parser.add_argument('--rms', action='store_true', help='Whether to use rms (default is adam)')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

if not os.path.exists(opt.modeldir):
    os.mkdir(opt.modeldir)

acc_ini = opt.acc_initial
tr_acc_ini = opt.tr_acc_initial


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])



dset_train, dset_val, character_str, char2index = \
    get_train_val_dataset(opt.label_csv_path, opt.data_dir, save_character_str = True, transform=transform, size=(256, 48), max_length=None)
#print(dset_train.img_name_list)
dloader_train = torch.utils.data.DataLoader(dset_train, shuffle=True, batch_size=opt.train_batchsize,
                                            num_workers=int(opt.workers))
dloader_val = torch.utils.data.DataLoader(dset_val, shuffle=True, batch_size=opt.val_batchsize,
                                          num_workers=int(opt.workers))
dloader_train_eval = torch.utils.data.DataLoader(dset_train, shuffle=True, batch_size=opt.val_batchsize,
                                            num_workers=int(opt.workers))
print(character_str)
def weights_init(m):
    #    classname = m.__class__.__name__
    #    if classname.find('Conv') != -1:
    if isinstance(m, nn.Conv2d):
        # m.weight.data.normal_(0.0, 0.02)
        # m.weight.data.normal_(0.0, 2)
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='leaky_relu')
        m.bias.data.fill_(0)
    #    elif classname.find('BatchNorm') != -1:
    elif isinstance(m, nn.BatchNorm2d):
        # m.weight.data.normal_(1.0, 0.02)
        m.weight.data.uniform_(1.0, 5)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.GRU):
        nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('leaky_relu'))

#    elif isinstance(m, nn.Linear):
#        m.weight.data.normal_(0.0, 0.02)
#        m.bias.data.fill_(0)


net = CRNN(48, 1, len(char2index), 256, opt.nrnn, opt.dropout, opt.variational_dropout, leakyRelu=True)
print(net)
params = net.state_dict()
params_shape = []
for k, v in params.items():
    #    print(k, v.numpy().shape, reduce(mul, v.numpy().shape))
    params_shape.append(reduce(mul, v.numpy().shape))
params_total = sum(params_shape)
print('params_total:', params_total)

if opt.finetune:
    print('Loading model from', opt.modeldir + opt.modelname)
    net.load_state_dict(torch.load(opt.modeldir + opt.modelname))
else:
    print('create new model')
    net.apply(weights_init)

if opt.ngpu > 1:
    # print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = nn.DataParallel(net, device_ids=range(opt.ngpu))
net.cuda()
criterion = CTCLoss().cuda()

if opt.adadelta:
    optimizer = optim.Adadelta(net.parameters(), lr=opt.lr)  # , weight_decay=1e-8)
elif opt.rms:
    optimizer = optim.RMSprop(net.parameters(), lr=opt.lr)
else:
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=0.003)

def val_test():
    print('Start val_test')
    
    for p in net.parameters():
        p.requires_grad = False
    net.eval()

    n_correct_test = 0
    with open("./test_error_f.csv",'a') as test_error:
        title_test = ['pic_name','target','res']
        writer_test = csv.DictWriter(test_error,fieldnames=title_test)
        writer_test.writeheader()

        for i, (inputs, labels, lengths) in enumerate(dloader_val):
            # inputs, labels, lengths = inputs.cuda(), labels, lengths
    #        print('labels:',labels)
            inputs = inputs.cuda()
            label = []
            for j in range(labels.size(0)):
                # if lengths.data[j, 0] == 0:
                if lengths[j, 0] == 0:
                    continue
                # label.append(labels[j, :lengths.data[j, 0]])
                label.append(labels[j, :lengths[j, 0]])
    #        print('input: ',inputs)
            preds = net(inputs)
            _, preds = preds.max(2)
    #        print('prediction in testing: ',preds.cpu().numpy())
            ress = []
            # for x in preds.data.cpu().numpy():
            for x in preds.cpu().numpy():
                if len(ress) == 0 or x != ress[-1]:
                    ress.append(x[0])
            # print(preds.data.cpu().numpy(), '\n#####', ress)
            res = ''
            for x in ress:
                res += character_str[x]
            res = res.replace('_', '')

            #        res = ''
            #        for s in preds.data.cpu().numpy():
            #            res += label_dict[s[0]]
            ##        print('res-----------------:', res)
            #        p = re.compile(r"([a-z-])(\1+)")
            #        res = p.sub(r"\1",res)
            #        res = res.replace('-','')
            target = ''
            # for s in label[0].data.cpu().numpy():
            for s in label[0].numpy(): # assuming that batchsize=1
                if s >=len(character_str) or s < 0:
                    print(s)
                    print(len(character_str))
                target += character_str[s]
    #        print(target)
            # print('res:', res, 'target:', target)
            if res == target:
                n_correct_test += 1
            else:
                wt = {'pic_name':str(dset_val.img_name_list[i]), 'target':target , 'res':res}
                writer_test.writerow(wt)

            if i >=opt.num_val:
                break

    # accuracy = n_correct / float((i + 1) * opt.val_batchsize)
    accuracy_test = n_correct_test / float(opt.num_val * opt.val_batchsize)

    print('accuray_test: %f' % (accuracy_test))
    return accuracy_test

def val_train():
    print('Start val')
    for p in net.parameters():
        p.requires_grad = False
    net.eval()

    n_correct_train = 0
 
    with open("./train_error_f.csv",'a') as train_error:
        title_train = ['pic_name','target','res']
        writer_train = csv.DictWriter(train_error,fieldnames=title_train)
        writer_train.writeheader()
        
        for i_tr, (inputs_tr, labels_tr, lengths_tr) in enumerate(dloader_train_eval):

            # inputs, labels, lengths = inputs.cuda(), labels, lengths
            inputs_tr = inputs_tr.cuda()
    #        print(labels_tr)
            label_tr = []
            for j_tr in range(labels_tr.size(0)):
                # if lengths.data[j, 0] == 0:
                if lengths_tr[j_tr, 0] == 0:
                    continue
                # label.append(labels[j, :lengths.data[j, 0]])
                label_tr.append(labels_tr[j_tr, :lengths_tr[j_tr, 0]])
    #        print('input_tr: ',inputs_tr)
            preds_tr = net(inputs_tr)
            _, preds_tr = preds_tr.max(2)
    #        print(preds_tr.cpu().numpy())
            ress_tr = []
            # for x in preds.data.cpu().numpy():
            for x_tr in preds_tr.cpu().numpy():
    #           print('ress:',ress_tr,'xtr:',x_tr) 
               if len(ress_tr) == 0 or x_tr != ress_tr[-1]:
                    ress_tr.append(x_tr[0])
            # print(preds.data.cpu().numpy(), '\n#####', ress)
            res_tr = ''
            for x_tr in ress_tr:
                res_tr += character_str[x_tr]
            res_tr = res_tr.replace('_', '')

            #        res = ''
            #        for s in preds.data.cpu().numpy():
            #            res += label_dict[s[0]]
            ##        print('res-----------------:', res)
            #        p = re.compile(r"([a-z-])(\1+)")
            #        res = p.sub(r"\1",res)
            #        res = res.replace('-','')
            target_tr = ''
            # for s in label[0].data.cpu().numpy():
    #        print('label_tr length: ',len(label_tr[0].numpy()),'character_str length: ',len(character_str))
            for s_tr in label_tr[0].numpy(): # assuming that batchsize=1
                if s_tr >=len(character_str) or s_tr < 0:
                    print(s_tr)
                    print(len(character_str))
    #            print('s_tr:',s_tr)
                target_tr += character_str[s_tr]

            # print('res:', res, 'target:', target)
            if res_tr == target_tr:
                n_correct_train += 1
            else:
                wt ={'pic_name':str(dset_train.img_name_list[i_tr]),'target': target_tr, 'res':res_tr}
                writer_train.writerow(wt)

            if i_tr >=opt.num_val:
                break

    # accuracy = n_correct / float((i + 1) * opt.val_batchsize)
    accuracy_train = n_correct_train / float(opt.num_val * opt.val_batchsize)

    print('accuray_train: %f' % (accuracy_train))
    return accuracy_train

step = 0
for epoch in range(opt.num_epoch):
    for i, (inputs, labels, lengths) in enumerate(dloader_train):
        for p in net.parameters():
            # p.requires_grad = True
            p.requires_grad_(True)

        net.train()
        # inputs, labels, lengths = Variable(inputs.cuda()), Variable(labels), Variable(lengths)
        inputs = inputs.cuda()
        label = []
        for j in range(labels.size(0)):
            if lengths.data[j, 0] == 0:
                continue
            label.append(labels[j, :lengths.data[j, 0]])
        label = torch.cat(label)
        preds = net(inputs)
        bs = inputs.size(0)
        # pred_size = torch.IntTensor([preds.size(0)] * bs)
        pred_size = torch.tensor([preds.size(0)] * bs, dtype=torch.int32)
        optimizer.zero_grad()
        loss = criterion(preds, label, pred_size, lengths[:, 0]) / opt.train_batchsize
        loss.backward()
        #        nn.utils.clip_grad_norm(net.parameters(), 10)
        optimizer.step()
        step += 1
        if i % opt.ndisplay == 0:
            print('Epoch: {}, Iter: {}, Loss:{:.4f}'.format(epoch, i, loss.data[0]))
        # if i % opt.nsave == 0:
        #     acc_test = val_test()
        #     acc_train = val_train()
        #     if (acc_test > acc_ini) or (acc_train > tr_acc_ini):
        #         acc_ini = acc_test
        #         tr_acc_ini = acc_train
        #         torch.save(net, join(opt.modeldir, 'crnn-'+str(epoch)+'-'+str(i)+'.pkl'))
        #         torch.save(net.state_dict(), join(opt.modeldir, 'params-'+str(epoch)+'-'+str(i)+'.pkl'))
        #         print('max_acc: train_{},val_{}  Epoch: {}'.format(tr_acc_ini,acc_ini,epoch))
    if epoch % opt.nsave == 0:
        acc_test = val_test()
        acc_train = val_train()
        if (acc_test > acc_ini) or (acc_train > tr_acc_ini):
            acc_ini = acc_test
            tr_acc_ini = acc_train
            torch.save(net, join(opt.modeldir, 'crnn-'+str(epoch)+'-'+str(i)+'.pkl'))
            torch.save(net.state_dict(), join(opt.modeldir, 'params-'+str(epoch)+'-'+str(i)+'.pkl'))
            print('max_acc: train_{},val_{}  Epoch: {}'.format(tr_acc_ini,acc_ini,epoch))
#torch.save(net, join(opt.modeldir, 'crnn-'+str(epoch)+'-'+str(i)+'.pkl'))
#torch.save(net.state_dict(), join(opt.modeldir, 'params-'+str(epoch)+'-'+str(i)+'.pkl'))
