from torchvision import transforms
import torch
import torch.nn as nn
import os
import argparse
import cv2
import numpy as np
import csv
from ds import TestDataset
from ds import get_test_dataset
from crnn import CRNN
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', help='path to test dataset directory', default='../test_dataset')
parser.add_argument('--output_file', help='path to test result file', default='../test_dataset')
parser.add_argument('--gpus', '-g', type=str, default='0', help='which gpu(s) it could use')
parser.add_argument('--modeldir', help='path to save model', default='./saved_models_all')
parser.add_argument('--modelname0', help='model name', default='params-1635-112.pkl')
parser.add_argument('--modelname1', help='model name', default='0_params-551.pkl')
parser.add_argument('--modelname2', help='model name', default='3new_params-3.pkl')
parser.add_argument('--modelname3', help='model name', default='params-2167-112.pkl')
parser.add_argument('--modelname4', help='model name', default='_lby_params-8-95.pkl')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--test_batchsize', type=int, default=1, help='test batch size')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
# parser.add_argument('--char_len', type=int, default=1824, help='number of characters')
parser.add_argument('--char_dir', type=str, default='../all_characters_all.txt', help='path of all_characters.txt')

parser.add_argument('--nrnn', type=int, default=2, help='the number of layers of rnn(sru)')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout applied between RNN layers(sru), default=0.0')
parser.add_argument('--variational_dropout', type=float, default=0.0,
                                        help='variational dropout applied on linear transformation(sru), default=0.0')
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus

if not os.path.exists(opt.modeldir):
        os.mkdir(opt.modeldir)

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dset_test = get_test_dataset(opt.data_dir, transform=transform, size=(256, 48), max_length=None)
dloader_test = torch.utils.data.DataLoader(dset_test, shuffle=False, batch_size=opt.test_batchsize,num_workers=int(opt.workers))
dset_test384 = get_test_dataset(opt.data_dir, transform=transform, size=(384, 48), max_length=None)
dloader_test384 = torch.utils.data.DataLoader(dset_test384, shuffle=False, batch_size=opt.test_batchsize,num_workers=int(opt.workers))
character_str=open(opt.char_dir,'r').read()    
print('character  ',character_str[13])

net_t_list = []

net_t_list.append(CRNN(48, 1, len(character_str) - 1, 256, opt.nrnn, 0.5, opt.variational_dropout, leakyRelu=True))
net_t_list.append(CRNN(48, 1, len(character_str) - 1, 256, opt.nrnn, 0.5, opt.variational_dropout, RRelu=True))
net_t_list.append(CRNN(48, 1, len(character_str) - 1, 256, opt.nrnn, 0.7, opt.variational_dropout, RRelu=True))
net_t_list.append(CRNN(48, 1, len(character_str) - 1, 256, opt.nrnn, 0.5, opt.variational_dropout, leakyRelu=True))
net_t_list.append(CRNN(48, 1, len(character_str) - 1, 256, opt.nrnn, 0.5, opt.variational_dropout, leakyRelu=True))
# net_t_list.append(CRNN(48, 1, len(character_str) - 1, 256, opt.nrnn, opt.dropout, opt.variational_dropout, leakyRelu=True))
# net_t384 = CRNN(48, 1, len(character_str) - 1, 256, opt.nrnn, 0.7, opt.variational_dropout, RRelu=True)

for net_t in net_t_list:
    print(net_t)

# print(net_t384)

print('Loading model from', os.path.join(opt.modeldir , opt.modelname0))
net_t_list[0].load_state_dict(torch.load(os.path.join(opt.modeldir , opt.modelname0)))

print('Loading model from', os.path.join(opt.modeldir , opt.modelname1))
net_t_list[1].load_state_dict(torch.load(os.path.join(opt.modeldir , opt.modelname1))) 

print('Loading model from', os.path.join(opt.modeldir , opt.modelname2))
net_t_list[2].load_state_dict(torch.load(os.path.join(opt.modeldir , opt.modelname2))) 

print('Loading model from', os.path.join(opt.modeldir , opt.modelname3))
net_t_list[3].load_state_dict(torch.load(os.path.join(opt.modeldir , opt.modelname3))) 

print('Loading model from', os.path.join(opt.modeldir , opt.modelname4))
net_t_list[4].load_state_dict(torch.load(os.path.join(opt.modeldir , opt.modelname4))) 

                      

if opt.ngpu > 1:
        # print("Let's use", torch.cuda.device_count(), "GPUs!")
        net_t_1 = nn.DataParallel(net_t_1, device_ids=range(opt.ngpu))

for net_t in net_t_list:
    net_t.cuda()

# net_t384.cuda()

print('Start testing')

for net_t in net_t_list:
    for p in net_t.parameters():
        p.requires_grad = False
    net_t.eval()

# for p in net_t384.parameters():
#     p.requires_grad = False
# net_t384.eval()

op_file = os.path.join(opt.output_file,"test_result_f.csv")
with open(op_file,'a') as test_result:

#with open("./test_result_f.csv",'a') as test_result:
    title = ['name','content']
    test = csv.DictWriter(test_result,fieldnames=title)
    test.writeheader()

    for i_test, (inputs_test) in enumerate(dloader_test):

            inputs_test = inputs_test.cuda()
            # inputs_test384 = inputs_test384.cuda()
#    print(inputs_test)
            preds_test_list = []
            for net_t in net_t_list:
                preds_test_list.append(net_t(inputs_test))

            # print(preds_test.shape)
            preds_test = sum(preds_test_list)
            # preds_test = preds_test + net_t384(inputs_test384)
            _, preds_test = preds_test.max(2)
            # print(preds_test.shape)
            # os.system('pause')
#        print('prediction in testing: ',preds.cpu().numpy())
            ress_test = []
        # for x in preds.data.cpu().numpy():
            for x_test in preds_test.cpu().numpy():
                    if len(ress_test) == 0 or x_test != ress_test[-1]:
                            ress_test.append(x_test[0])
        # print(preds.data.cpu().numpy(), '\n#####', ress)
            res_test = ''
            for x_test in ress_test:
                    res_test += character_str[x_test]
            res_test = res_test.replace('_', '')
            t_r = {'name':str(dset_test.img_name_list[i_test]),'content':str(res_test)}
            test.writerow(t_r)

