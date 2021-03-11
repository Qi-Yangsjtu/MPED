import torch as th
import numpy as np
from AE_model import AutoEncoder
from Datalode import snc_category_to_synth_id, PointCloudDataset_train
from MPED import MPED_VALUE
from torch.utils.data import DataLoader
import os.path as osp
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
EPOCH=50
BTACH_SIZE=50
LR = 0.0005
# database path, please update top_in_dir, for example:  top_in_dir = '/home/data/shape_net_core_uniform_samples_2048/'
top_in_dir = '/home/data/shape_net_core_uniform_samples_2048/'
class_name = 'chair'
syn_id = snc_category_to_synth_id()[class_name]
class_dir = osp.join(top_in_dir, syn_id)

# model path, please update train_dir
train_dir = '/home/qiyang/MPED_test/'
if not os.path.exists(train_dir):
    os.makedirs(train_dir);

net2 = AutoEncoder()
net2= net2.cuda()

data = PointCloudDataset_train(class_dir)
optimizer = th.optim.Adam(net2.parameters(),lr=LR)
loss_func1 = MPED_VALUE
dataloader = DataLoader(data, batch_size=BTACH_SIZE, shuffle = True, num_workers=16, drop_last = True)

for epoch in range(EPOCH):
    for pc_tuple in enumerate(dataloader):
        pc = pc_tuple[1]
        pc = pc.cuda()
        encoded, decoded = net2(pc)
        pc_ori = pc.permute(0, 2, 1)
        pc_rec = decoded.permute(0, 2, 1)
        loss1 = loss_func1(pc_ori, pc_rec)
        loss = loss1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 5 == 0:
        data_cpu = th.Tensor.cpu(loss.data)
        print('Epoch: ', epoch, '| Train loss: %.6f' % data_cpu.numpy())
        model_name = train_dir + 'parameter_' + str(epoch) + '.pkl'
        th.save(net2.state_dict(), model_name)