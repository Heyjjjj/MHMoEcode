import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import argparse

from torch.utils.data import DataLoader
from MHMoE_Net import MHMoE_Net
from nets.unet_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import LossHistory, EvalCallback
from utils.dataloader_public import UnetDataset, unet_dataset_collate
from utils.utils import download_weights, show_config
from utils.utils_fit_public import fit_one_epoch




if __name__ == "__main__":
    cuda = True
    num_classes = 20
    backbone = "resnet50"
    pretrained = False
    model_path = ''

    input_shape = (448, 448)
    init_epoch = 0
    freeze_epoch = 0
    unFreeze_epoch = 300
    freeze_batch_size = 4
    unfreeze_batch_size = 4
    freeze_train = False
    init_lr = 1e-3
    min_lr = 1e-3
    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = 'cos'
    eval_flag = True
    eval_period = 1
    num_workers = 6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = './results/model/'
    data_path = './Data_Annotations/'
    image_path = '/Data_Annotations/'
    model_path = ''
    loss_fuc = "Diceloss"

    parser = argparse.ArgumentParser()

    parser.add_argument('--root_path', type=str,
                        default='', help='root dir for data')
    parser.add_argument('--volume_path', type=str,
                    default='', help='root dir for validation volume data')  # for acdc volume_path=root_dir
    parser.add_argument('--dataset', type=str,
                        default='Synapse', help='experiment_name')  # 突触
    parser.add_argument('--list_dir', type=str,
                        default='./lists/lists_Synapse', help='list dir')
    parser.add_argument('--num_classes', type=int,
                        default=9, help='output channel of network')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--max_epochs', type=int,
                        default=150, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=1, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--img_size', type=int,
                        default=448, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--n_skip', type=int,
                        default=3, help='using number of skip-connect, default is num')
    parser.add_argument('--vit_name', type=str,
                        default='R50-ViT-B_16', help='select one vit model')
    parser.add_argument('--vit_patches_size', type=int,
                        default=16, help='vit_patches_size, default is 16')
    args = parser.parse_args()
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    print("##########################################"+backbone+"#######################################################")
    model = Dilateformer(img_size=448,in_chans=3,embed_dim=96,att_mode = [True, True, False, False],num_classes=num_classes).cuda()


    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M')
    save_dir = os.path.join(save_dir, "ModelLabel_train_2_public_"+time_str)
    loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    model_train = model.train()

    if cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    with open(os.path.join(data_path, "train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(data_path, "test.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    show_config(
        num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape, \
        Init_Epoch=init_epoch, Freeze_Epoch=freeze_epoch, UnFreeze_Epoch=unFreeze_epoch,
        Freeze_batch_size=freeze_batch_size, Unfreeze_batch_size=unfreeze_batch_size, Freeze_Train=freeze_train, \
        Init_lr=init_lr, Min_lr=min_lr, optimizer_type=optimizer_type, momentum=momentum,
        lr_decay_type=lr_decay_type, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val, \
        loss_fuc=loss_fuc
    )

    UnFreeze_flag = False
    batch_size = freeze_batch_size if freeze_train else unfreeze_batch_size
    # -------------------------------------------------------------------#
    #   判断当前batch_size，自适应调整学习率
    # -------------------------------------------------------------------#
    nbs = 16
    lr_limit_max = 1e-4 if optimizer_type == 'adam' else 1e-1
    lr_limit_min = 1e-4 if optimizer_type == 'adam' else 5e-4
    init_lr_fit = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
    min_lr_fit = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # ---------------------------------------#
    #   根据optimizer_type选择优化器
    # ---------------------------------------#
    optimizer = {
        'adam': optim.Adam(model.parameters(), init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
        'sgd': optim.SGD(model.parameters(), init_lr_fit, momentum=momentum, nesterov=True,
                         weight_decay=weight_decay)
    }[optimizer_type]

    # ---------------------------------------#
    #   获得学习率下降的公式
    # ---------------------------------------#
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, init_lr_fit, min_lr_fit, unFreeze_epoch)

    # ---------------------------------------#
    #   判断每一个世代的长度
    # ---------------------------------------#
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size



    train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, image_path)
    val_dataset = UnetDataset(val_lines, input_shape, num_classes, False, image_path)
    train_sampler = None
    val_sampler = None
    shuffle = True
    eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, data_path, save_dir, cuda,
                                 eval_flag=eval_flag, period=eval_period)

    # ---------------------------------------#
    #   开始模型训练
    # ---------------------------------------#
    no_improve_count = 0  # 早停
    for epoch in range(init_epoch, unFreeze_epoch):
        # if epoch>200:
        #     model.freeze()
        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,prefetch_factor=4,
                         pin_memory=True, drop_last=True, collate_fn=unet_dataset_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True, drop_last=True, collate_fn=unet_dataset_collate, sampler=val_sampler)
        no_improve_count = fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                                         epoch_step, epoch_step_val, gen, gen_val, unFreeze_epoch, loss_fuc,
                                         num_classes, save_dir, no_improve_count)

