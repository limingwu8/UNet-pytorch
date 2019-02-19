"""
UNet
Train Unet model
"""
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from dataset import get_train_valid_loader, get_test_loader
from model import UNet2
from utils import Option, encode_and_save, compute_iou
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt

def train(model, train_loader, opt, criterion, epoch):
    model.train()
    num_batches = 0
    avg_loss = 0
    with open('logs.txt', 'a') as file:
        for batch_idx, sample_batched in enumerate(train_loader):
            data = sample_batched['image']
            target = sample_batched['mask']
            data, target = Variable(data.type(opt.dtype)), Variable(target.type(opt.dtype))
            optimizer.zero_grad()
            output = model(data)
            # output = (output > 0.5).type(opt.dtype)	# use more gpu memory, also, loss does not change if use this line
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            avg_loss += loss.data[0]
            num_batches += 1
        avg_loss /= num_batches
        # avg_loss /= len(train_loader.dataset)
        print('epoch: ' + str(epoch) + ', train loss: ' + str(avg_loss))
        file.write('epoch: ' + str(epoch) + ', train loss: ' + str(avg_loss) + '\n')

def val(model, val_loader, opt, criterion, epoch):
    model.eval()
    num_batches = 0
    avg_loss = 0
    with open('logs.txt', 'a') as file:
        for batch_idx, sample_batched in enumerate(val_loader):
            data = sample_batched['image']
            target = sample_batched['mask']
            data, target = Variable(data.type(opt.dtype)), Variable(target.type(opt.dtype))
            output = model.forward(data)
            # output = (output > 0.5).type(opt.dtype)	# use more gpu memory, also, loss does not change if use this line
            loss = criterion(output, target)
            avg_loss += loss.data[0]
            num_batches += 1
        avg_loss /= num_batches
        # avg_loss /= len(val_loader.dataset)

        print('epoch: ' + str(epoch) + ', validation loss: ' + str(avg_loss))
        file.write('epoch: ' + str(epoch) + ', validation loss: ' + str(avg_loss) + '\n')

# train and validation
def run(model, train_loader, val_loader, opt, criterion):
    for epoch in range(1, opt.epochs):
        train(model, train_loader, opt, criterion, epoch)
        val(model, val_loader, opt, criterion, epoch)

# only train
def run_train(model, train_loader, opt, criterion):
    for epoch in range(1, opt.epochs):
        train(model, train_loader, opt, criterion, epoch)

# make prediction
def run_test(model, test_loader, opt):
    """
    predict the masks on testing set
    :param model: trained model
    :param test_loader: testing set
    :param opt: configurations
    :return:
        - predictions: list, for each elements, numpy array (Width, Height)
        - img_ids: list, for each elements, an image id string
    """
    model.eval()
    predictions = []
    img_ids = []
    for batch_idx, sample_batched in enumerate(test_loader):
        data, img_id, height, width = sample_batched['image'], sample_batched['img_id'], sample_batched['height'], sample_batched['width']
        data = Variable(data.type(opt.dtype))
        output = model.forward(data)
        # output = (output > 0.5)
        output = output.data.cpu().numpy()
        output = output.transpose((0, 2, 3, 1))    # transpose to (B,H,W,C)
        for i in range(0,output.shape[0]):
            pred_mask = np.squeeze(output[i])
            id = img_id[i]
            h = height[i]
            w = width[i]
            # in p219 the w and h above is int
            # in local the w and h above is LongTensor
            if not isinstance(h, int):
                h = h.cpu().numpy()
                w = w.cpu().numpy()
            pred_mask = resize(pred_mask, (h, w), mode='constant')
            pred_mask = (pred_mask > 0.5)
            predictions.append(pred_mask)
            img_ids.append(id)

    return predictions, img_ids

if __name__ == '__main__':
    """Train Unet model"""
    opt = Option()
    model = UNet2(input_channels=3, nclasses=1)
    if opt.is_train:
        # split all data to train and validation, set split = True
        train_loader, val_loader = get_train_valid_loader(opt.root_dir, batch_size=opt.batch_size,
                                              split=True, shuffle=opt.shuffle,
                                              num_workers=opt.num_workers,
                                              val_ratio=0.1, pin_memory=opt.pin_memory)

        # load all data for training
        # train_loader = get_train_valid_loader(opt.root_dir, batch_size=opt.batch_size,
        #                                       split=False, shuffle=opt.shuffle,
        #                                       num_workers=opt.num_workers,
        #                                       val_ratio=0.1, pin_memory=opt.pin_memory)
        if opt.n_gpu > 1:
            model = nn.DataParallel(model)
        if opt.is_cuda:
            model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
        criterion = nn.BCELoss().cuda()
        # start to run a training
        run_train(model, train_loader, opt, criterion)
        # make prediction on validation set
        predictions, img_ids = run_test(model, val_loader, opt)
        # compute IOU between prediction and ground truth masks
        compute_iou(predictions, img_ids, val_loader)
        # SAVE model
        if opt.save_model:
            torch.save(model.state_dict(), os.path.join(opt.checkpoint_dir, 'model-01.pt'))
    else:
        # load testing data for making predictions
        test_loader = get_test_loader(opt.test_dir, batch_size=opt.batch_size, shuffle=opt.shuffle,
                                      num_workers=opt.num_workers, pin_memory=opt.pin_memory)
        # load the model and run test
        model.load_state_dict(torch.load(os.path.join(opt.checkpoint_dir, 'model-01.pt')))
        if opt.n_gpu > 1:
            model = nn.DataParallel(model)
        if opt.is_cuda:
            model = model.cuda()
        predictions, img_ids = run_test(model, test_loader, opt)
        # run length encoding and save as csv
        encode_and_save(predictions, img_ids)
