"""
UNet
opturations and data loading code for Kaggle Data Science Bowl 2018
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from utils import Option
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split


"""Transforms:
Data augmentation
"""
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size, train=True):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.train = train

    def __call__(self, sample):
        if self.train:
            image, mask, img_id, height, width = sample['image'], sample['mask'], sample['img_id'], sample['height'],sample['width']

            if isinstance(self.output_size, int):
                new_h = new_w = self.output_size
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            # resize the image,
            # preserve_range means not normalize the image when resize
            img = transform.resize(image, (new_h, new_w), preserve_range=True, mode='constant')
            mask = transform.resize(mask, (new_h, new_w), preserve_range=True, mode='constant')
            return {'image': img, 'mask': mask, 'img_id': img_id, 'height':height, 'width':width}
        else:
            image, img_id, height,width = sample['image'], sample['img_id'], sample['height'],sample['width']
            if isinstance(self.output_size, int):
                new_h = new_w = self.output_size
            else:
                new_h, new_w = self.output_size

            new_h, new_w = int(new_h), int(new_w)

            # resize the image,
            # preserve_range means not normalize the image when resize
            img = transform.resize(image, (new_h, new_w), preserve_range=True, mode='constant')
            return {'image': img, 'height': height,'width':width, 'img_id':img_id}

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, mask, img_id, height, width = sample['image'], sample['mask'], sample['img_id'], sample['height'], \
                                             sample['width']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        if h - new_h > 0 and w - new_w > 0:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
        else:
            top = 0
            left = 0

        image = image[top: top + new_h,
                      left: left + new_w]

        mask = mask[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'mask': mask, 'img_id':img_id, 'height':height, 'width':width}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, train=True):
        self.train = train

    def __call__(self, sample):
        if self.train:
            # if sample.keys
            image, mask, img_id, height, width = sample['image'], sample['mask'], sample['img_id'], sample['height'],sample['width']

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
            image = image.transpose((2, 0, 1))
            mask = mask.transpose((2, 0, 1))
            return {'image': torch.from_numpy(image.astype(np.uint8)),
                    'mask': torch.from_numpy(mask.astype(np.uint8)),
                    'img_id': img_id,
                    'height':height,
                    'width':width}
        else:
            image, height, width, img_id = sample['image'], sample['height'],sample['width'], sample['img_id']
            image = image.transpose((2, 0, 1))
            return {'image': torch.from_numpy(image.astype(np.uint8)),
                    'height': height,
                    'width': width,
                    'img_id': img_id}

# Helper function to show a batch
def show_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, masks_batch = sample_batched['image'].numpy().astype(np.uint8), sample_batched['mask'].numpy().astype(np.bool)
    batch_size = len(images_batch)
    for i in range(batch_size):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.tight_layout()
        plt.imshow(images_batch[i].transpose((1, 2, 0)))
        plt.subplot(1, 2, 2)
        plt.tight_layout()
        plt.imshow(np.squeeze(masks_batch[i].transpose((1, 2, 0))))

# Load Data Science Bowl 2018 training dataset
class DSB2018Dataset(Dataset):
    def __init__(self, root_dir, img_id, train=True, transform=None):
        """
        Args:
        :param root_dir (string): Directory with all the images
        :param img_id (list): lists of image id
        :param train: if equals true, then read training set, so the output is image, mask and imgId
                      if equals false, then read testing set, so the output is image and imgId
        :param transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.img_id = img_id
        self.train = train
        self.transform = transform
        self.opt = Option()

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        if self.train:
            img_dir = os.path.join(self.root_dir, self.img_id[idx], 'image.png')
            mask_dir = os.path.join(self.root_dir, self.img_id[idx], 'mask.png')
            img = io.imread(img_dir).astype(np.uint8)
            mask = io.imread(mask_dir, as_grey=True).astype(np.bool)
            mask = np.expand_dims(mask, axis=-1)
            sample = {'image':img, 'mask':mask, 'img_id':self.img_id[idx], "height":img.shape[0], "width":img.shape[1]}

        else:
            img_dir = os.path.join(self.root_dir, self.img_id[idx], 'image.png')
            img = io.imread(img_dir).astype(np.uint8)
            # size = (img.shape[0],img.shape[1])  # (Height, Weidth)
            sample = {'image': img, 'img_id': self.img_id[idx], "height":img.shape[0], "width":img.shape[1]}

        if self.transform:
            sample = self.transform(sample)

        return sample

def get_train_valid_loader(root_dir, batch_size=16, split=True,
                           shuffle=False, num_workers=4, val_ratio=0.1, pin_memory=False):

    """Utility function for loading and returning training and validation Dataloader
    :param root_dir: the root directory of data set
    :param batch_size: batch size of training and validation set
    :param split: if split data set to training set and validation set
    :param shuffle: if shuffle the image in training and validation set
    :param num_workers: number of workers loading the data, when using CUDA, set to 1
    :param val_ratio: ratio of validation set size
    :param pin_memory: store data in CPU pin buffer rather than memory. when using CUDA, set to True
    :return:
        if split the data set then returns:
        - train_loader: Dataloader for training
        - valid_loader: Dataloader for validation
        else returns:
        - dataloader: Dataloader of all the data set
    """
    img_id = os.listdir(root_dir)
    if split:
        train_id, val_id = train_test_split(img_id, test_size=val_ratio)

        train_transformed_dataset = DSB2018Dataset(root_dir=root_dir,
                                                   img_id=train_id,
                                                   train=True,
                                                   transform=transforms.Compose([
                                                      RandomCrop(256),
                                                      Rescale(256),
                                                      ToTensor()
                                                   ]))
        val_transformed_dataset = DSB2018Dataset(root_dir=root_dir,
                                                 img_id=val_id,
                                                 train=True,
                                                 transform=transforms.Compose([
                                                      # RandomCrop(256),    # for validation set, do not use augmentation
                                                      Rescale(256),
                                                      ToTensor()
                                                   ]))


        train_loader = DataLoader(train_transformed_dataset,batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_transformed_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return (train_loader, val_loader)
    else:
        transformed_dataset = DSB2018Dataset(root_dir=root_dir,
                                             img_id=img_id,
                                             train=True,
                                             transform=transforms.Compose([
                                                 RandomCrop(256),
                                                 Rescale(256),
                                                 ToTensor()
                                             ]))
        dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                                shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        return dataloader

def get_test_loader(root_dir, batch_size=16, shuffle=False, num_workers=4, pin_memory=False):

    """Utility function for loading and returning training and validation Dataloader
    :param root_dir: the root directory of data set
    :param batch_size: batch size of training and validation set
    :param shuffle: if shuffle the image in training and validation set
    :param num_workers: number of workers loading the data, when using CUDA, set to 1
    :param pin_memory: store data in CPU pin buffer rather than memory. when using CUDA, set to True
    :return:
        - testloader: Dataloader of all the test set
    """
    img_id = os.listdir(root_dir)
    transformed_dataset = DSB2018Dataset(root_dir=root_dir,
                                         img_id=img_id,
                                         train=False,
                                         transform=transforms.Compose([
                                             # RandomCrop(256),
                                             Rescale(256, train=False),
                                             ToTensor(train=False)
                                         ]))
    testloader = DataLoader(transformed_dataset, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
    return testloader

if __name__ == '__main__':
    opt = Option()
    trainloader, val_loader = get_train_valid_loader(opt.root_dir, batch_size=opt.batch_size,
                                                        split=True, shuffle=opt.shuffle,
                                                        num_workers=opt.num_workers,
                                                        val_ratio=0.1, pin_memory=opt.pin_memory)

    for i_batch, sample_batched in enumerate(val_loader):
        print(i_batch, sample_batched['image'].size(), sample_batched['mask'].size())
        show_batch(sample_batched)
        plt.show()

    # testloader = get_test_loader(opt.test_dir, batch_size=opt.batch_size,shuffle=opt.shuffle,
    #                                 num_workers=opt.num_workers, pin_memory=opt.pin_memory)
    #
    # for i_batch, sample_batched in enumerate(testloader):
    #     # print(i_batch, sample_batched['image'].size(), sample_batched['img_size'])
    #     plt.imshow(np.squeeze(sample_batched['image'][0].cpu().numpy().transpose((1, 2, 0))))
    #     plt.show()
