#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by: ludi
# Created on: 2020/1/8


import numpy as np
import struct
import os
from PIL import Image
import torch


def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'   #'>IIII'是说使用大端法读取4个unsinged int32
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    print("offset: ",offset)
    fmt_image = '>' + str(image_size) + 'B'   # '>784B'的意思就是用大端法读取784个unsigned byte
    images = np.empty((num_images, num_rows*num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows*num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def get_mnist(mnist_path, resize=None):
    train_images_file = os.path.join(mnist_path, 'train-images-idx3-ubyte')
    train_labels_file = os.path.join(mnist_path, 'train-labels-idx1-ubyte')
    test_images_file = os.path.join(mnist_path,'t10k-images-idx3-ubyte')
    test_labels_file = os.path.join(mnist_path, 't10k-labels-idx1-ubyte')
    if os.path.exists('./mnist_data.npz'):
        r = np.load('./mnist_data.npz')
        train_images, train_labels, test_images, test_labels = \
            r['train_images'], r['train_labels'], r['test_images'], r['test_labels']
        # return r['train_images'], r['train_labels'], r['test_images'], r['test_labels']
    else:
        test_labels = decode_idx1_ubyte(test_labels_file)
        test_images = decode_idx3_ubyte(test_images_file)
        train_labels = decode_idx1_ubyte(train_labels_file)
        train_images = decode_idx3_ubyte(train_images_file)
        # np.savez('./mnist_data.npz', train_images=train_images, train_labels=train_labels,
        # test_images=test_images, test_labels=test_labels)
    return train_images, train_labels, test_images, test_labels


class BatchIterator():

    def __init__(self, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __call__(self, inputs, targets):
        starts = np.arange(0, len(inputs), self.batch_size)
        if self.shuffle:
            idx = np.arange(len(inputs))
            np.random.shuffle(idx)
            inputs = inputs[idx]
            targets = targets[idx]

        for start in starts:
            end = start + self.batch_size
            batch_inputs = inputs[start: end]
            batch_targets = targets[start: end]
            yield (batch_inputs, batch_targets)




def img_resize(data, re=None):
    ret_list = []
    if re:
        for img in data:
            img = Image.fromarray(np.array(img[0]))
            im = torch.tensor(np.array(img.resize((re, re)))).view(-1, 1, re, re)
            ret_list.append(im)
        return torch.cat(ret_list, 0)
    else:
        return data


def select_two_labels(data, label_sel = (0, 1)):
    train_image, train_label, test_image, test_label = data[0], data[1], data[2], data[3]
    train_image_0 = train_image[train_label==label_sel[0]]
    train_image_1 = train_image[train_label==label_sel[1]]
    train_label_0 = np.ones(train_image_0.shape[0]) * label_sel[0]
    train_label_1 = np.ones(train_image_1.shape[0]) * label_sel[1]
    train_image_sel = np.concatenate((train_image_0, train_image_1), axis=0)
    train_label_sel = np.concatenate((train_label_0, train_label_1), axis=0)
    idx = list(range(train_image_sel.shape[0]))
    np.random.shuffle(idx)
    train_image_sel = train_image_sel[idx]
    train_label_sel = train_label_sel[idx]

    test_image_0 = test_image[test_label==label_sel[0]]
    test_image_1 = test_image[test_label==label_sel[1]]
    test_label_0 = np.ones(test_image_0.shape[0]) * label_sel[0]
    test_label_1 = np.ones(test_image_1.shape[0]) * label_sel[1]
    test_image_sel = np.concatenate((test_image_0, test_image_1), axis=0)
    test_label_sel = np.concatenate((test_label_0, test_label_1), axis=0)
    idx = list(range(test_image_sel.shape[0]))
    np.random.shuffle(idx)
    test_image_sel = test_image_sel[idx]
    test_label_sel = test_label_sel[idx]
    return train_image_sel, train_label_sel, test_image_sel, test_label_sel

