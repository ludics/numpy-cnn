#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by: ludi
# Created on: 2020/1/8

from abc import ABCMeta, abstractmethod
import numpy as np


class Layer(metaclass=ABCMeta):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    @abstractmethod
    def forward(self, *args):
        pass

    @abstractmethod
    def backward(self, *args):
        pass


class Tensor(object):
    def __init__(self, data, requires_grad=True, skip_decay=False):
        self.data = data
        self.grad = None
        self.skip_decay = skip_decay
        self.requires_grad = requires_grad

    @property
    def T(self):
        return self.data.T


def xavier_uniform(shape):
    num_in = shape[0] if len(shape) == 2 else np.prod(shape[1:])
    num_out = shape[1] if len(shape) == 2 else shape[0]
    a = np.sqrt(6.0 / (num_in + num_out))
    return np.random.uniform(low=-a, high=a, size=shape)
