#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by: ludi
# Created on: 2020/1/8


import numpy as np
from .layer import Layer


class MaxPool2d(Layer):
    def __init__(self, k_size, stride):
        super().__init__("MaxPool2d")
        self.k_size = k_size
        self.stride = stride
        self.mask = None

    def forward(self, inputs):
        n, h, w, c = inputs.shape
        s_h, s_w = self.stride
        k_h, k_w = self.k_size
        out_h, out_w = h // s_h, w // s_w
        out = inputs.reshape(n, out_h, k_h, out_w, k_w, c)
        out = out.max(axis=(2, 4))
        self.mask = out.repeat(k_h, axis=1).repeat(k_w, axis=2) != inputs
        return out

    def backward(self, grad):
        k_h, k_w = self.k_size
        grad = grad.repeat(k_h, axis=1).repeat(k_w, axis=2)
        grad[self.mask] = 0
        return grad


