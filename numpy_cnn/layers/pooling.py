#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by: ludi
# Created on: 2020/1/8


import numpy as np
from .layer import Layer

class MaxPool2d(Layer):
    def __init__(self, num_in, num_out, requires_grad=True, **kwargs):
        super().__init__("MaxPool2d")
        w = np.random.randn(*(num_in, num_out)) * (2 / num_in**0.5)
        b = np.zeros(num_out)
        self.params = {
            "w": Tensor(w),
            "b": Tensor(b)}
        self.inputs = None
        self.requires_grad = requires_grad

    def forward(self, inputs):
        self.inputs = inputs
        out = np.dot(inputs, self.params['w'].data) + self.params['b'].data
        return out

    def backward(self, grad):
        if self.requires_grad:
            self.params['w'].grad = np.dot(self.inputs.T, grad)
            self.params['b'].grad = np.sum(grad, axis=0)
        return np.dot(grad, self.params['w'].T)


