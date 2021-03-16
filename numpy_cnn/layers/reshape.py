#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by: ludi
# Created on: 2020/1/8

import numpy as np
from .layer import Layer


class Reshape(Layer):
    def __init__(self, *out_shape):
        super().__init__("Reshape")
        self.out_shape = out_shape
        self.in_shape = None

    def forward(self, inputs):
        self.in_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], *self.out_shape)

    def backward(self, grad):
        return grad.reshape(self.in_shape)
