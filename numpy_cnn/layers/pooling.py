#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by: ludi
# Created on: 2020/1/8


import numpy as np
from .layer import Layer


class MaxPool2d(Layer):
    def __init__(self, kernel_size, stride):
        super().__init__("MaxPool2d")



    def forward(self, inputs):
        pass

    def backward(self, grad):
        pass


