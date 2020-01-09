#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by: ludi
# Created on: 2020/1/8


import numpy as np
from .layer import Layer


class Conv2d(Layer):
    def __init__(self, shape, out_channels, ksize=3):
        super().__init__("Conv2d")
        pass

    def forward(self, *args):
        pass

    def backward(self, *args):
        pass
