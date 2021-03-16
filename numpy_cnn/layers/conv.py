#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by: ludi
# Created on: 2020/1/8


import numpy as np
from .layer import Layer, xavier_uniform, Tensor


class Conv2d(Layer):
    def __init__(self, in_channels, out_channels, k_size=(3, 3), stride=(1, 1), padding=None, requires_grad=True):
        super().__init__("Conv2d")
        self.k_shape = (k_size[0], k_size[1], in_channels, out_channels)
        w = xavier_uniform(self.k_shape)
        b = np.zeros(out_channels)
        self.params = {
            "w": Tensor(w, requires_grad),
            "b": Tensor(b, requires_grad)}
        self.stride = stride
        self.padding = padding
        self.inputs = None
        self.in_shape, self.col, self.w = None, None, None
        self.requires_grad = requires_grad

    def __repr__(self):
        return self.name + ': ' + str(self.k_shape)

    def forward(self, inputs):
        x = inputs
        if self.padding:
            p_h, p_w = self.padding
            x = np.pad(inputs, ((0, 0), (p_h, p_h), (p_w, p_w), (0, 0)), 'constant')
        k_h, k_w, in_c, out_c = self.k_shape
        s_h, s_w = self.stride
        col = im2col(x, k_h, k_w, s_h, s_w)
        w = self.params['w'].data.reshape(-1, out_c)
        z = np.matmul(col, w)
        batch_sz, in_h, in_w, _ = x.shape
        out_h = (in_h - k_h) // s_h + 1
        out_w = (in_w - k_w) // s_w + 1
        z = z.reshape((batch_sz, z.shape[0] // batch_sz, out_c))
        z = z.reshape(batch_sz, out_h, out_w, out_c)
        z += self.params['b'].data
        self.in_shape, self.col, self.w = x.shape, col, w
        return z

    def backward(self, grad):
        k_h, k_w, in_c, out_c = self.k_shape
        s_h, s_w = self.stride
        batch_sz, in_h, in_w, in_c = self.in_shape
        if self.padding:
            p_h, p_w = self.padding
        else:
            p_h, p_w = 0, 0
        flat_grad = grad.reshape((-1, out_c))
        d_w = np.matmul(self.col.T, flat_grad)
        self.params['w'].grad = d_w.reshape(self.k_shape)
        self.params['b'].grad = np.sum(flat_grad, axis=0)
        # print(grad.shape)
        # print(self.params['w'].T.shape)
        d_x = np.matmul(grad, self.w.T)
        d_in = np.zeros(shape=self.in_shape)
        for i, r in enumerate(range(0, in_h - k_h + 1, s_h)):
            for j, c in enumerate(range(0, in_w - k_w + 1, s_w)):
                patch = d_x[:, i, j, :]
                patch = patch.reshape((batch_sz, k_h, k_w, in_c))
                d_in[:, r:r + k_h, c:c + k_w, :] += patch
        d_in = d_in[:, p_h : in_h - p_h, p_w : in_w - p_w, :]
        return d_in


def im2col(img, k_h, k_w, s_h, s_w):
    """Transform padded image into column matrix.
    :param img: padded inputs of shape (B, in_h, in_w, in_c)
    :param k_h: kernel height
    :param k_w: kernel width
    :param s_h: stride height
    :param s_w: stride width
    :return col: column matrix of shape (B*out_h*out_w, k_h*k_h*inc)
    """
    batch_sz, h, w, in_c = img.shape
    # calculate result feature map size
    out_h = (h - k_h) // s_h + 1
    out_w = (w - k_w) // s_w + 1
    # allocate space for column matrix
    col = np.empty((batch_sz * out_h * out_w, k_h * k_w * in_c))
    # fill in the column matrix
    batch_span = out_w * out_h
    for r in range(out_h):
        r_start = r * s_h
        matrix_r = r * out_w
        for c in range(out_w):
            c_start = c * s_w
            patch = img[:, r_start: r_start+k_h, c_start: c_start+k_w, :]
            patch = patch.reshape(batch_sz, -1)
            col[matrix_r+c::batch_span, :] = patch
    return col

