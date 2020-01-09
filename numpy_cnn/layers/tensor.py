class Tensor(object):
    def __init__(self, data):
        self.data = data
        self.grad = None
        # self.requires_grad = requires_grad

    @property
    def T(self):
        return self.data.T