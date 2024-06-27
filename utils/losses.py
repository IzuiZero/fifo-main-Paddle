import paddle
import paddle.nn.functional as F
import paddle.nn as nn

class CrossEntropy2d(nn.Layer):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.stop_gradient
        assert predict.ndim == 4
        assert target.ndim == 3
        n, c, h, w = predict.shape
        n1, h1, w1 = target.shape
        target_mask = paddle.logical_and(target >= 0, target != self.ignore_label)
        target = paddle.masked_select(target, target_mask)
        if not target.shape:
            return paddle.zeros([1])
        predict = predict.transpose([0, 2, 3, 1])
        predict = paddle.reshape(predict, [-1, c])
        predict = paddle.masked_select(predict, paddle.unsqueeze(target_mask.flatten(), [-1]))
        loss = F.cross_entropy(predict, target, weight=weight, reduction='mean' if self.size_average else 'sum')
        return loss
