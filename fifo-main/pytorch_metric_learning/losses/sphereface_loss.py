import paddle

from .large_margin_softmax_loss import LargeMarginSoftmaxLoss


class SphereFaceLoss(LargeMarginSoftmaxLoss):
    # implementation of https://arxiv.org/pdf/1704.08063.pdf
    def scale_logits(self, logits, embeddings):
        embedding_norms = paddle.norm(embeddings, p=2, axis=1)
        return logits * embedding_norms.unsqueeze(1) * self.scale
