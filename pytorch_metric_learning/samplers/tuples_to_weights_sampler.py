import numpy as np
import paddle
from paddle.io import Sampler
from paddle.io import Subset
from paddle.io import WeightedRandomSampler

from ..testers import BaseTester
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu


class TuplesToWeightsSampler(Sampler):
    def __init__(self, model, miner, dataset, subset_size=None, **tester_kwargs):
        self.model = model
        self.miner = miner
        self.dataset = dataset
        self.subset_size = subset_size
        self.tester = BaseTester(**tester_kwargs)
        self.device = self.tester.data_device
        self.weights = None

    def __len__(self):
        if self.subset_size:
            return self.subset_size
        return len(self.dataset)

    def __iter__(self):
        c_f.LOGGER.info("Computing embeddings in {}".format(self.__class__.__name__))

        if self.subset_size:
            indices = c_f.safe_random_choice(
                np.arange(len(self.dataset)), size=self.subset_size
            )
            curr_dataset = Subset(self.dataset, indices)
        else:
            indices = paddle.arange(len(self.dataset), dtype='int64')
            curr_dataset = self.dataset

        embeddings, labels = self.tester.get_all_embeddings(curr_dataset, self.model)
        labels = paddle.squeeze(labels, axis=1)
        hard_tuples = self.miner(embeddings, labels)

        self.weights = paddle.zeros(len(self.dataset), dtype='float32')
        self.weights.set_value(lmu.convert_to_weights(
            hard_tuples, labels.numpy(), dtype='float32'
        ))
        return iter(
            WeightedRandomSampler(
                self.weights, self.__len__(), replacement=True
            )
        )
