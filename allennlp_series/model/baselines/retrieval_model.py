from typing import Dict, List, Tuple, Union, Any

import torch
import numpy as np

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from allennlp_series.common.constants import *
from torch.nn.modules.rnn import LSTMCell
from torch.nn.modules.linear import Linear
import torch.nn.functional as F

# import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
from allennlp.modules.token_embedders import Embedding
from allennlp_series.training.metrics.confusion_matrix import (
    ConfusionMatrix,
    F1MeasureCustom,
)


@Model.register("retrieval")
class RetrievalModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        distance_function: str = "l2",
        num_labels=6,
        transformation="none",
        model_name: str = None,
    ) -> None:

        super().__init__(vocab)
        # self._accuracy = ConfusionMatrix(num_labels=num_programs)  # CategoricalAccuracy()
        # self._f1 = F1MeasureCustom()
        self.distance_function = distance_function
        self.num_labels = num_labels
        self.transformation = transformation
        self.model_name = model_name

    def _24to12(self, val):
        return [v for i, v in enumerate(val) if i % 2 == 0]

    def forward(self, train_feats, val_feats) -> Tuple[Any, Any]:  # type: ignore

        from sklearn.neighbors import NearestNeighbors
        import numpy as np

        X = np.array(train_feats)
        print("Fitting NearestNeighbors...")
        if self.distance_function == "l2":
            p = 2
        elif self.distance_function == "l1":
            p = 1
        else:
            raise NotImplementedError
        nbrs = NearestNeighbors(n_neighbors=1, algorithm="ball_tree", p=p).fit(X)
        print("nbrs = ", nbrs)
        if self.transformation == "none":
            pass
        elif self.transformation == "24to12":
            val_feats = [self._24to12(val_featsi) for val_featsi in val_feats]
        else:
            raise NotImplementedError

        val_X = np.array(val_feats)
        distances, indices = nbrs.kneighbors(val_X)
        return distances, indices

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        metrics = {}
        # if reset:
        #     print("CM = ", self._accuracy)
        # metrics.update(self._accuracy.get_metric(reset))
        # metrics.update(self._f1.get_metric(reset))
        return metrics
