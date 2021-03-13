import dataclasses

import numpy as np
from umap import UMAP
from sklearn.decomposition import NMF, TruncatedSVD
from cuml.manifold import TSNE


@dataclasses.dataclass
class Decomposer:
    method: str
    n_components: int = 2
    random_state: int = 0

    def __post_init__(self):
        if self.method == "SVD":
            self.model = TruncatedSVD(
                n_components=self.n_components,
                random_state=self.random_state,
            )
        elif self.method == "TSNE":
            self.model = TSNE(
                n_components=self.n_components,
                random_state=self.random_state,
            )
        elif self.method == "UMAP":
            self.model = UMAP(
                n_components=self.n_components,
                random_state=self.random_state,
            )
        elif self.method == "NMF":
            self.model = NMF(
                n_components=self.n_components,
                random_state=self.random_state,
            )

    def fit(self, array: np.array) -> None:
        self.model.fit(array)

    def transform(self, array: np.array) -> np.array:
        decomposed_array = self.model.transform(array)
        return decomposed_array

    def fit_transform(self, array: np.array) -> None:
        decomposed_array = self.model.fit_transform(array)
        return decomposed_array
