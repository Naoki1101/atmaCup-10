from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype

from utils.data import DataHandler

dh = DataHandler()


def get_sparse_matrix(data, index: str, columns: str, values: str):

    data_ = data[[index, columns]].copy()

    if values == "one":
        data_[values] = 1
    else:
        data_[values] = data[values]

    data_.dropna(inplace=True)

    index_cate = CategoricalDtype(sorted(data_[index].unique()), ordered=True)
    columns_cate = CategoricalDtype(sorted(data_[columns].unique()), ordered=True)

    row = data_[index].astype(index_cate).cat.codes
    col = data_[columns].astype(columns_cate).cat.codes
    sparse_matrix = csr_matrix(
        (data_[values], (row, col)),
        shape=(index_cate.categories.size, columns_cate.categories.size),
    )

    index_values = index_cate.categories.values
    columns_values = columns_cate.categories.values

    return sparse_matrix, index_values, columns_values