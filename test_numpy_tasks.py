import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput,
    RectangleInput, TimeSeriesInput,
)
from numpy_tasks import (
    sum_prod, binarize, unique_rows, unique_columns,
    matrix_statistics, chess, draw_rectangle, draw_ellipse,
    analyze_time_series, one_hot,
)


def test_sum_prod():
    A = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])
    x = np.array([[[1], [1]], [[1], [2]]])
    res = sum_prod(MatrixVectorBatchInput(A, x))
    assert res.shape == (2, 1)
    assert np.allclose(res, [[3], [5]])


def test_binarize():
    M = np.array([[0.1, 0.5], [0.6, 0.9]])
    out = binarize(BinarizeInput(M, 0.5))
    assert np.array_equal(out, [[0, 0], [1, 1]])
    assert np.array_equal(M, [[0.1, 0.5], [0.6, 0.9]])  # исходная не меняется


def test_unique():
    M = np.array([[1, 1, 2], [3, 4, 4]])
    assert unique_rows(MatrixInput(M)) == [[1, 2], [3, 4]]
    assert unique_columns(MatrixInput(M)) == [[1, 3], [1, 4], [2, 4]]


def test_matrix_statistics():
    s = matrix_statistics(RandomMatrixInput(2, 3, seed=0))
    assert s.matrix.shape == (2, 3)
    assert s.row_means.shape == (2,)
    assert s.column_means.shape == (3,)
    assert s.row_variances.shape == (2,)
    assert s.column_variances.shape == (3,)


def test_chess():
    c = chess(ChessInput(3, 4, 0, 1))
    assert np.array_equal(c, [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])


def test_images():
    img = draw_rectangle(RectangleInput(2, 2, 4, 4, (255, 0, 0), (0, 0, 0)))
    assert img.shape == (4, 4, 3)
    assert tuple(img[2, 2]) == (255, 0, 0)

    img = draw_ellipse(EllipseInput(1, 1, 5, 5, (0, 255, 0), (0, 0, 0)))
    assert img.shape == (5, 5, 3)
    assert tuple(img[2, 2]) == (0, 255, 0)


def test_time_series():
    ts = analyze_time_series(TimeSeriesInput([1, 3, 2, 4, 1], 3))
    assert ts.mean == 2.2
    assert list(ts.local_maxima_indices) == [1, 3]
    assert list(ts.local_minima_indices) == [2]
    assert np.allclose(ts.moving_average, [2.0, 3.0, 7 / 3])


def test_one_hot():
    oh = one_hot(OneHotInput([0, 2, 3, 0]))
    assert oh.shape == (4, 4)
    assert np.array_equal(oh, [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ])
    oh2 = one_hot(OneHotInput([0, 1], class_count=5))
    assert oh2.shape == (2, 5)