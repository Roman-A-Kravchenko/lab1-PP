"""Заготовки задач на NumPy."""

import numpy as np
from grader_contracts.numpy_tasks import (
    BinarizeInput, ChessInput, EllipseInput, MatrixInput, MatrixStatistics,
    MatrixVectorBatchInput, OneHotInput, RandomMatrixInput, RectangleInput,
    TimeSeriesInput, TimeSeriesStatistics,
)


def sum_prod(data: MatrixVectorBatchInput) -> np.ndarray:
    matrices = np.asarray(data.matrices)
    vectors = np.asarray(data.vectors)
    n = matrices.shape[1]
    result = np.zeros((n, 1), dtype=np.result_type(matrices, vectors))
    for A, x in zip(matrices, vectors):
        result = result + A @ x
    return result


def binarize(data: BinarizeInput) -> np.ndarray:
    matrix = np.asarray(data.matrix)
    return (matrix > data.threshold).astype(matrix.dtype)


def unique_rows(data: MatrixInput) -> list[list[float]]:
    return [list(np.unique(row)) for row in np.asarray(data.matrix)]


def unique_columns(data: MatrixInput) -> list[list[float]]:
    return [list(np.unique(col)) for col in np.asarray(data.matrix).T]


def matrix_statistics(data: RandomMatrixInput) -> MatrixStatistics:
    rng = np.random.default_rng(data.seed)
    matrix = rng.normal(
        loc=data.mean, scale=data.std, size=(data.rows, data.columns)
    )
    return MatrixStatistics(
        matrix=matrix,
        row_means=np.mean(matrix, axis=1),
        column_means=np.mean(matrix, axis=0),
        row_variances=np.var(matrix, axis=1),
        column_variances=np.var(matrix, axis=0),
    )


def chess(data: ChessInput) -> np.ndarray:
    i = np.arange(data.rows).reshape(-1, 1)
    j = np.arange(data.columns).reshape(1, -1)
    return np.where((i + j) % 2 == 0, data.first, data.second)


def draw_rectangle(data: RectangleInput) -> np.ndarray:
    H, W = data.image_height, data.image_width
    image = np.empty((H, W, 3), dtype=np.uint8)
    image[:] = np.asarray(data.background_color, dtype=np.uint8)

    cy, cx = H // 2, W // 2
    y1 = max(0, cy - data.height // 2)
    x1 = max(0, cx - data.width // 2)
    y2 = min(H, y1 + data.height)
    x2 = min(W, x1 + data.width)
    image[y1:y2, x1:x2] = np.asarray(data.shape_color, dtype=np.uint8)
    return image


def draw_ellipse(data: EllipseInput) -> np.ndarray:
    H, W = data.image_height, data.image_width
    image = np.empty((H, W, 3), dtype=np.uint8)
    image[:] = np.asarray(data.background_color, dtype=np.uint8)

    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    mask = ((x - cx) ** 2) / data.semi_axis_x ** 2 \
         + ((y - cy) ** 2) / data.semi_axis_y ** 2 <= 1
    image[mask] = np.asarray(data.shape_color, dtype=np.uint8)
    return image


def analyze_time_series(data: TimeSeriesInput) -> TimeSeriesStatistics:
    values = np.asarray(data.values, dtype=float)
    p = data.window

    maxima = [i for i in range(1, len(values) - 1)
              if values[i] > values[i - 1] and values[i] > values[i + 1]]
    minima = [i for i in range(1, len(values) - 1)
              if values[i] < values[i - 1] and values[i] < values[i + 1]]

    moving_average = (np.convolve(values, np.ones(p) / p, mode="valid")
                      if 0 < p <= len(values)
                      else np.array([], dtype=float))

    return TimeSeriesStatistics(
        mean=float(np.mean(values)),
        variance=float(np.var(values)),
        std=float(np.std(values)),
        local_maxima_indices=np.array(maxima, dtype=int),
        local_minima_indices=np.array(minima, dtype=int),
        moving_average=moving_average,
    )


def one_hot(data: OneHotInput) -> np.ndarray:
    labels = np.asarray(data.labels, dtype=int)
    class_count = (data.class_count if data.class_count is not None
                   else (int(labels.max()) + 1 if labels.size else 0))
    result = np.zeros((labels.size, class_count), dtype=int)
    if labels.size:
        result[np.arange(labels.size), labels] = 1
    return result
