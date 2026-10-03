"""Registry of clustering algorithms.

To add a new algorithm: create `my_algo.py` next to these files with NAME, DESCRIPTION, PARAMS
and run(), then add it to the dictionary below. The tabs, parameter widgets, scores and
heatmaps pick it up automatically.
"""
from . import dbscan_algo, infomap_algo, kmeans_algo, knn_algo

ALGORITHMS = {
    kmeans_algo.NAME: kmeans_algo,
    dbscan_algo.NAME: dbscan_algo,
    knn_algo.NAME: knn_algo,
    infomap_algo.NAME: infomap_algo,
}
