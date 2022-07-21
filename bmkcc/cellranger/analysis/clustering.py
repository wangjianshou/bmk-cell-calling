#!/usr/bin/env python
#
# Copyright (c) 2017 10X Genomics, Inc. All rights reserved.
#
from __future__ import absolute_import

import collections
import numpy as np
import os
import sys
import bmkcc.cellranger.analysis.io as analysis_io
import bmkcc.cellranger.io as cr_io

CLUSTER_TYPE_KMEANS = "kmeans"
CLUSTER_TYPE_GRAPHCLUST = "graphclust"
CLUSTER_TYPE_KMEDOIDS = "kmedoids"
CLUSTER_TYPE_CELLTYPES = "celltype"

CLUSTERING = collections.namedtuple(
    "CLUSTERING",
    [
        "clusters",
        "num_clusters",
        "cluster_score",
        "clustering_type",
        "global_sort_key",
        "description",
    ],
)


def format_legacy_clustering_key(cluster_type, cluster_param):
    assert cluster_type == CLUSTER_TYPE_KMEANS
    return "_%d" % cluster_param


def format_clustering_key(cluster_type, cluster_param):
    """ Generate a machine-readable string that describes a particular clustering """
    if cluster_type == CLUSTER_TYPE_KMEANS:
        return "%s_%d_clusters" % (CLUSTER_TYPE_KMEANS, cluster_param)
    elif cluster_type == CLUSTER_TYPE_KMEDOIDS:
        return "%s_%d_clusters" % (CLUSTER_TYPE_KMEDOIDS, cluster_param)
    elif cluster_type == CLUSTER_TYPE_GRAPHCLUST:
        return CLUSTER_TYPE_GRAPHCLUST
    elif cluster_type == CLUSTER_TYPE_CELLTYPES:
        return CLUSTER_TYPE_CELLTYPES
    else:
        raise ValueError("Unsupported cluster type: %s" % cluster_type)


def parse_clustering_key(clustering_key):
    """ Parse the output of format_clustering_key() """
    if clustering_key == CLUSTER_TYPE_GRAPHCLUST:
        return (clustering_key, 0)
    elif clustering_key.startswith(CLUSTER_TYPE_KMEANS):
        _, n_clusters, _ = clustering_key.split("_")
        return (CLUSTER_TYPE_KMEANS, int(n_clusters))
    elif clustering_key.startswith(CLUSTER_TYPE_KMEDOIDS):
        _, n_clusters, _ = clustering_key.split("_")
        return (CLUSTER_TYPE_KMEDOIDS, int(n_clusters))
    elif clustering_key == CLUSTER_TYPE_CELLTYPES:
        return (clustering_key, 0)
    else:
        raise ValueError("Unsupported clustering type for clustering key: %s" % clustering_key)


def humanify_clustering_key(clustering_key):
    """ Make a cluster_key string human-readable """
    cluster_type, cluster_param = parse_clustering_key(clustering_key)
    if cluster_type == CLUSTER_TYPE_GRAPHCLUST:
        return "Graph-based"
    elif cluster_type == CLUSTER_TYPE_KMEANS:
        return "K-means (K=%d)" % cluster_param
    elif cluster_type == CLUSTER_TYPE_KMEDOIDS:
        return "K-medoids (K=%d)" % cluster_param
    elif cluster_type == CLUSTER_TYPE_CELLTYPES:
        return "Celltypes"
    else:
        raise ValueError(
            "Unsupported clustering type %s for clustering key: %s" % (cluster_type, clustering_key)
        )


def relabel_by_size(labels):
    """Relabel clusters so they are sorted by number of members, descending.
    Args: labels (np.array(int)): 1-based cluster labels"""
    order = np.argsort(np.argsort(-np.bincount(labels)))
    return 1 + order[labels]


def save_clustering_csv(base_dir, clustering_key, labels, barcodes):
    out_dir = os.path.join(base_dir, clustering_key)
    cr_io.makedirs(out_dir, allow_existing=True)

    clusters_fn = os.path.join(out_dir, "clusters.csv")

    header = ["Barcode", "Cluster"]
    analysis_io.save_matrix_csv(clusters_fn, labels, header, barcodes)


def create_legacy_kmeans_nodes(f, new_group_name, legacy_group_name, namedtuple, clustering_key):
    """Soft-link a legacy-structured (CR 1.2) kmeans subgroup (dest) to a new-style (CR 1.3) subgroup (src).
    The old-style was a group called 'kmeans' with subgroups named _K.
    The new-style is a group called 'clustering' with subgroups named kmeans_K_clusters, etc."""
    group = f.create_group(f.root, legacy_group_name)

    cluster_type, cluster_param = parse_clustering_key(clustering_key)
    if cluster_type != CLUSTER_TYPE_KMEANS:
        return

    legacy_key = format_legacy_clustering_key(cluster_type, cluster_param)
    subgroup = f.create_group(group, legacy_key)
    for field in namedtuple._fields:
        target = "/%s/_%s/%s" % (new_group_name, clustering_key, field)

        if f.__contains__(target):
            # NOTE: coerce `target` to 'str' here because pytables chokes on unicode `target`
            f.create_soft_link(subgroup, field, target=str(target))
        else:
            sys.stderr.write(
                "Skipped soft-link of legacy dataset to %s; node doesn't exist\n" % target
            )


def subselect_barcodes(clustering, bc_indices):
    """
    Args:
        clustering (CLUSTERS namedtuple)
    """
    return CLUSTERING(
        clusters=clustering.clusters[bc_indices],
        num_clusters=clustering.num_clusters,
        cluster_score=clustering.cluster_score,
        clustering_type=clustering.clustering_type,
        global_sort_key=clustering.global_sort_key,
        description=clustering.description,
    )


def get_cluster_sizes(clustering):
    """ Returns a numpy array containing cell-counts for each cluster """
    return np.bincount(clustering.clusters)[1:]


def sort_clusterings(clusterings):
    return sorted(clusterings, key=lambda x: x.global_sort_key)


def create_clustering(
    clusters, num_clusters, cluster_score, clustering_type, global_sort_key, description
):
    """ Create a clustering namedtuple. Use numpy arrays/scalars to ensure h5 compatibility """
    return CLUSTERING(
        clusters=np.asarray(clusters),
        num_clusters=np.int64(num_clusters),
        cluster_score=np.float64(cluster_score),
        clustering_type=np.string_(clustering_type),
        global_sort_key=np.float64(global_sort_key),
        description=np.string_(description),
    )
