import math

import numpy as np
import torch
import random
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from utils.kmeans import Kmeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn import metrics
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
def get_cluster_score(cluster_pred, cluster_true):
  score = 0
  acc_score = 0
  nmi_score = 0
  test_idx = [i for i, x in enumerate(cluster_true) if x >= 0]
  cluster_pred = cluster_pred[test_idx]
  cluster_true = cluster_true[test_idx]
  score = adjusted_rand_score(cluster_pred, cluster_true)
  acc_score = cluster_acc(cluster_true, cluster_pred)
  nmi_score = metrics.normalized_mutual_info_score(cluster_pred, cluster_true)
  return score, acc_score, nmi_score

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200):
  # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
  # negatives for validation / test set)
  assert negative_edge_sampler.seed is not None
  negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      _, negative_samples = negative_edge_sampler.sample(size)

      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  return np.mean(val_ap), np.mean(val_auc)


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc

def compute_similarity_matrix(model, node_embeddings):
  similarity_matrix = []
  for i in range(len(node_embeddings)):
    data_nn = torch.cat([node_embeddings[i].view(1,-1)] * len(node_embeddings), dim=0)
    similarity_score = model.affinity_score(data_nn, node_embeddings).squeeze(dim=0).sigmoid()
    similarity_score = similarity_score.cpu().numpy()
    similarity_matrix.append(similarity_score)
  return np.array(similarity_matrix)

def compute_similarity_matrix_inner_product(model, node_embeddings):
  similarity_martix = []
  for i in range(len(node_embeddings)):
    data_nn = torch.cat([node_embeddings[i].view(1,-1)] * len(node_embeddings), dim=0)
    similarity_score = torch.sum(torch.mul(data_nn, node_embeddings), dim=1).sigmoid()
    similarity_score = similarity_score.cpu().numpy()
    similarity_martix.append(similarity_score)
  return np.array(similarity_martix)


def eval_cluster_quality(model, data, labels, cluster_num, device, batch_size=200, t=2.0, use_inner_product=False, cluster_method='AP'):
  node_embeddings = [None] * (len(np.unique(data.sources))+len(np.unique(data.destinations)))
  item_num = np.sum(labels >= 0, axis=0)
  assert item_num == len(np.unique(data.destinations))
  with torch.no_grad():
    model = model.eval()
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx:e_idx]
      source_node_embeddings, destination_node_emebddings, _ = model.compute_temporal_embeddings(sources_batch, destinations_batch, None, \
                                                                                                 timestamps_batch, edge_idxs_batch,contains_negative=False)
      # print(len(destination_node_emebddings), len(destinations_batch), len(sources_batch))
      for i in range(len(sources_batch)):
        try:
          node_embeddings[destinations_batch[i]-1] = destination_node_emebddings[i].cpu().numpy().tolist()
          node_embeddings[sources_batch[i]-1] = source_node_embeddings[i].cpu().numpy().tolist()
        except IndexError:
          print('Index out of range:', i, 'destinations_batch[i]:', destinations_batch[i], 'len(node_embeddings):', len(node_embeddings))
    # print(node_embeddings[0])
    node_embeddings = np.array(node_embeddings)
    node_embeddings = torch.from_numpy(node_embeddings).float().to(device)
    print(node_embeddings.shape)
    if cluster_method == 'AP':
      if use_inner_product:
        similarity_matrix = compute_similarity_matrix_inner_product(model, node_embeddings)
      else:
        similarity_matrix = compute_similarity_matrix(model, node_embeddings)
    # model_sp = SpectralClustering(n_clusters=cluster_num, affinity='precomputed')
      print(similarity_matrix.shape)
      if len(similarity_matrix.shape) > 2:
        similarity_matrix = similarity_matrix.squeeze(axis=-1)
      model_ap = AffinityPropagation(damping=0.9, preference=-7, affinity='precomputed')
      rs = model_ap.fit(similarity_matrix)
      cluster_length = len(rs.cluster_centers_indices_)
      cluster_pred = rs.labels_
      # cluster_pred = model_sp.fit_predict(similarity_matrix)
      print('cluster_length:', cluster_length)
    elif cluster_method == 'Kmeans':
      k_means = Kmeans(model=model, n_clusters=cluster_num, device=device, t=t, use_inner_product=use_inner_product)
      cluster_pred = k_means.fit(node_embeddings)
    print('cluster_pred:', cluster_pred.shape)
    print('labels:', labels.shape)
    assert len(cluster_pred) == len(labels)
    if len(cluster_pred.shape) > 1:
      cluster_pred = np.squeeze(cluster_pred,axis=1)
    ARAND, ACC, NMI = get_cluster_score(cluster_pred, labels)
  return ARAND, ACC, NMI
