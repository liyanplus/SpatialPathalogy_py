import sys

sys.path.insert(0, r'/Volumes/GoogleDrive/My Drive/Colab Notebooks/')
# sys.path.insert(0, r'/Volumes/GoogleDrive/My Drive/Research/Current/UMichCancer/col/spatial_pathalogy')
# sys.path.insert(0, r'C:/Users/MajidCSci/Downloads/Experiment')

import os
import numpy as np
import torch as th
import torch.nn as th_nn
import torch.optim as optim
from src.UMichPathology.pathology_classifier import PathologyClassifier
import src.helper.multivariate_correlation_proc as mcp
import src.UMichPathology.file_operation as fo
import src.UMichPathology.data_augment as da
import random
from datetime import datetime
import gc
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_auc_score
from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from scipy.spatial import KDTree
import itertools
from multiprocessing.pool import ThreadPool

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


def valid_proc(valid_file_paths, valid_ys, classifier,
               min_grid_scale, max_grid_scale, grid_scale_count,
               neighborhood_distance,
               feature_type_count, sampling_ratio):
    pred_y_probs = []
    true_ys = []
    # validation
    for valid_file_path, valid_y in zip(valid_file_paths, valid_ys):
        raw_data = fo.read_file(valid_file_path)
        for partial_idx, partial_data in enumerate(da.get_partial_data(raw_data)):
            pointpairs = fo.get_pointpairs(valid_file_path, partial_data, partial_idx, neighborhood_distance)
            for augmented_data in da.get_rotate_data(partial_data):
                neighborhood_tensor, core_point_idxs = \
                    fo.get_neighborhood_representation(valid_file_path, augmented_data, pointpairs,
                                                       min_grid_scale, max_grid_scale, grid_scale_count,
                                                       feature_type_count, sampling_ratio)

                augmented_data = augmented_data[:, 2].int().to(device)
                neighborhood_tensor = neighborhood_tensor.to(device)
                core_point_idxs = core_point_idxs.to(device)

                y, _ = classifier(augmented_data, neighborhood_tensor,
                                  core_point_idxs)
                pred_y_probs.append(y.item())
                true_ys.append(valid_y)
            th.cuda.empty_cache()

    pred_y_probs = np.array(pred_y_probs)
    true_ys = np.array(true_ys)

    print(true_ys)
    print(pred_y_probs)

    print('MCNet: auc: {}; precision: {}; recall: {}; f1: {}, acc: {}'.format(
        roc_auc_score(true_ys, pred_y_probs), precision_score(true_ys, pred_y_probs > 0.5),
        recall_score(true_ys, pred_y_probs > 0.5), f1_score(true_ys, pred_y_probs > 0.5),
        accuracy_score(true_ys, pred_y_probs > 0.5)
    ))


def train_proc(min_grid_scale=1,
               max_grid_scale=100,
               grid_scale_count=10,
               neighborhood_distance=200,
               feature_type_count=9,
               pr_representation_dim=32,
               pp_representation_dim=128,
               learning_rate=1e-4,
               apply_attention=True,
               epoch=100,
               relu_slope=1e-2,
               regularization_weight=50,
               diff_weight=1e-3,
               batch_size=32,
               sampling_ratio=1,
               model_path=r'/content/drive/MyDrive/Colab Notebooks/prnet.model'):
    class1_file_paths = list(
        fo.file_system_scrawl('/content/drive/MyDrive/Research/Current/UMichCancer/Data/Anon_Group1', '.txt'))

    class2_file_paths = list(
        fo.file_system_scrawl('/content/drive/MyDrive/Research/Current/UMichCancer/Data/Anon_Group2', '.txt'))

    train_file_paths1, valid_file_paths1, train_ys1, valid_ys1 = \
        fo.split_train_valid(class1_file_paths, [0] * len(class1_file_paths))

    train_file_paths2, valid_file_paths2, train_ys2, valid_ys2 = \
        fo.split_train_valid(class2_file_paths, [1] * len(class2_file_paths))

    valid_file_paths = np.hstack((valid_file_paths1, valid_file_paths2))
    valid_ys = np.hstack((valid_ys1, valid_ys2))

    classifier = PathologyClassifier(feature_type_count, grid_scale_count,
                                     pr_representation_dim, pp_representation_dim,
                                     apply_attention, leaky_relu_slope=relu_slope).to(device)

    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    criterion = th_nn.CrossEntropyLoss()

    batch_pred_ys = th.zeros(batch_size, dtype=th.float, device=device)
    batch_true_ys = th.zeros(batch_size, dtype=th.long, device=device)

    train_true_pos = 0
    train_false_pos = 0
    train_true_neg = 0
    train_false_neg = 0

    idx = 0
    pr_diff_sum = 0
    for epoch_idx in range(epoch):
        # train
        batch_idx = 0

        for train_pair1, train_pair2 in zip(
                random.sample(list(zip(train_file_paths1, train_ys1)), min(len(train_ys1), len(train_ys2))),
                random.sample(list(zip(train_file_paths2, train_ys2)), min(len(train_ys1), len(train_ys2)))
        ):
            train_file_path1, cur_y1 = train_pair1
            train_file_path2, cur_y2 = train_pair2

            raw_data1 = fo.read_file(train_file_path1)
            raw_data2 = fo.read_file(train_file_path2)

            for partial_idx, (partial_data1, partial_data2) in enumerate(zip(
                    da.get_partial_data(raw_data1),
                    da.get_partial_data(raw_data2)
            )):
                pointpairs1 = fo.get_pointpairs(train_file_path1, partial_data1, partial_idx, neighborhood_distance)
                pointpairs2 = fo.get_pointpairs(train_file_path2, partial_data2, partial_idx, neighborhood_distance)

                for rotate_idx, (augmented_data1, augmented_data2) in enumerate(zip(
                        da.get_rotate_data(partial_data1),
                        da.get_rotate_data(partial_data2)
                )):
                    if rotate_idx != epoch_idx % 4:
                        continue

                    neighborhood_tensor1, core_point_idxs1 = \
                        fo.get_neighborhood_representation(train_file_path1, augmented_data1, pointpairs1,
                                                           min_grid_scale, max_grid_scale, grid_scale_count,
                                                           feature_type_count, sampling_ratio)

                    augmented_data1 = augmented_data1[:, 2].int().to(device)
                    neighborhood_tensor1 = neighborhood_tensor1.to(device)
                    core_point_idxs1 = core_point_idxs1.to(device)

                    batch_pred_ys[idx], pred_pr1 = classifier(augmented_data1, neighborhood_tensor1,
                                                              core_point_idxs1)
                    batch_true_ys[idx] = cur_y1

                    if batch_pred_ys[idx] >= 0.5 and batch_true_ys[idx] == 1:
                        train_true_pos += 1
                    elif batch_pred_ys[idx] <= 0.5 and batch_true_ys[idx] == 0:
                        train_true_neg += 1
                    elif batch_pred_ys[idx] >= 0.5 and batch_true_ys[idx] == 0:
                        train_false_pos += 1
                    else:
                        train_false_neg += 1

                    neighborhood_tensor2, core_point_idxs2 = \
                        fo.get_neighborhood_representation(train_file_path2, augmented_data2, pointpairs2,
                                                           min_grid_scale, max_grid_scale, grid_scale_count,
                                                           feature_type_count, sampling_ratio)

                    augmented_data2 = augmented_data2[:, 2].int().to(device)
                    neighborhood_tensor2 = neighborhood_tensor2.to(device)
                    core_point_idxs2 = core_point_idxs2.to(device)

                    batch_pred_ys[idx + 1], pred_pr2 = classifier(augmented_data2, neighborhood_tensor2,
                                                                  core_point_idxs2)
                    batch_true_ys[idx + 1] = cur_y2

                    pr_diff_sum += 1 / (th.norm(
                        pred_pr1 - pred_pr2, 1
                    ) + 1e-5)

                    if batch_pred_ys[idx + 1] >= 0.5 and batch_true_ys[idx + 1] == 1:
                        train_true_pos += 1
                    elif batch_pred_ys[idx + 1] <= 0.5 and batch_true_ys[idx + 1] == 0:
                        train_true_neg += 1
                    elif batch_pred_ys[idx + 1] >= 0.5 and batch_true_ys[idx + 1] == 0:
                        train_false_pos += 1
                    else:
                        train_false_neg += 1

                    idx += 2

                    if idx % 2 == 0:
                        th.cuda.empty_cache()

                    if idx >= batch_size:
                        # print('{0} - Epoch {1} batch {2} start training backward.'.format(
                        #     datetime.now().strftime("%H:%M:%S"), epoch_idx, batch_idx))
                        optimizer.zero_grad()
                        paras = th.cat([x.view(-1) for x in classifier.parameters()])
                        regularization = th.norm(paras, 1) / (paras.shape[0] + 1)
                        ce = criterion(th.stack([1 - batch_pred_ys, batch_pred_ys], 1), batch_true_ys)
                        loss = ce + regularization_weight * regularization + diff_weight * pr_diff_sum

                        loss.backward()
                        th_nn.utils.clip_grad_value_(classifier.parameters(), 0.5)
                        optimizer.step()

                        # for param_group in optimizer.param_groups:
                        #     param_group['lr'] = 1e-3 - (1e-3 - 1e-4) / 10 * min(epoch_idx, 10)

                        print(
                            '{4} - Epoch: {0:5d}, Batch: {1:5d}, Training loss: {2:.3f}, Cross entropy: {3:.3f}'.format(
                                epoch_idx, batch_idx, loss.item(), ce.item(), datetime.now().strftime("%H:%M:%S")
                            ))

                        batch_pred_ys = th.zeros(batch_size, dtype=th.float, device=device)
                        batch_true_ys = th.zeros(batch_size, dtype=th.long, device=device)

                        idx = 0
                        pr_diff_sum = 0

                        print('{5} - Epoch: {0:5d}; Batch: {6:5d}; Training: '
                              'True positive: {1:5d}; '
                              'True negative: {2:5d}; '
                              'False positive: {3:5d}; '
                              'False negative: {4:5d}.'.format(epoch_idx,
                                                               train_true_pos, train_true_neg,
                                                               train_false_pos, train_false_neg,
                                                               datetime.now().strftime("%H:%M:%S"),
                                                               batch_idx))

                        train_true_pos = 0
                        train_false_pos = 0
                        train_true_neg = 0
                        train_false_neg = 0
                        gc.collect()
                        th.cuda.empty_cache()

                        batch_idx += 1

        th.save(classifier.state_dict(), model_path)

        if (epoch_idx + 1) % 4 == 0:
            valid_proc(valid_file_paths, valid_ys, classifier,
                       min_grid_scale, max_grid_scale, grid_scale_count,
                       neighborhood_distance, feature_type_count, 1)


def feature_importance(min_grid_scale=10,
                       max_grid_scale=50,
                       grid_scale_count=10,
                       neighborhood_distance=200,
                       feature_type_count=9,
                       pr_representation_dim=32,
                       pp_representation_dim=128,
                       apply_attention=True,
                       relu_slope=1e-2,
                       sampling_ratio=1,
                       model_path=r'/content/drive/MyDrive/Colab Notebooks/prnet.model'):
    class1_file_paths = list(
        fo.file_system_scrawl('/content/drive/MyDrive/Research/Current/UMichCancer/Data/Anon_Group1', '.txt'))

    class2_file_paths = list(
        fo.file_system_scrawl('/content/drive/MyDrive/Research/Current/UMichCancer/Data/Anon_Group2', '.txt'))

    ys1 = [0] * len(class1_file_paths)
    ys2 = [1] * len(class2_file_paths)

    classifier = PathologyClassifier(feature_type_count, grid_scale_count,
                                     pr_representation_dim, pp_representation_dim,
                                     apply_attention, leaky_relu_slope=relu_slope).to(device)

    results = defaultdict(list)

    if os.path.exists(model_path):
        classifier.load_state_dict(th.load(model_path))

    for train_pair1, train_pair2 in zip(
            random.choices(list(zip(class1_file_paths, ys1)), k=max(len(class1_file_paths), len(class2_file_paths))),
            random.choices(list(zip(class2_file_paths, ys2)), k=max(len(class1_file_paths), len(class2_file_paths)))
    ):
        train_file_path1, cur_y1 = train_pair1
        train_file_path2, cur_y2 = train_pair2

        raw_data1 = fo.read_file(train_file_path1)
        raw_data2 = fo.read_file(train_file_path2)

        print(train_file_path2)

        for partial_idx, (partial_data1, partial_data2) in enumerate(zip(
                da.get_partial_data(raw_data1),
                da.get_partial_data(raw_data2)
        )):
            pointpairs1 = fo.get_pointpairs(train_file_path1, partial_data1, partial_idx, neighborhood_distance)
            pointpairs2 = fo.get_pointpairs(train_file_path2, partial_data2, partial_idx, neighborhood_distance)

            for rotate_idx, (augmented_data1, augmented_data2) in enumerate(zip(
                    da.get_rotate_data(partial_data1),
                    da.get_rotate_data(partial_data2)
            )):
                neighborhood_tensor1, core_point_idxs1 = \
                    fo.get_neighborhood_representation(train_file_path1, augmented_data1, pointpairs1,
                                                       min_grid_scale, max_grid_scale, grid_scale_count,
                                                       feature_type_count, sampling_ratio)

                augmented_data1 = augmented_data1[:, 2].int().to(device)
                neighborhood_tensor1 = neighborhood_tensor1.to(device)
                core_point_idxs1 = core_point_idxs1.to(device)

                pr_representations1 = classifier.pr_net(augmented_data1, neighborhood_tensor1, core_point_idxs1)

                neighborhood_tensor2, core_point_idxs2 = \
                    fo.get_neighborhood_representation(train_file_path2, augmented_data2, pointpairs2,
                                                       min_grid_scale, max_grid_scale, grid_scale_count,
                                                       feature_type_count, sampling_ratio)

                augmented_data2 = augmented_data2[:, 2].int().to(device)
                neighborhood_tensor2 = neighborhood_tensor2.to(device)
                core_point_idxs2 = core_point_idxs2.to(device)

                pr_representations2 = classifier.pr_net(augmented_data2, neighborhood_tensor2, core_point_idxs2)

                for idx in range(feature_type_count ** 2):
                    tmp_pr_rep1 = pr_representations1.detach().clone()
                    tmp_pr_rep2 = pr_representations2.detach().clone()

                    tmp_pr_rep1[idx * pr_representation_dim: (idx + 1) * pr_representation_dim] = \
                        pr_representations2[idx * pr_representation_dim: (idx + 1) * pr_representation_dim]
                    tmp_pr_rep2[idx * pr_representation_dim: (idx + 1) * pr_representation_dim] = \
                        pr_representations1[idx * pr_representation_dim: (idx + 1) * pr_representation_dim]

                    x1 = classifier.classify_activate(classifier.classify_linear1(tmp_pr_rep1))
                    x1 = classifier.classify_activate(classifier.classify_linear2(x1))
                    y1 = classifier.final(classifier.classify_linear3(x1))

                    x2 = classifier.classify_activate(classifier.classify_linear1(tmp_pr_rep2))
                    x2 = classifier.classify_activate(classifier.classify_linear2(x2))
                    y2 = classifier.final(classifier.classify_linear3(x2))

                    results[idx].append([y1.item(), 0])
                    results[idx].append([y2.item(), 1])

                    gc.collect()
                    th.cuda.empty_cache()

    aucs = []
    for idx in range(feature_type_count ** 2):
        curr = np.array(results[idx])
        fpr, tpr, _ = roc_curve(curr[:, 1], curr[:, 0])
        aucs.append(auc(fpr, tpr))
    print(sorted(zip(aucs, range(len(aucs))), reverse=True))


def baseline():
    def get_category_data(point_data):
        point_data = point_data.detach().cpu().numpy()
        input_data = defaultdict(list)
        types = ["Treg", "APC", "Epithelial", "HelperT", "PDL1_CD3", "PDL1_CD8", "PDL1_FoxP3", "CD4", "CTLs"]
        for row_idx in range(point_data.shape[0]):
            input_data[types[point_data[row_idx, 2]]].append([point_data[row_idx, 0], point_data[row_idx, 1]])
        return {k: KDTree(np.array(v)) for k, v in input_data.items()}

    def get_augmented_data(raw_data_filepath, label):
        prs_data = []
        crossk_data = []
        neighbor_radii = [1, 50, 100, 150, 200]
        phenotype_names = ["Treg", "APC", "Epithelial", "HelperT", "PDL1_CD3", "PDL1_CD8", "PDL1_FoxP3", "CD4", "CTLs"]
        patterns = list(itertools.permutations(phenotype_names, 2))
        class_file_paths = list(fo.file_system_scrawl(raw_data_filepath, '.txt'))

        pool = ThreadPool(processes=8)

        def helper(file_path):
            print(file_path)
            raw_data = fo.read_file(file_path)
            for partial_data in da.get_partial_data(raw_data):
                input_data = get_category_data(partial_data)

                pr_data_entry = [label]
                crossk_data_entry = [label]

                for p in patterns:
                    if p[0] not in input_data or p[1] not in input_data:
                        pr_data_entry.extend([0] * len(neighbor_radii))
                        crossk_data_entry.extend([0] * len(neighbor_radii))
                    else:
                        prs, crossks = mcp.get_pr_n_cross_k(input_data, p[0], p[1], neighbor_radii)
                        pr_data_entry.extend([prs[r] for r in sorted(prs.keys())])
                        crossk_data_entry.extend([crossks[r] for r in sorted(crossks.keys())])
                prs_data.append(pr_data_entry)
                crossk_data.append(crossk_data_entry)

        def insert_res(a):
            prs_data.extend(a[0])
            crossk_data.extend(a[1])

        for fp in class_file_paths:
            pool.apply_async(helper, (fp,), callback=insert_res)

        pool.close()
        pool.join()
        return np.array(prs_data), np.array(crossk_data)

    if os.path.exists(r'/Volumes/GoogleDrive/My Drive/Research/Current/UMichCancer/prs.csv'):
        prs_data = np.loadtxt(r'/Volumes/GoogleDrive/My Drive/Research/Current/UMichCancer/prs.csv', delimiter=',')
        crossK_data = np.loadtxt(r'/Volumes/GoogleDrive/My Drive/Research/Current/UMichCancer/crossKs.csv',
                                 delimiter=',')
    else:
        class1_prs_data, class1_crossK_data = get_augmented_data(
            r'/Volumes/GoogleDrive/My Drive/Research/Current/UMichCancer/Data/Anon_Group1', 0)
        np.savetxt(r'/Volumes/GoogleDrive/My Drive/Research/Current/UMichCancer/prs_class1.csv', class1_prs_data,
                   delimiter=',')
        np.savetxt(r'/Volumes/GoogleDrive/My Drive/Research/Current/UMichCancer/crossKs_class1.csv', class1_crossK_data,
                   delimiter=',')
        class2_prs_data, class2_crossK_data = get_augmented_data(
            r'/Volumes/GoogleDrive/My Drive/Research/Current/UMichCancer/Data/Anon_Group2', 1)
        prs_data = np.vstack((class1_prs_data, class2_prs_data))
        crossK_data = np.vstack((class1_crossK_data, class2_crossK_data))
        np.savetxt(r'/Volumes/GoogleDrive/My Drive/Research/Current/UMichCancer/prs.csv', prs_data, delimiter=',')
        np.savetxt(r'/Volumes/GoogleDrive/My Drive/Research/Current/UMichCancer/crossKs.csv', crossK_data,
                   delimiter=',')

    roc_auc = []
    precision = []
    recall = []
    f1 = []
    acc = []
    for _ in range(10):
        X_train, X_test, y_train, y_test = train_test_split(prs_data[:, 1:], prs_data[:, 0], test_size=.2,
                                                            stratify=prs_data[:, 0])
        clf = DecisionTreeClassifier(max_depth=4)
        clf.fit(X_train, y_train)
        y_pred_prob = clf.predict_proba(X_test)[:, 1]
        roc_auc.append(roc_auc_score(y_test, y_pred_prob))
        precision.append(precision_score(y_test, y_pred_prob > 0.5))
        recall.append(recall_score(y_test, y_pred_prob > 0.5))
        f1.append(f1_score(y_test, y_pred_prob > 0.5))
        acc.append(accuracy_score(y_test, y_pred_prob > 0.5))
    print('PR-DT: auc: {}; precision: {}; recall: {}; f1: {}, acc: {}'.format(
        np.mean(roc_auc), np.mean(precision), np.mean(recall), np.mean(f1), np.mean(acc)
    ))

    roc_auc = []
    precision = []
    recall = []
    f1 = []
    acc = []
    for _ in range(10):
        X_train, X_test, y_train, y_test = train_test_split(crossK_data[:, 1:], crossK_data[:, 0], test_size=.2,
                                                            stratify=crossK_data[:, 0])
        clf = DecisionTreeClassifier(max_depth=4)
        clf.fit(X_train, y_train)
        y_pred_prob = clf.predict_proba(X_test)[:, 1]
        roc_auc.append(roc_auc_score(y_test, y_pred_prob))
        precision.append(precision_score(y_test, y_pred_prob > 0.5))
        recall.append(recall_score(y_test, y_pred_prob > 0.5))
        f1.append(f1_score(y_test, y_pred_prob > 0.5))
        acc.append(accuracy_score(y_test, y_pred_prob > 0.5))
    print('crossK-DT: auc: {}; precision: {}; recall: {}; f1: {}, acc: {}'.format(
        np.mean(roc_auc), np.mean(precision), np.mean(recall), np.mean(f1), np.mean(acc)
    ))

    roc_auc = []
    precision = []
    recall = []
    f1 = []
    acc = []
    for _ in range(10):
        X_train, X_test, y_train, y_test = train_test_split(prs_data[:, 1:], prs_data[:, 0], test_size=.2,
                                                            stratify=prs_data[:, 0])
        clf = RandomForestClassifier(max_depth=3, n_estimators=500)
        clf.fit(X_train, y_train)
        y_pred_prob = clf.predict_proba(X_test)[:, 1]
        roc_auc.append(roc_auc_score(y_test, y_pred_prob))
        precision.append(precision_score(y_test, y_pred_prob > 0.5))
        recall.append(recall_score(y_test, y_pred_prob > 0.5))
        f1.append(f1_score(y_test, y_pred_prob > 0.5))
        acc.append(accuracy_score(y_test, y_pred_prob > 0.5))
    print('PR-RF: auc: {}; precision: {}; recall: {}; f1: {}, acc: {}'.format(
        np.mean(roc_auc), np.mean(precision), np.mean(recall), np.mean(f1), np.mean(acc)
    ))

    roc_auc = []
    precision = []
    recall = []
    f1 = []
    acc = []
    for _ in range(10):
        X_train, X_test, y_train, y_test = train_test_split(crossK_data[:, 1:], crossK_data[:, 0], test_size=.2,
                                                            stratify=crossK_data[:, 0])
        clf = RandomForestClassifier(max_depth=3, n_estimators=500)
        clf.fit(X_train, y_train)
        y_pred_prob = clf.predict_proba(X_test)[:, 1]
        roc_auc.append(roc_auc_score(y_test, y_pred_prob))
        precision.append(precision_score(y_test, y_pred_prob > 0.5))
        recall.append(recall_score(y_test, y_pred_prob > 0.5))
        f1.append(f1_score(y_test, y_pred_prob > 0.5))
        acc.append(accuracy_score(y_test, y_pred_prob > 0.5))
    print('crossK-RF: auc: {}; precision: {}; recall: {}; f1: {}, acc: {}'.format(
        np.mean(roc_auc), np.mean(precision), np.mean(recall), np.mean(f1), np.mean(acc)
    ))

    roc_auc = []
    precision = []
    recall = []
    f1 = []
    acc = []
    for _ in range(5):
        X_train, X_test, y_train, y_test = train_test_split(prs_data[:, 1:], prs_data[:, 0], test_size=.2,
                                                            stratify=prs_data[:, 0])
        clf = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(4096, 4096, 4096))
        clf.fit(X_train, y_train)
        y_pred_prob = clf.predict_proba(X_test)[:, 1]
        roc_auc.append(roc_auc_score(y_test, y_pred_prob))
        precision.append(precision_score(y_test, y_pred_prob > 0.5))
        recall.append(recall_score(y_test, y_pred_prob > 0.5))
        f1.append(f1_score(y_test, y_pred_prob > 0.5))
        acc.append(accuracy_score(y_test, y_pred_prob > 0.5))
        print('PR-NN', roc_auc[-1], precision[-1], recall[-1], f1[-1], acc[-1])
    print('PR-NN: auc: {}; precision: {}; recall: {}; f1: {}, acc: {}'.format(
        np.mean(roc_auc), np.mean(precision), np.mean(recall), np.mean(f1), np.mean(acc)
    ))

    roc_auc = []
    precision = []
    recall = []
    f1 = []
    acc = []
    for _ in range(5):
        X_train, X_test, y_train, y_test = train_test_split(crossK_data[:, 1:], crossK_data[:, 0], test_size=.2,
                                                            stratify=crossK_data[:, 0])
        clf = MLPClassifier(alpha=1e-3, hidden_layer_sizes=(4096, 4096, 4096))
        clf.fit(X_train, y_train)
        y_pred_prob = clf.predict_proba(X_test)[:, 1]
        roc_auc.append(roc_auc_score(y_test, y_pred_prob))
        precision.append(precision_score(y_test, y_pred_prob > 0.5))
        recall.append(recall_score(y_test, y_pred_prob > 0.5))
        f1.append(f1_score(y_test, y_pred_prob > 0.5))
        acc.append(accuracy_score(y_test, y_pred_prob > 0.5))
        print('crossK-NN', roc_auc[-1], precision[-1], recall[-1], f1[-1], acc[-1])
    print('crossK-NN: auc: {}; precision: {}; recall: {}; f1: {}, acc: {}'.format(
        np.mean(roc_auc), np.mean(precision), np.mean(recall), np.mean(f1), np.mean(acc)
    ))


if __name__ == "__main__":
    baseline()
