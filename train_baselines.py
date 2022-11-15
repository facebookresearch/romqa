#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import bz2
import hydra
import torch
import warnings
import json as json
import pickle
import random
from evaluation import evaluate
from collections import defaultdict
from wrangl.learn import SupervisedModel
from torch.utils.data import Dataset


warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


class MyDataset(Dataset):

    def __init__(self, data, proc, single=False, limit=None, seed=0):
        if single:
            rng = random.Random(seed)
            by_cluster = defaultdict(list)
            for ex in data:
                by_cluster[ex['cluster_id']].append(ex)
            cluster_ids = sorted(list(by_cluster.keys()))
            self.data = [rng.choice(by_cluster[c]) for c in cluster_ids]
        else:
            self.data = data
        if limit is not None:
            self.data = data[:limit]
        self.processed = {}
        self.proc = proc

    def __getitem__(self, index):
        if index not in self.processed:
            self.processed[index] = self.proc(self.data[index])
        return self.processed[index]

    def __len__(self):
        return len(self.data)


def evaluate_dataset(raw_dataset, dataset, fname, Model, cfg):
    pred = Model.run_inference(cfg, cfg.test_resume, dataset, test=False)
    res, per_pred, all_per_pred = evaluate(raw_dataset, pred, return_per_pred=True)

    eval_split = os.path.splitext(os.path.basename(fname))[0]
    with open('{}.eval.json'.format(eval_split), 'wt') as f:
        json.dump(res, f)
    print('eval results for {}'.format(fname))
    print(res)

    with open('{}.pred.pkl'.format(eval_split), 'wb') as f:
        pickle.dump(pred, f)

    with open('{}.per_pred.eval.pkl'.format(eval_split), 'wt') as f:
        json.dump(dict(per_pred=per_pred, all_per_pred=all_per_pred), f)


def predict_dataset(raw_dataset, dataset, fname, Model, cfg):
    pred = Model.run_inference(cfg, cfg.test_resume, dataset, test=False)
    pred = {ex['id']: list(p) for ex, p in zip(raw_dataset, pred)}
    with open('{}.pred.json'.format(os.path.basename(fname)), 'wt') as f:
        json.dump(pred, f)


@hydra.main(config_path='conf', config_name='classify')
def main(cfg):
    torch.manual_seed(cfg.seed)
    Model = SupervisedModel.load_model_class(cfg.model)

    print('Loading data')
    with bz2.open(cfg.fval, 'rt') as f:
        val = json.load(f)

    if cfg.debug:
        val = val[:100]

    dataset_val = MyDataset(val, Model.process)

    if not cfg.test_only:
        with bz2.open(cfg.ftrain, 'rt') as f:
            train = json.load(f)
        limit = None
        if cfg.limit:
            num_clusters = len({ex['cluster_id'] for ex in train})
            limit = num_clusters
        dataset_train = MyDataset(train, Model.process, cfg.single_example_per_cluster, seed=cfg.seed, limit=limit)
        Model.run_train_test(cfg, dataset_train, dataset_val)

    evaluate_dataset(val, dataset_val, cfg.fval, Model, cfg)

    with open(cfg.ftest, 'rt') as f:
        test = json.load(f)

    if cfg.debug:
        test = test[:100]

    dataset_test = MyDataset(test, Model.process)
    predict_dataset(test, dataset_test, cfg.ftest, Model, cfg)


if __name__ == '__main__':
    main()
