#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import bz2
import json as json
import argparse
import omegaconf
from wrangl.learn import SupervisedModel
from train_baselines import MyDataset


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dsave', help='save folder of experiment')
    parser.add_argument('--overwrite', nargs='*', help='list of key=value overwrite arguments', default=[])
    parser.add_argument('--fsave', default='last.ckpt', help='checkpoint file')
    parser.add_argument('--fdata', default='dataset_construction/evidence/closed/top_5.dev.json.bz2', help='data file to predict on')
    parser.add_argument('--fout', default='pred.dev.json')
    args = parser.parse_args()

    fconfig = os.path.join(args.dsave, 'config.yaml')
    assert os.path.isfile(fconfig), 'Missing experiment config file at {}'.format(fconfig)

    cfg = omegaconf.OmegaConf.load(fconfig)
    overwrite = {}
    for kv in args.overwrite:
        k, v = kv.split('=')
        overwrite[k] = v
    cfg.update(overwrite)

    Model = SupervisedModel.load_model_class(cfg.model, root_dir=os.getcwd())

    print('Loading data')
    with bz2.open(args.fdata, 'rt') as f:
        val = json.load(f)

    dataset_val = MyDataset(val, Model.process)
    fout = os.path.join(args.dsave, args.fout)
    fsave = os.path.join(args.dsave, args.fsave)

    pred = Model.run_inference(cfg, fsave, dataset_val, test=False)
    # hack to serialize OrderedSets
    assert len(pred) == len(val)
    pred = {ex['id']: list(p) for ex, p in zip(val, pred)}
    print('Saving to {}'.format(fout))
    with open(fout, 'wt') as f:
        json.dump(pred, f)


if __name__ == '__main__':
    main()
