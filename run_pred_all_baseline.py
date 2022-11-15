#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import bz2
import hydra
import warnings
import json
from ordered_set import OrderedSet
from evaluation import evaluate


warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


@hydra.main(config_path='conf', config_name='static')
def main(cfg):

    print('Loading data from {}'.format(cfg.fdata))
    with bz2.open(cfg.fdata) as f:
        data = json.load(f)
    data = [ex for ex in data if ex['complete_answer']]

    pred = []
    for ex in data:
        pred.append(OrderedSet([c['text'] for c in ex['candidates']]))
    print('predict all candidates')
    out = evaluate(data, pred)
    print(out)
    split = os.path.basename(cfg.fdata)
    with open('eval.{}'.format(split), 'wt') as f:
        json.dump(out, f, indent=2)


if __name__ == '__main__':
    main()
