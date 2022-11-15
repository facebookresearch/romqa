import os
import bz2
import ujson as json
from typing import List, Set
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from ordered_set import OrderedSet


class Metric:
    """
    Interface for a metric.
    """

    def compute_one(self, pred, gold):
        """
        Computes metrics for one example.
        You must implement this.

        Args:
            pred: single prediction.
            gold: corresponding ground truth.
        """
        raise NotImplementedError()

    def __call__(self, pred, gold):
        return self.forward(pred, gold)

    def forward(self, pred: list, gold: list) -> dict:
        """
        Computes metric over list of predictions and ground truths and returns a dictionary of scores.

        Args:
            pred: list of predictions.
            gold: corresponding ground truths.
        """
        metrics = defaultdict(list)
        for pi, gi in zip(pred, gold):
            m = self.compute_one(pi, gi)
            for k, v in m.items():
                metrics[k].append(v)
        return {k: sum(v)/len(v) for k, v in metrics.items()}


class Accuracy(Metric):
    """
    Computes exact match accuracy under the key "acc".
    """

    def compute_one(self, pred, gold):
        return {'acc': pred == gold}


class SetF1(Metric):
    """
    Computes F1 score under the key "f1", "precision", and "recall".
    Here, both single prediction and ground truth are assumed to be a `set`.
    """

    def compute_one(self, pred: set, gold: set):
        common = pred.intersection(gold)
        precision = len(common) / max(1, len(pred))
        recall = len(common) / max(1, len(gold))
        denom = precision + recall
        f1 = (precision * recall * 2 / denom) if denom > 0 else 0
        return dict(f1=f1, precision=precision, recall=recall)

    def __call__(self, pred, gold, ignore_empty=False):
        metrics = dict(f1=[], recall=[], precision=[])
        for pi, gi in zip(pred, gold):
            if ignore_empty and (not gi or gi == {''}):
                continue
            for k, v in self.compute_one(pi, gi).items():
                metrics[k].append(v)
        return {k: (sum(v)/len(v)) for k, v in metrics.items()}


def evaluate_gold(cluster_ids: List[int], gold: List[Set], pred: List[Set]):
    assert len(cluster_ids) == len(pred)
    assert len(gold) == len(pred)
    f1 = SetF1()
    acc = Accuracy()
    metrics = f1(pred, gold, ignore_empty=True)
    metrics.update(acc(pred, gold))

    cluster_metrics = defaultdict(lambda: defaultdict(list))
    per_example = []
    for c, p, g in zip(cluster_ids, pred, gold):
        this = dict()
        this_f1 = this['f1'] = f1([p], [g])
        for k, v in this_f1.items():
            cluster_metrics[c][k].append(v)
        this_acc = this['acc'] = acc([p], [g])
        for k, v in this_acc.items():
            cluster_metrics[c][k].append(v)
        per_example.append(this)

    mins = defaultdict(list)
    for m in cluster_metrics.values():
        for k, vs in m.items():
            mins[k].append(min(vs))

    for k, v in mins.items():
        metrics['cluster_min_{}'.format(k)] = sum(v) / len(v)
    return metrics, per_example


def evaluate(data: List[dict], pred: List[OrderedSet], return_per_pred=False):
    cluster_ids = [ex['cluster_id'] for ex in data]
    gold = [{c['text'] for c in ex['candidates'] if c['is_answer']} for ex in data]
    all_gold = [{a['text'] for a in ex['complete_answer']} for ex in data]

    m, per_pred = evaluate_gold(cluster_ids, gold, pred)

    all_per_pred = {}
    for top_k in [1, 10, 100, None]:
        pred_k = [p[:top_k] for p in pred]
        all_ans, all_per_pred[top_k] = evaluate_gold(cluster_ids, all_gold, pred_k)
        for k, v in all_ans.items():
            m['complete_{}_@k{}'.format(k, top_k)] = v
    if return_per_pred:
        return m, per_pred, all_per_pred
    return m


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--fpred', help='prediction file')
    parser.add_argument('--fdata', help='data file')
    parser.add_argument('--fout', help='optional output file')
    cfg = parser.parse_args()

    print('Debugging')
    for root, dirs, files in os.walk(os.getcwd()):
        for f in files:
            print(os.path.join(root, f))

    print('Loading data')
    with bz2.open(cfg.fdata, 'rt') as f:
        val = json.load(f)

    with open(cfg.fpred) as f:
        pred = json.load(f)

    assert len(pred) == len(val), "Expected {} predictions but found {}".format(len(val), len(pred))
    val = [ex for ex in val if ex['complete_answer']]
    pred_set = [OrderedSet(pred[str(ex['id'])]) for ex in val]
    res, per_pred, all_per_pred = evaluate(val, pred_set, return_per_pred=True)

    if cfg.fout is not None:
        with open(cfg.fout, 'wt') as f:
            json.dump(res, f)
    print(json.dumps(res, indent=2))
