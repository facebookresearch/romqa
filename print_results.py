import pandas as pd
import omegaconf
import argparse
import csv
import os


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--root', nargs='+')
parser.add_argument('--group', nargs='+', default=['model'])
parser.add_argument('--agg', nargs='+', default=['step', 'val_f1'])
parser.add_argument('--early_stop', default='val_f1')
args = parser.parse_args()


def load_best(dlog):
    best = 0
    best_row = None
    for root, dirs, files in os.walk(dlog):
        if 'metrics.csv' in files:
            with open(os.path.join(root, 'metrics.csv')) as f:
                reader = csv.reader(f)
                header = next(reader)
                for r in reader:
                    d = dict(zip(header, r))
                    for k, v in d.items():
                        try:
                            d[k] = float(v)
                        except Exception:
                            pass
                    d['path'] = root
                    v = d[args.early_stop]
                    if v and not pd.isna(v) and v > best:
                        best_row = d
                        best = v
    return best_row


all_logs = []
for p in args.root:
    for root, dirs, files in os.walk(p):
        dlog = os.path.join(root, 'logs')
        fconfig = os.path.join(root, 'config.yaml')
        if os.path.isfile(fconfig) and os.path.isdir(dlog):
            best = load_best(dlog)
            if best is not None:
                cfg = omegaconf.OmegaConf.load(fconfig)
                best.update(cfg)
                all_logs.append(best)
df = pd.DataFrame(all_logs)

agg = {}
for a in args.agg:
    agg['mean_{}'.format(a)] = (a, 'mean')
    agg['std_{}'.format(a)] = (a, 'std')
agg['count'] = (args.early_stop, 'count')

df = df[args.group + args.agg]
summary = df.groupby(args.group).agg(**agg)
print(summary)
