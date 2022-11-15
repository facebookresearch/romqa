import ujson as json
import glob
import tqdm


def check(fname, write=False):
    print('checking', fname)
    with open(fname, 'rt') as f:
        data = json.load(f)
    for ex in tqdm.tqdm(data):
        expect = {'id', 'cluster_id', 'constraints', 'candidates', 'complete_answer', 'question'}
        for a in ex['complete_answer']:
            assert set(a.keys()) == {'text', 'uri'}, 'Got {}'.format(a.keys())
        assert set(ex.keys()) == expect, 'Got {}'.format(ex.keys())

        for c in ex['candidates']:
            expect = {'uri', 'is_answer', 'text', 'aliases', 'desc'}
            assert set(c.keys()) == expect, 'Got {}'.format(c.keys())
    if write:
        with open(fname, 'wt') as f:
            json.dump(data, f)


for fname in glob.glob('*.json'):
    check(fname, write=False)
