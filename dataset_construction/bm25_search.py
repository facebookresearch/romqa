import argparse
import string
import os
from tqdm.auto import tqdm
import pickle
import bz2
import ray
import numpy as np
import ujson as json
from stop_words import STOP_WORDS
from wrangl.data import IterableDataset, Processor


# We lower case our text and remove stop-words from indexing
def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in STOP_WORDS:
            tokenized_doc.append(token)
    return tokenized_doc


@ray.remote
class MyProcessor(Processor):

    def __init__(self, fbm25, top_k):
        with open(fbm25, 'rb') as f:
            self.bm25 = pickle.load(f)
        self.top_k = top_k

    def process(self, query):
        bm25_scores = self.bm25.get_scores(bm25_tokenizer(query))
        top_k_inds = np.argpartition(bm25_scores, -self.top_k)[-self.top_k:].tolist()
        return query, top_k_inds, [bm25_scores[i] for i in top_k_inds]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--doc_index', default='evidence/bm25_index.pkl')
    parser.add_argument('--data_dir', default='evidence/gold')
    parser.add_argument('--data_out', default='evidence')
    parser.add_argument('--top_k', default=5, type=int)
    parser.add_argument('--num_workers', default=24, type=int)
    args = parser.parse_args()

    print('loading bm25')
    pool = ray.util.ActorPool([MyProcessor.remote(args.doc_index, args.top_k) for _ in range(args.num_workers)])

    print('computing entity and question set')
    entities = set()
    questions = set()
    for fname in os.listdir(args.data_dir):
        if fname.endswith('.json.bz2'):
            fname = os.path.join(args.data_dir, fname)
            with bz2.open(fname, 'rt') as f:
                print('loading', fname)
                data = json.load(f)
            for ex in tqdm(data, desc='finding entities and questions in {}'.format(fname)):
                questions.add(ex['question'])
                for c in ex['candidates']:
                    entities.add(c['text'])
    entities = sorted(list(entities))
    questions = sorted(list(questions))

    loader = IterableDataset(entities, pool, cache_size=args.num_workers*10, shuffle=True, timeout=1200)

    with bz2.open(os.path.join(args.data_out, 'bm25_docs_by_entity.jsonl.bz2'), 'wt') as f:
        for ent, top_k_inds, top_k_scores in tqdm(loader, total=len(entities), desc='bm25 searching entities'):
            lst = []
            for i, s in zip(top_k_inds, top_k_scores):
                lst.append(dict(doc_index=i, bm25_score=s))
            x = json.dumps([ent, lst])
            f.write(x + '\n')

    loader = IterableDataset(questions, pool, cache_size=args.num_workers*10, shuffle=True, timeout=1200)

    with bz2.open(os.path.join(args.data_out, 'bm25_docs_by_question.jsonl.bz2'), 'wt') as f:
        for que, top_k_inds, top_k_scores in tqdm(loader, total=len(questions), desc='bm25 searching questions'):
            lst = []
            for i, s in zip(top_k_inds, top_k_scores):
                lst.append(dict(doc_index=i, bm25_score=s))
            x = json.dumps([que, lst])
            f.write(x + '\n')
