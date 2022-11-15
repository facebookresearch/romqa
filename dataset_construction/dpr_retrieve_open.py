import os
import spacy
import tqdm
import argparse
import ujson as json
import bz2
from dpr_retrieve import TextFinder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--din', help='data directory', default='evidence/gold')
    parser.add_argument('--dout', help='data directory', default='evidence/open')
    parser.add_argument('--fdocs', help='document corpus', default='evidence/sorted_docs.json.bz2')
    parser.add_argument('--fmatch', help='match results', default='evidence/bm25_docs_by_question.jsonl.bz2')
    parser.add_argument('--dpretrained_question_encoder', default='./pretrained/facebook-dpr-question_encoder-multiset-base')
    parser.add_argument('--dpretrained_context_encoder', default='./pretrained/facebook-dpr-ctx_encoder-multiset-base')
    parser.add_argument('--top_k', default=20, type=int)
    args = parser.parse_args()

    fsents = args.fdocs.replace('docs', 'sents')
    assert fsents != args.fdocs
    print('loading docs')
    with bz2.open(args.fdocs, 'rt') as f:
        docs = json.load(f)

    if os.path.isfile(fsents):
        print('loading sentences')
        with bz2.open(fsents, 'rt') as f:
            sents = json.load(f)
    else:
        print('converting docs to sents')
        ids, texts = [], []
        for i, t in docs:
            ids.append(i)
            texts.append(t)
        nlp = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"])
        nlp.enable_pipe("senter")

        sents = {}
        for i, d in tqdm.tqdm(zip(ids, nlp.pipe(texts, n_process=64)), total=len(ids), desc='convert docs to sents'):
            sents[i] = [s.text for s in d.sents]
        with bz2.open(fsents, 'wt') as f:
            json.dump(sents, f)

    print('loading matches')
    match = {}
    with bz2.open(args.fmatch, 'rt') as f:
        for line in f:
            k, m = json.loads(line)
            match[k] = match_k = []
            for x in m:
                doc_id, text = docs[x['doc_index']]
                match[k].append(doc_id)

    # for proc in tqdm.tqdm(loader, desc='parallel passage lookup', total=len(data)):
    finder = TextFinder(args.dpretrained_context_encoder, args.dpretrained_question_encoder, args.top_k, match, sents).to('cuda')

    for fname in os.listdir(args.din):
        if not fname.endswith('.bz2'):
            continue
        print('Loading {}'.format(fname))
        with bz2.open(os.path.join(args.din, fname), 'rt') as f:
            data = json.load(f)
        for ex in tqdm.tqdm(data, desc='sentence lookup', total=len(data)):
            dpr, dpr_scores = finder.lookup([dict(text=ex['question'])], ex['question'], args.top_k)
            assert len(dpr) == len(dpr_scores) == 1
            ex['dpr'] = dpr[0]
            ex['dpr_scores'] = dpr_scores[0]
        fout = os.path.join(args.dout, 'top_{}.'.format(args.top_k) + fname)
        with bz2.open(fout, 'wt') as f:
            json.dump(data, f)
