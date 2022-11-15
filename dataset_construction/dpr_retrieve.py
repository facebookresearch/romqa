from sentence_transformers import SentenceTransformer, util
import argparse
import torch
import bz2
import os
import tqdm
import ujson as json
import spacy


class TextFinder:

    def __init__(self, dpretrained_context_encoder, dpretrained_question_encoder, top_k, ent2docs, doc2sents):
        self.query_encoder = SentenceTransformer(dpretrained_question_encoder)
        self.passage_encoder = SentenceTransformer(dpretrained_context_encoder)
        self.ent2docs = ent2docs
        self.doc2sents = doc2sents
        self.top_k = top_k

    def to(self, device):
        self.query_encoder = self.query_encoder.to(device)
        self.passage_encoder = self.passage_encoder.to(device)
        return self

    def lookup(self, entities, query, top_k):
        # for each entity, look up topk docs
        indices = []
        sentences = []
        queries = []
        for i, ent in enumerate(entities):
            queries.append('{}. {}'.format(ent, query))
            sents_i = []
            for doc_id in self.ent2docs[ent['text']]:
                sents_i.extend(self.doc2sents[str(doc_id)])
            sentences.extend(sents_i)
            indices.extend([i] * len(sents_i))
        query_embedding = self.query_encoder.encode(queries, show_progress_bar=False)
        sentence_embedding = self.passage_encoder.encode(sentences, show_progress_bar=False)
        scores = util.cos_sim(query_embedding, sentence_embedding)
        output = []
        for j, sent_i, score_i in zip(indices, sentences, scores.transpose(0, 1)):
            if j > len(output)-1:
                output.append(dict(sents=[], scores=[]))
            o = output[j]
            o['sents'].append(sent_i)
            o['scores'].append(score_i[j].item())

        dpr = []
        dpr_scores = []
        for i, o in enumerate(output):
            scores = torch.tensor(o['scores'])
            values, indices = scores.topk(k=min(self.top_k, len(o['sents'])), largest=True, sorted=True)
            ent = entities[i]
            ordered_sents = [o['sents'][x] for x in indices.tolist()]
            dpr.append(ordered_sents)
            dpr_scores.append(values.tolist())
        assert len(dpr) == len(entities)
        return dpr, dpr_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--din', help='data directory', default='evidence/gold')
    parser.add_argument('--dout', help='data directory', default='evidence/closed')
    parser.add_argument('--fdocs', help='document corpus', default='evidence/sorted_docs.json.bz2')
    parser.add_argument('--fmatch', help='match results', default='evidence/bm25_docs_by_entity.jsonl.bz2')
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
            dpr, dpr_scores = finder.lookup(ex['candidates'], ex['question'], args.top_k)
            assert len(dpr) == len(dpr_scores) == len(ex['candidates'])
            for c, r, s in zip(ex['candidates'], dpr, dpr_scores):
                c['dpr'] = r
                c['dpr_score'] = s
        fout = os.path.join(args.dout, 'top_{}.'.format(args.top_k) + fname)
        with bz2.open(fout, 'wt') as f:
            json.dump(data, f)
