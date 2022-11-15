# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from ordered_set import OrderedSet
from transformers import AutoModelForSequenceClassification
from model.seq2seq import Model as Base


class Model(Base):

    MyAutoModel = AutoModelForSequenceClassification

    @classmethod
    def process_single(cls, candidate, constraints, caption=None):
        query = cls.construct_query(constraints)
        context = 'is {} the {}?'.format(candidate['text'], query)
        return dict(context=context, label=candidate.get('is_answer', False), query=query, text=candidate['text'])

    @classmethod
    def process(cls, ex):
        return dict(
            candidates=[cls.process_single(cand, ex['constraints'], caption=ex['question']) for cand in ex['candidates']],
            id=ex['id'],
            cluster_id=ex['cluster_id'],
        )

    def __init__(self, cfg):
        super().__init__(cfg)
        self.rng = np.random.default_rng(cfg.seed)

    def build_lm(self):
        return self.MyAutoModel.from_pretrained(self.hparams.lm, num_labels=2)

    def featurize(self, batch):
        context = []
        label = []
        text = []
        ids = []
        for i, d in enumerate(batch):
            candidates = d['candidates']
            if self.training:
                candidates = self.rng.choice(candidates, size=min(len(candidates), self.hparams.sample_size), replace=False).tolist()
            for j, c in enumerate(candidates):
                ids.append((i, j))
                text.append(c['text'])
                context.append(c['context'])
                label.append(c['label'])
        return dict(
            ids=ids,
            text=text,
            context_str=context,
            label=torch.tensor(label, dtype=torch.long, device=self.device),
        )

    def extract_context(self, feat, batch):
        context = []
        for d in batch:
            candidates = d['candidates']
            context.append(candidates[0]['query'][:1024])
        return context

    def extract_pred(self, out, feat, batch):
        preds = [OrderedSet() for _ in range(len(batch))]
        for idx, txt, out in zip(feat['ids'], feat['text'], out):
            i, j = idx
            if out:
                preds[i].add(txt)
        return preds

    @classmethod
    def extract_gold(cls, feat, batch):
        gold = []
        for d in batch:
            candidates = d['candidates']
            gold.append({c['text'] for c in candidates if c['label']})
        return gold

    def forward(self, feat, batch):
        context = self.encode_text(feat['context_str'], max_length=self.hparams.max_context_length).to(self.device)
        out = self.lm(context['input_ids'], attention_mask=context['attention_mask'], labels=feat['label'])
        return out

    def infer(self, feat, batch):
        batch_size = self.hparams.batch_size
        num_ex = len(feat['context_str'])
        outputs = []
        for i in range(0, num_ex, batch_size):
            feat_i = {k: v[i:i+batch_size] for k, v in feat.items()}
            batch_i = batch[i:i+batch_size]
            out_i = self.forward(feat_i, batch_i).logits.max(1)[1].tolist()
            outputs.extend(out_i)
        return outputs
