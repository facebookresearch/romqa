# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from model.seq2seq import Model as Base


class Model(Base):

    @classmethod
    def process(cls, perm):
        query = Model.construct_query(perm['constraints'])
        evidence = ' ; '.join([t for t, s in zip(perm['dpr'], perm['dpr_score'])])
        context = 'what {}? {}'.format(query, evidence)
        label = [a['text'] for a in perm.get('complete_answer', ())]
        out = dict(context=context, label=set(label), label_str=', '.join(label))
        return out
