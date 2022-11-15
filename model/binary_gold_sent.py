# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from model.binary import Model as Base


class Model(Base):

    @classmethod
    def process_single(cls, candidate, constraints, caption=None):
        query = cls.construct_query(constraints)
        evidence = ' '.join([ev['text'] for ev in candidate['evidence']])
        context = 'is {} the {}? {}'.format(candidate['text'], query, evidence)
        return dict(context=context, label=candidate.get('is_answer', False), text=candidate['text'], query=query)
