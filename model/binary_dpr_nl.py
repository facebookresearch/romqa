# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from model.binary import Model as Base


class Model(Base):

    @classmethod
    def process_single(cls, candidate, constraints, caption=None):
        keep = [d for d, s in zip(candidate['dpr'], candidate['dpr_score']) if s >= 0.65]
        keep = keep[:10]
        if not keep:
            keep = candidate['dpr'][:1]
        evidence = ' '.join(keep)
        context = '{} ; {} ; {}'.format(candidate['text'], caption, evidence)
        return dict(context=context, label=candidate.get('is_answer', False), text=candidate['text'], query=caption)
