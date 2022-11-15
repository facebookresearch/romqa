from model.binary import Model as Base


class Model(Base):

    @classmethod
    def process_single(cls, candidate, constraints, caption=None):
        evidence = ' '.join([ev['text'] for ev in candidate['evidence']])
        context = '{} ; {} ; {}'.format(candidate['text'], caption, evidence)
        return dict(context=context, label=candidate.get('is_answer', False), text=candidate['text'], query=caption)
