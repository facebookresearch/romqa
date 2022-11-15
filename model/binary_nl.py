from model.binary import Model as Base


class Model(Base):

    @classmethod
    def process_single(cls, candidate, constraints, caption=None):
        context = '{} ; {}'.format(candidate['text'], caption)
        return dict(context=context, label=candidate.get('is_answer', False), text=candidate['text'], query=caption)
