from model.binary import Model as Base


class Model(Base):

    @classmethod
    def process_single(cls, candidate, constraints, caption=None):
        query = cls.construct_query(constraints)
        keep = [d for d, s in zip(candidate['dpr'], candidate['dpr_score']) if s >= 0.65]
        keep = keep[:10]
        if not keep:
            keep = candidate['dpr'][:1]
        evidence = ' '.join(keep)
        context = 'is {} the {}? {}'.format(candidate['text'], query, evidence)
        return dict(context=context, label=candidate.get('is_answer', False), text=candidate['text'], query=query)
