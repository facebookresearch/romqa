from model.seq2seq import Model as Base


class Model(Base):

    @classmethod
    def process(cls, perm):
        evidence = ' ; '.join([t for t, s in zip(perm['dpr'], perm['dpr_score'])])
        context = '{} : {}'.format(perm['question'], evidence)
        label = [a['text'] for a in perm.get('complete_answer', ())]
        out = dict(context=context, label=set(label), label_str=', '.join(label))
        return out
