from ordered_set import OrderedSet
from wrangl.learn import SupervisedModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM as AutoModel
from wrangl.learn.metrics import SetF1, Accuracy


def shift_tokens_right(input_ids, pad_token_id):
    """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
        This is taken directly from modeling_bart.py
    """
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


class Model(SupervisedModel):

    MyAutoModel = AutoModel

    @classmethod
    def construct_query(cls, constraints):
        yes, no = [], []
        for c in constraints:
            if c['prop_dir'] == 'subj':
                x = '{} {}'.format(c['prop']['text'], c['other_ent']['text'])
            else:
                x = '{} {}'.format(c['other_ent']['text'], c['prop']['text'])
            if c['truthy']:
                yes.append(x)
            else:
                no.append(x)
        query = ' and '.join(yes)
        if no:
            query += ' but not ' + ' and not '.join(no)
        return query

    @classmethod
    def process(cls, perm):
        query = Model.construct_query(perm['constraints'])
        context = 'what {}?'.format(query)
        label = [a['text'] for a in perm.get('complete_answer', ())]
        out = dict(context=context, label=set(label), label_str=', '.join(label), id=perm['id'], cluster_id=perm['cluster_id'])
        return out

    def build_lm(self):
        return self.MyAutoModel.from_pretrained(self.hparams.lm)

    def __init__(self, cfg):
        super().__init__(cfg)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.lm)
        self.lm = self.build_lm()

    def encode_text(self, texts, max_length):
        return self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            truncation=True,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
            max_length=max_length,
        )

    def featurize(self, batch):
        context = self.encode_text([e['context'] for e in batch], max_length=self.hparams.generate.max_context_length).to(self.device)
        label = self.encode_text([e['label_str'] for e in batch], max_length=self.hparams.generate.max_label_length)['input_ids'].to(self.device)
        return dict(context=context, label=label)

    @classmethod
    def compute_metrics(cls, pred, gold, batch) -> dict:
        f1 = SetF1()
        acc = Accuracy()
        m = {k: v for k, v in f1(pred, gold, ignore_empty=True).items()}
        m.update(acc(pred, gold))
        return m

    def compute_loss(self, out, feat, batch):
        return out.loss

    def extract_context(self, feat, batch):
        return [ex['context'][:1024] for ex in batch]

    def extract_pred(self, out, feat, batch):
        all_ents = []
        for sent in out:
            ents = OrderedSet([e.strip() for e in sent.split(',') if e.strip()])
            all_ents.append(ents)
        return all_ents

    @classmethod
    def extract_gold(cls, feat, batch):
        return [ex['label'] for ex in batch]

    def forward(self, feat, batch):
        context = feat['context']
        decoder_input_ids = shift_tokens_right(feat['label'], self.tokenizer.pad_token_id)
        out = self.lm(context['input_ids'], attention_mask=context['attention_mask'], decoder_input_ids=decoder_input_ids, labels=feat['label'])
        return out

    def infer(self, feat, batch):
        context = feat['context']
        generated_ids = self.lm.generate(
            context['input_ids'],
            attention_mask=context['attention_mask'],
            use_cache=True,
            decoder_start_token_id=self.tokenizer.pad_token_id,
            num_beams=self.hparams.generate.num_beams,
            max_length=self.hparams.generate.max_pred_label_length,
            early_stopping=True,
        )
        return self.tokenizer.batch_decode(generated_ids, clean_up_tokenization_spaces=True, skip_special_tokens=True)
