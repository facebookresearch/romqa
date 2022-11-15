# RoMQA

This repository contains the source code for [RoMQA: A Benchmark for Robust, Multi-evidence, Multi-answer Question Answering
](https://arxiv.org/abs/2210.14353).
If you are looking for the leaderboard, please see [this Codalab worksheet](https://worksheets.codalab.org/worksheets/0xc13c82ceb0414938b758a318dcc21dae).
If you find this helpful, please cite:

```
@inproceedings{ zhong2022romqa,
  title={ {RoMQA}: A Benchmark for Robust, Multi-evidence, Multi-answer Question Answering },
  author={ Victor Zhong and Weijia Shi and Wen-tau Yih and Luke Zettlemoyer },
  booktitle={ CoRR abs/2210.14353 },
  year={ 2022 }
}
```


## Data

Due to legal reasons, Meta cannot host reproduced Wikidata.
We include scripts to reproduce RoMQA data from annotations, Wikidata, and T-REx in the `dataset_construction` directory.
Alternatively, you can [download data from a third party](https://s3.us-west-1.wasabisys.com/vzhong-public/RoMQA/romqa_data.zip), which has been produced using scripts from this directory.
Once you have downloaded the data `romqa_data.zip`, unzip it `unzip romqa_data.zip` from the root folder to place data files in `./data`.
The experiment code assumes that `./data` contains the correct data files.
If you decide to produce data yourself, then you should manually place splits in the `data/{open,closed,gold}` directories.


## Running experiments

Run open setting
```bash
python train_baselines.py --config-name open --multirun hydra/launcher=slurm hydra.launcher.partition=<partition> model=seq2seq_nl,seq2seq_dpr_nl hydra.launcher.constraint=volta32gb seed=1,2,3,4,5 project=open-1
```

Run closed setting
```bash
python train_baselines.py --config-name closed --multirun hydra/launcher=slurm hydra.launcher.partition=<partition> model=binary_nl,binary_dpr_nl hydra.launcher.constraint=volta32gb seed=1,2,3,4,5 project=closed-1
```

Run gold evidence setting
```bash
python train_baselines.py --config-name gold --multirun hydra/launcher=slurm hydra.launcher.partition=<partition> model=binary_gold_sent_nl hydra.launcher.constraint=volta32gb seed=1,2,3,4,5 project=gold-1
```

You can run all of these commands without slurm by removing the `multirun` and `launcher` flags.
For example
```bash
python train_baselines.py --config-name closed model=binary_dpr_nl seed=1 project=local-closed-1
```

You can monitor the runs from the command line using the plotting utility:

```bash
wrangl plot -n 3 --curves eval -y val_f1 --type supervised saves/open-1/sweep/*-seq2seq-default*
```

You can also print the early-stopping results so far for the sweep:

```bash
python print_results.py --root saves/*/sweep
```

## Leaderboard and Submission

The RoMQA leaderboard is hosted on [CodaLab here](https://worksheets.codalab.org/worksheets/0xc13c82ceb0414938b758a318dcc21dae).

The released RoMQA test data contains no labels.
Submission involves submitting your model predictions to be evaluated against the gold test labels on [CodaLab](https://worksheets.codalab.org).
Your submission should be a JSON file containing a dictionary of key-value pairs.
The keys are the example `id`.
The values are the model predictions in the form of a list of top-k entities.
Entities are expected to match their Wikidata cononical text labels.

You should predict and evaluate with the dev set first and verify that the formatting is correct:

```bash
# open setting example
python predict.py --fdata data/open/top_20.dev.json.bz2 --fout pred.open.dev.json saves/open-1/sweep/15-seq2seq_dpr_nl-default/
python evaluation.py --fpred saves/open-1/sweep/15-seq2seq_dpr_nl-default/pred.open.dev.json --fdata data/gold/dev.json.bz2 --fout open.dev.eval.json
```

Make sure that `open.dev.eval.json` contains what you expect to see.

Next, generate predictions using your saved model on the unlabeled test data:

```bash
# open setting example
python predict.py --fdata data/open/top_20.test.noanswer.json.bz2 --fout pred.open.test.json saves/open-1/sweep/15-seq2seq_dpr_nl-default/
# closed setting example
python predict.py --fdata data/closed/top_20.test.noanswer.json.bz2 --fout pred.closed.test.json saves/closed-1/sweep/15-binary_dpr_nl-default/
```

Next, upload your dev predictions to CodaLab and note your `bundle id`.

```bash
cl upload pred.open.dev.json
# this will return you your bundle UID <my_open_dev_uid>
```

Run your bundle using the RoMQA official evaluation.

```bash
cl run -n <open_or_closed>_dev_<my_model_name> -d "<model_name> by <my_name> at <my_affiliation>" \
  --request-docker-image vzhong/romqa:0.1 \
  --request-memory 8g \
  evaluation.py:0x627bae34595e4bf4971197c9cb917f5e \
  pred.json:<my_open_dev_uid> \
  data.json.bz2:0x110deb430b3d46459099462ea65ceaf1 \
  --- python evaluation.py --fpred pred.json --fdata data.json.bz2 --fout results.json
```

For example, to run the example submissions files, the closed setting command is:

```bash
cl upload -n closed_binary_dpr_nl.dev.json  # return <my_closed_dev_uid>
cl run -n closed_dev_binary_dpr_nl -d "BART large binary classifier w/ DPR by Victor Zhong at University of Washington" \
  --request-docker-image vzhong/romqa:0.1 \
  --request-memory 8g \
  evaluation.py:0x627bae34595e4bf4971197c9cb917f5e \
  pred.json:<my_closed_dev_uid> \
  data.json.bz2:0x110deb430b3d46459099462ea65ceaf1 \
  --- python evaluation.py --fpred pred.json --fdata data.json.bz2 --fout results.json
```

This should give you a resulting bundle with your open dev evaluation results with the UID
`<my_closed_dev_eval_results_uid>`.
Similarly, the open setting command is:

```bash
cl upload -n open_dev_seq2seq_dpr_nl.dev.json  # return <my_open_dev_uid>
cl run -n open_seq2seq_dpr_nl -d "BART large seq2seq w/ DPR by Victor Zhong at University of Washington" \
  --request-docker-image vzhong/romqa:0.1 \
  --request-memory 8g \
  evaluation.py:0x627bae34595e4bf4971197c9cb917f5e \
  pred.json:<my_open_dev_uid> \
  data.json.bz2:0x110deb430b3d46459099462ea65ceaf1 \
  --- python evaluation.py --fpred pred.json --fdata data.json.bz2 --fout results.json
```

This should give you a resulting bundle with your open dev evaluation results with the UID
`<my_open_dev_eval_results_uid>`.

Once you see that the resulting bundle contains the results you expect, upload your test predictions to CodaLab.
At this point, you should have six bundles.
Please make a Github pull request to modify `submissions.md` and add your entry [as follows](https://github.com/facebookresearch/romqa/pull/2).
The RoMQA authors will then evaluate your test bundles against the test data.
Once the evaluation is finished, the bundle IDs of `<my_closed_test_eval_results_uid>` and `<my_open_test_eval_results_uid>` will be added to your pull request and the pull request will be merged.
Your results will then be displayed on the leaderboard.


### Limitation of test submissions

Each group may submit at most once a month.
We will verify your submission history with the author list of your manuscript.
Authors that abuse the test submission system will be delisted from the leaderboard.


### Anonymous submissions

If you must maintain anonymity (e.g. for submission of a manuscript), please put `anonymous` for your CodaLab names and affiliations.
Then, please email `victor@victorzhong.com` with the same (non-anonymous) information you would put in your pull request, and I will email you back your results.
I will then make the same pull request with my own account, but with anonymous identifying information.
Once you with to make your results public, you can make another pull request to remove anonimity.


To summarize, the steps for test submission are:
1. Produce test and dev prediction JSONs.
2. Create an account on CodaLab
3. Upload predicitions
4. Run dev evaluations
5. Create pull request to initiate test evalutation


## Licence
The majority of RoMQA is licensed under CC-BY-NC, however portions of the project are available under separate license terms:

- [qwickidata](https://github.com/kensho-technologies/qwikidata): Apache 2.0
- [hydra-core](https://github.com/facebookresearch/hydra): MIT
- [torch](https://github.com/pytorch/pytorch): [link](https://github.com/pytorch/pytorch/blob/master/LICENSE)
- [tqdm](https://github.com/tqdm/tqdm): MIT
- [rank_bm25](https://github.com/dorianbrown/rank_bm25): Apache 2.0
- [spacy](https://github.com/explosion/spaCy): MIT
- [sentence_transformers](https://github.com/UKPLab/sentence-transformers): Apache 2.0
- [ray](https://github.com/ray-project/ray): Apache 2.0
- [wrangl](https://github.com/vzhong/wrangl): Apache 2.0
- [ujson](https://github.com/ultrajson/ultrajson): [link](https://github.com/ultrajson/ultrajson/blob/main/LICENSE.txt)
