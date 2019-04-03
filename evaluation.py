import os
import json
import pickle

import numpy as np
import pandas as pd

from nltk import word_tokenize

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import ijson

from sentiments import (
  parse_sentiwordnet,
  parse_mpqa,
  parse_senticnet,
  parse_vader,
  parse_huliu,
  parse_general_inquirer,
  parse_vae,
  FPATHS,
)

imdb_dir = './data/sentiment-data/imdb'
imdb_train = f'{imdb_dir}/train'
imdb_test = f'{imdb_dir}/train'

yelp_dir = './data/sentiment-data/yelp'
yelp_train = f'{yelp_dir}/yelp_academic_dataset_review_train.json'
yelp_test = f'{yelp_dir}/yelp_academic_dataset_review_test.json'

semeval_dir = './data/sentiment-data/SemEval-2017-Task4A'
semeval_train = f'{semeval_dir}/SemEval2017-task4-dev.subtask-A.english.INPUT_train.txt'
semeval_test = f'{semeval_dir}/SemEval2017-task4-dev.subtask-A.english.INPUT_test.txt'

multidom_dir = './data/sentiment-data/multi-domain-sentiment'
multidom_train = f'{multidom_dir}/multi-domain-sentiment_indomain_train.txt'
multidom_test = f'{multidom_dir}/multi-domain-sentiment_indomain_test.txt'

peerread_acl_dir = './data/sentiment-data/PeerRead/acl_2017'
peerread_acl_train = f'{peerread_acl_dir}/train'
peerread_acl_test = f'{peerread_acl_dir}/test'

peerread_iclr_dir = './data/sentiment-data/PeerRead/iclr_2017'
peerread_iclr_train = f'{peerread_iclr_dir}/train'
peerread_iclr_test = f'{peerread_iclr_dir}/test'

SPLIT_SEED = 11235
SPLIT = False
SAVE = True


def save_object(obj, fpath):
    """
    Pickle an object and save it to file
    """
    with open(fpath, 'wb') as o:
        pickle.dump(obj, o)

def load_object(fpath):
    """
    Load a pickled object from file
    """
    with open(fpath, 'rb') as i:
        return pickle.load(i)


def split_file(dir, in_fname, split_prop, seed=None):
  """
  Split train and test data
  """
  np.random.seed(seed)
  fname, ext = os.path.splitext(in_fname)
  with open(f'{dir}/{in_fname}', 'r', encoding='utf-8') as f,\
       open(f'{dir}/{fname}_train{ext}', 'w', encoding='utf-8') as train,\
       open(f'{dir}/{fname}_test{ext}', 'w', encoding='utf-8') as test:
    for i, line in enumerate(f):
      print(i, end='\r')
      train_split = np.random.uniform() < split_prop
      if train_split:
        train.write(line)
      else:
        test.write(line)


def split_files(dir, split_prop, seed=None):
  """
  Split train and test data
  """
  np.random.seed(seed)

  datadir = os.listdir(dir)
  for fold in datadir:
      if not fold in ["books", "dvd", "electronics", "kitchen"]:
          continue
      split2label = {"negative.review": "0", "positive.review": "1"}
      for datasplit in ["negative.review", "positive.review"]:
          in_fname = os.path.join(dir, fold, datasplit)
          fname = os.path.join(dir, "multi-domain-sentiment_indomain")
          ext = ".txt"
          with open(f'{in_fname}', 'r', encoding='utf-8') as f,\
               open(f'{fname}_train{ext}', 'a', encoding='utf-8') as train,\
               open(f'{fname}_test{ext}', 'a', encoding='utf-8') as test:
            for i, line in enumerate(f):
                print(i, end='\r')
                line = line.strip("\n")
                toks = line.split(" ")
                toks_final = []
                for tok in toks:
                    if "_" in tok or tok.startswith("#label"):
                        continue
                    toks_final.append(tok.split(":")[0])
                text = " ".join(toks_final)

                train_split = np.random.uniform() < split_prop
                if train_split:
                    train.write(text + "\t" + split2label[datasplit] + "\n")
                else:
                    test.write(text + "\t" + split2label[datasplit] + "\n")


if SPLIT:
  split_file(yelp_dir, 'yelp_academic_dataset_review.json', split_prop=0.8, seed=SPLIT_SEED)
  split_file(semeval_dir, 'SemEval2017-task4-dev.subtask-A.english.INPUT.txt', split_prop=0.8, seed=SPLIT_SEED)
  split_files(multidom_dir, split_prop=0.8, seed=SPLIT_SEED)


def gen_imdb_data(dir, sent_data, score_fn, limit_to=None):
    """
    Create imdb dataset from sentiment lexicon
    """

    pos_data = [('pos', fname) for fname in os.listdir(os.path.join(dir, 'pos'))][:limit_to]
    neg_data = [('neg', fname) for fname in os.listdir(os.path.join(dir, 'neg'))][:limit_to]

    n = len(pos_data) + len(neg_data)

    y = np.concatenate([np.ones(len(pos_data)), np.zeros(len(neg_data))])
    x = np.zeros((n, len(score_fn('good bad', sent_data))))

    for i, (sent, fname) in enumerate(pos_data + neg_data):
        with open(os.path.join(dir, sent, fname), 'r', encoding='latin1') as textfile:
            text = textfile.read()
            sent_score = score_fn(text, sent_data)

        x[i] = sent_score
        print(f'{i/n*100:0.2f}%', end='\r')
    return x, y


def gen_yelp_data(fpath, sent_data, score_fn, limit_to, balance=False):
    """
    Create yelp dataset from sentiment lexicon

    All written under the assumption that we're never going to read in all data
    """
    n = limit_to
    y = np.zeros(n)
    x = np.zeros((n, len(score_fn('good bad', sent_data))))
    i = 0

    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            sent_score = score_fn(data['text'], sent_data)

            if balance and (np.sum(y == data['stars']) >= (limit_to // 5)):
                continue

            x[i] = sent_score
            y[i] = data['stars']
            i += 1

            print(f'{i/n*100:0.2f}%', end='\r')
            if i >= n:
                break

    return x, y


def gen_multidom_data(fpath, sent_data, score_fn, limit_to=None):
    """
    Create multi-domain sentiment analysis dataset from sentiment lexicon
    """

    n = limit_to

    y = np.zeros(n)
    x = np.zeros((n, len(score_fn('good bad', sent_data))))

    i = 0

    with open(fpath, 'r', encoding='utf-8') as f:
        for line in f:
            text, label = line.strip("\n").split("\t")
            sent_score = score_fn(text, sent_data)

            x[i] = sent_score
            y[i] = int(label)
            i += 1

            print(f'{i/n*100:0.2f}%', end='\r')
            if i >= n:
                break

    return x, y



def gen_semeval_data(fpath, sent_data, score_fn, limit_to=None, balance=False):
    """
    Create yelp dataset from sentiment lexicon
    """

    data = pd.read_csv(fpath, sep='\t', names=['id', 'sent', 'text', '_'], encoding='utf-8')
    data['sent'] = data.sent.replace({'negative': 0, 'neutral': 1, 'positive': 2})

    n = limit_to or len(data)
    if balance:
        class_counts = data.groupby('sent')['id'].count()
        n = class_counts.min() * len(class_counts)

    y = np.zeros(n)
    x = np.zeros((n, len(score_fn('good bad', sent_data))))

    i = 0
    for _, row in data.iterrows():
        sent_score = score_fn(row.text, sent_data)

        if balance and (np.sum(y == row.sent) >= class_counts.min()):
            continue

        x[i] = sent_score
        y[i] = row.sent
        i += 1

        print(f'{i/n*100:0.2f}%', end='\r')
        if i >= n:
            break

    return x, y


def gen_acl_data(dir, sent_data, score_fn, limit_to=None, merge=True):
    """
    Create PeerReview ACL dataset from sentiment lexicon
    """

    acl_data = [fname for fname in os.listdir(os.path.join(dir, 'reviews'))][:limit_to]

    n = limit_to or len(acl_data)

    y = np.zeros(n)
    x = np.zeros((n, len(score_fn('good bad', sent_data))))

    if merge:
        score2norm = {"1": 0, "2": 0, "3": 1, "4": 2, "5": 2, "6": 2}
    else:
        score2norm = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5}
    
    largecnt = 0

    for i, fname in enumerate(acl_data):
        currpath = os.path.join(dir, 'reviews', fname)
        f = open(currpath, encoding="utf-8")
        objects = ijson.items(f, 'reviews')
        for ii, obj in enumerate(objects):
            for j, objj in enumerate(obj):
                text = objj["comments"]
                sent_score = score_fn(text, sent_data)

                x[i] = sent_score
                y[i] = score2norm[objj["RECOMMENDATION"]]
                largecnt += 1
        print(f'{i/n*100:0.2f}%', end='\r')
    print(largecnt)
    return x, y


def gen_iclr_data(dir, sent_data, score_fn, limit_to=None, merge=True):
    """
    Create PeerReview ICLR dataset from sentiment lexicon
    """

    iclr_data = [fname for fname in os.listdir(os.path.join(dir, 'reviews'))][:limit_to]

    n = limit_to or len(iclr_data)

    y = np.zeros(n)
    x = np.zeros((n, len(score_fn('good bad', sent_data))))

    if merge:
        score2norm = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 1, "6": 2, "7": 2, "8": 2, "9": 2, "10": 2}
    else:
        score2norm = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8, "10": 9}

    largecnt = 0

    for i, fname in enumerate(iclr_data):
        currpath = os.path.join(dir, 'reviews', fname)
        f = open(currpath, encoding="utf-8")
        objects = ijson.items(f, 'reviews')
        for ii, obj in enumerate(objects):
            for j, objj in enumerate(obj):
                # some are meta-reviews without scores
                if not "RECOMMENDATION" in objj.keys():
                    continue
                text = objj["comments"]
                sent_score = score_fn(text, sent_data)

                x[i] = sent_score
                y[i] = score2norm[str(objj["RECOMMENDATION"])]
                largecnt += 1
        print(f'{i/n*100:0.2f}%', end='\r')
    print(largecnt)
    return x, y


def score_sent(text, sent_data, normalize=False):
    """
    Evaluate the data
    """
    test_sent = next(iter(sent_data.values()))
    sents = np.zeros_like(test_sent).astype(np.float).reshape(-1)

    tokens = word_tokenize(text.lower())
    for token in tokens:
        try:
            sent = np.array(sent_data[token])
        except KeyError:
            continue
        if normalize:
            sent = sent / sent.sum()

        sents += sent

    score = sents / len(tokens)
    return score


def read_lexica():
    sent_to_dict = lambda x: x.set_index("word")["sent"].to_dict()

    sentiments = {
        'vae_3': sent_to_dict(parse_vae(
            './models/vae/sent_dict.csv',
            sent_cols=[f'alpha_{i}' for i in range(1, 4)],
            from_vae_only=True,
        )),
        'sentiwordnet': sent_to_dict(parse_sentiwordnet(FPATHS['sentiwordnet'], group=True)),
        'mpqa': sent_to_dict(parse_mpqa(FPATHS['mpqa'])),
        'senticnet': sent_to_dict(parse_senticnet(FPATHS['senticnet'])),
        'huliu': sent_to_dict(parse_huliu(*FPATHS['huliu'])),
        'gi': sent_to_dict(parse_general_inquirer(FPATHS['general_inquirer'])),

        'vader_mean': sent_to_dict(parse_vader(FPATHS['vader'], group_mean=True)),
        'vader_multi': sent_to_dict(parse_vader(FPATHS['vader'], group_multinomial=True)),
    }

    # binned vader dataset
    sentiments['vader_binned'] = {
        k: np.array([v[:4].sum(), v[4].sum(), v[5:].sum()])
        for k, v in sentiments['vader_multi'].items()
    }
    print("VADER")
    print(len(sentiments['vader_multi']))

    return sentiments


def score_sentences(sentiments):
    if SAVE:
        imdb, yelp, semeval, multidom, acl3c, acl, iclr3c, iclr, \
        imdb_testd, yelp_testd, semeval_testd, multidom_testd, acl3c_testd, acl_testd, iclr3c_testd, iclr_testd \
            = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

        for lexicon in sentiments:
            print(f'On lexicon {lexicon}')
            if 'vae' in lexicon:
                scorer = lambda text, sent_data, : score_sent(text, sent_data, normalize=False)
            else:
                scorer = lambda text, sent_data, : score_sent(text, sent_data, normalize=False)

            print('IMDB')
            imdb[lexicon], imdb['y'] = gen_imdb_data(
                imdb_train, sentiments[lexicon], scorer, limit_to=None
            )

            print('Yelp')
            yelp[lexicon], yelp['y'] = gen_yelp_data(
                yelp_train, sentiments[lexicon], scorer, limit_to=int(1e5), balance=True
            )

            print('SemEval')
            semeval[lexicon], semeval['y'] = gen_semeval_data(
                semeval_train, sentiments[lexicon], scorer, limit_to=None, balance=True
            )

            print('MultiDom')
            multidom[lexicon], multidom['y'] = gen_multidom_data(
                multidom_train, sentiments[lexicon], scorer, limit_to=6500
            )

            print('ACL3c')
            acl3c[lexicon], acl3c['y'] = gen_acl_data(
                peerread_acl_train, sentiments[lexicon], scorer, limit_to=248, merge=True,
            )

            print('ACL')
            acl[lexicon], acl['y'] = gen_acl_data(
                peerread_acl_train, sentiments[lexicon], scorer, limit_to=248, merge=False,
            )

            print('ICLR3c')
            iclr3c[lexicon], iclr3c['y'] = gen_iclr_data(
                peerread_iclr_train, sentiments[lexicon], scorer, limit_to=2166, merge=True,
            )

            print('ICLR')
            iclr[lexicon], iclr['y'] = gen_iclr_data(
                peerread_iclr_train, sentiments[lexicon], scorer, limit_to=2166, merge=False,
            )

            print('ICLR test')
            iclr_testd[lexicon], iclr_testd['y'] = gen_iclr_data(
                peerread_iclr_test, sentiments[lexicon], scorer, limit_to=230, merge=False,
            )

            print('ICLR3C test')
            iclr3c_testd[lexicon], iclr3c_testd['y'] = gen_iclr_data(
                peerread_iclr_test, sentiments[lexicon], scorer, limit_to=230, merge=True,
            )

            print('ACL test')
            acl_testd[lexicon], acl_testd['y'] = gen_acl_data(
                peerread_acl_test, sentiments[lexicon], scorer, limit_to=15, merge=False,
            )

            print('ACL3C test')
            acl3c_testd[lexicon], acl3c_testd['y'] = gen_acl_data(
                peerread_acl_test, sentiments[lexicon], scorer, limit_to=15, merge=True,
            )

            print('Multidom test')
            multidom_testd[lexicon], multidom_testd['y'] = gen_multidom_data(
                multidom_test, sentiments[lexicon], scorer, limit_to=1575
            )

            print('SemEval test')
            semeval_testd[lexicon], semeval_testd['y'] = gen_semeval_data(
                semeval_test, sentiments[lexicon], scorer, limit_to=None
            )

            print('IMDB test')
            imdb_testd[lexicon], imdb_testd['y'] = gen_imdb_data(
                imdb_test, sentiments[lexicon], scorer, limit_to=None
            )

            print('Yelp test')
            yelp_testd[lexicon], yelp_testd['y'] = gen_yelp_data(
                yelp_test, sentiments[lexicon], scorer, limit_to=int(1e5)#int(1199429)
            )


        save_object(imdb, './models/evaluations/imdb.pkl')
        save_object(yelp, './models/evaluations/yelp.pkl')
        save_object(semeval, './models/evaluations/semeval.pkl')
        save_object(multidom, './models/evaluations/multidom.pkl')
        save_object(acl3c, './models/evaluations/acl_3class.pkl')
        save_object(acl, './models/evaluations/acl.pkl')
        save_object(iclr3c, './models/evaluations/iclr_3class.pkl')
        save_object(iclr, './models/evaluations/iclr.pkl')
        save_object(iclr_testd, './models/evaluations/iclr_test.pkl')
        save_object(iclr3c_testd, './models/evaluations/iclr3c_test.pkl')
        save_object(acl_testd, './models/evaluations/acl_test.pkl')
        save_object(acl3c_testd, './models/evaluations/acl3c_test.pkl')
        save_object(multidom_testd, './models/evaluations/multidom_test.pkl')
        save_object(semeval_testd, './models/evaluations/semeval_test.pkl')
        save_object(yelp_testd, './models/evaluations/yelp_test.pkl')
        save_object(imdb_testd, './models/evaluations/imdb_test.pkl')


    else:
        imdb = load_object('./models/evaluations/imdb.pkl')
        yelp = load_object('./models/evaluations/yelp.pkl')
        semeval = load_object('./models/evaluations/semeval.pkl')
        multidom = load_object('./models/evaluations/multidom.pkl')
        acl3c = load_object('./models/evaluations/acl_3class.pkl')
        acl = load_object('./models/evaluations/acl.pkl')
        iclr3c = load_object('./models/evaluations/iclr_3class.pkl')
        iclr = load_object('./models/evaluations/iclr.pkl')

        imdb_testd = load_object('./models/evaluations/imdb_test.pkl')
        yelp_testd = load_object('./models/evaluations/yelp_test.pkl')
        semeval_testd = load_object('./models/evaluations/semeval_test.pkl')
        multidom_testd = load_object('./models/evaluations/multidom_test.pkl')
        acl3c_testd = load_object('./models/evaluations/acl3c_test.pkl')
        acl_testd = load_object('./models/evaluations/acl_test.pkl')
        iclr3c_testd = load_object('./models/evaluations/iclr3c_test.pkl')
        iclr_testd = load_object('./models/evaluations/iclr_test.pkl')

    return imdb, yelp, semeval, multidom, acl3c, acl, iclr3c, iclr, \
           imdb_testd, yelp_testd, semeval_testd, multidom_testd, acl3c_testd, acl_testd, iclr3c_testd, iclr_testd


def make_combined_score(sentiments, imdb, yelp, semeval, multidom, acl3c, acl, iclr3c, iclr, imdb_testd, yelp_testd, semeval_testd, multidom_testd, acl3c_testd, acl_testd, iclr3c_testd, iclr_testd):
    # which datasets do we *not* want in the combined version?
    exclude = ['vae_3', 'vae_5', 'vae_9', 'vader_mean', 'vader_binned', 'combined', 'combined_binned']

    imdb['combined'] = np.hstack([imdb[lexicon] for lexicon in sentiments if lexicon not in exclude])
    yelp['combined'] = np.hstack([yelp[lexicon] for lexicon in sentiments if lexicon not in exclude])
    semeval['combined'] = np.hstack([semeval[lexicon] for lexicon in sentiments if lexicon not in exclude])
    multidom['combined'] = np.hstack([multidom[lexicon] for lexicon in sentiments if lexicon not in exclude])
    acl3c['combined'] = np.hstack([acl3c[lexicon] for lexicon in sentiments if lexicon not in exclude])
    acl['combined'] = np.hstack([acl[lexicon] for lexicon in sentiments if lexicon not in exclude])
    iclr3c['combined'] = np.hstack([iclr3c[lexicon] for lexicon in sentiments if lexicon not in exclude])
    iclr['combined'] = np.hstack([iclr[lexicon] for lexicon in sentiments if lexicon not in exclude])
    imdb_testd['combined'] = np.hstack([imdb_testd[lexicon] for lexicon in sentiments if lexicon not in exclude])
    yelp_testd['combined'] = np.hstack([yelp_testd[lexicon] for lexicon in sentiments if lexicon not in exclude])
    semeval_testd['combined'] = np.hstack([semeval_testd[lexicon] for lexicon in sentiments if lexicon not in exclude])
    multidom_testd['combined'] = np.hstack([multidom_testd[lexicon] for lexicon in sentiments if lexicon not in exclude])
    acl3c_testd['combined'] = np.hstack([acl3c_testd[lexicon] for lexicon in sentiments if lexicon not in exclude])
    acl_testd['combined'] = np.hstack([acl_testd[lexicon] for lexicon in sentiments if lexicon not in exclude])
    iclr3c_testd['combined'] = np.hstack([iclr3c_testd[lexicon] for lexicon in sentiments if lexicon not in exclude])
    iclr_testd['combined'] = np.hstack([iclr_testd[lexicon] for lexicon in sentiments if lexicon not in exclude])

    sentiments['combined'] = None  # dummy such that it's included in iterations

    return sentiments, imdb, yelp, semeval, multidom, acl3c, acl, iclr3c, iclr, \
           imdb_testd, yelp_testd, semeval_testd, multidom_testd, acl3c_testd, acl_testd, iclr3c_testd, iclr_testd


def make_binned_yelp(sentiments, yelp):
    yelp_binned = {}

    neutral = np.where(yelp['y'] == 3)[0]
    n = neutral.shape[0]
    neg = np.where(np.isin(yelp['y'], [1, 2]))[0][:n]
    pos = np.where(np.isin(yelp['y'], [4, 5]))[0][:n]

    yelp_binned_idx = np.concatenate([neutral, pos, neg])

    y = yelp['y'][yelp_binned_idx]
    yelp_binned['y'] = (y == 3) * 1 + (np.isin(y, [4, 5])) * 2

    #print(sentiments.keys())
    for lexicon in sentiments:
        #print(lexicon)
        yelp_binned[lexicon] = yelp[lexicon][yelp_binned_idx]

    return yelp_binned


def eval_all(sentiments, imdb, yelp, yelp_binned, semeval, multidom, acl3c, acl, iclr3c, iclr,
             imdb_testd, yelp_testd, yelp3c_testd, semeval_testd, multidom_testd, acl3c_testd, acl_testd, iclr3c_testd, iclr_testd):

    evaluation_data = (
      ('imdb', imdb, imdb_testd),
      ('yelp', yelp, yelp_testd),
      ('yelp_binned', yelp_binned, yelp3c_testd),
      ('semeval', semeval, semeval_testd),
      ('multidom', multidom, multidom_testd),
      ('acl3c', acl3c, acl3c_testd),
      ('acl', acl, acl_testd),
      ('iclr3c', iclr3c, iclr3c_testd),
      ('iclr', iclr, iclr_testd),
    )

    for eval_name, train_data, eval_data in evaluation_data:
        print(f'{eval_name} accuracy - {len(np.unique(train_data["y"]))} classes')
        for lexicon in list(sentiments.keys()):
            # making splits here
            #x_train, x_dev, y_train, y_dev = train_test_split(
            #    eval_data[lexicon],
            #    eval_data['y'],
            #    random_state=SPLIT_SEED,
            #    test_size=0.1,
            #)
            # separate test
            x_train = train_data[lexicon]
            y_train = train_data['y']
            x_dev = eval_data[lexicon]
            y_dev = eval_data['y']
            # print(len(y_train))

            logit = LogisticRegression()
            logit.fit(x_train, y_train)
            pred = logit.predict(x_dev)
            metric = np.mean(pred == y_dev)

            print(f'{lexicon:15}{metric:0.3f}')
        print('\n')



if __name__ == '__main__':
    sentiments = read_lexica()
    imdb, yelp, semeval, multidom, acl3c, acl, iclr3c, iclr, imdb_testd, \
        yelp_testd, semeval_testd, multidom_testd, acl3c_testd, acl_testd, iclr3c_testd, iclr_testd \
        = score_sentences(sentiments)
    sentiments, imdb, yelp, semeval, multidom, acl3c, acl, iclr3c, iclr, \
        imdb_testd, yelp_testd, semeval_testd, multidom_testd, acl3c_testd, acl_testd, iclr3c_testd, iclr_testd \
        = make_combined_score(sentiments, imdb, yelp, semeval, multidom, acl3c, acl, iclr3c, iclr, imdb_testd, yelp_testd, semeval_testd, multidom_testd, acl3c_testd, acl_testd, iclr3c_testd, iclr_testd)
    yelp_binned = make_binned_yelp(sentiments, yelp)
    yelp3c_testd = make_binned_yelp(sentiments, yelp_testd)
    eval_all(sentiments, imdb, yelp, yelp_binned, semeval, multidom, acl3c, acl, iclr3c, iclr,
             imdb_testd, yelp_testd, yelp3c_testd, semeval_testd, multidom_testd, acl3c_testd, acl_testd, iclr3c_testd, iclr_testd)
