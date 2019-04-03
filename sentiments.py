import re
import json

import numpy as np
import pandas as pd

FPATHS = {
    # sentiments
    'sentiwordnet': './data/sentiments/SentiWordNet_3.0.0_20130122.txt',
    'mpqa': './data/sentiments/subjclueslen1-HLTEMNLP05.tff',
    'senticnet': './data/sentiments/senticnet5.txt',
    'vader': './data/sentiments/vader_lexicon.txt',
    'huliu': [
        './data/sentiments/positive-words.txt',
        './data/sentiments/negative-words.txt',
    ],
    'general_inquirer': './data/sentiments/inquirerbasic.csv',
    
}

def parse_sentiwordnet(fpath, keep_pos=['a'], group=False):
    '''
    Read in and group SentiWordNet data by word, taking the average
    Source: https://sentiwordnet.isti.cnr.it/

    Data format:
        Words have both a real-valued positive and negative score, each in [0, 1]

    Args
        fpath : str
            Filepath of data
        keep_pos : list containing any of 'a', 'n', 'v', 'r'
            Desired POS tags upon which to filter the data
        group : bool
            Either do or do not group data
   '''
    sentiments = pd.read_csv(
        fpath,
        comment='#',
        sep='\t',
        names=['pos', 'id', 's_pos', 's_neg', 'word'],
        keep_default_na=False,
        na_values=''
    ).dropna()
    sentiments = sentiments.loc[sentiments.pos.isin(keep_pos)]
    if group:
        sentiments = (
            sentiments[['word', 's_pos', 's_neg']].groupby('word').mean().reset_index()
        )
    
    # treat as tuple
    sentiments['sent'] = sentiments[['s_pos', 's_neg']].apply(tuple, axis=1)
    return sentiments[['word', 'sent']]

def parse_mpqa(fpath, keep_pos=['adj', 'anypos'], group=False):
    '''
    Read in the MPQA Subjectivity Lexicon
    Souce: http://mpqa.cs.pitt.edu/lexicons/subj_lexicon/

    Data format:
        Words are hard-classified as either positive or negative.

    Args
        fpath : str
            Filepath of data
        keep_pos : list containing any of 'adj', 'noun', 'verb', 'anypos', 'adverb'
            Desired POS tags upon which to filter the data
    '''
    sentiments = pd.read_csv(
        fpath,
        sep=r'\s+',
        names=['type', 'len', 'word', 'pos', 'stemmed', 'sent'],
    )
    # clean up the fields
    clean_regex = {
        re.compile(x): ''
        for x in ['type=', 'len=', 'word1=', 'pos1=', 'stemmed1=', 'priorpolarity=']
    }
    sentiments = sentiments.replace(clean_regex)
    
    # filter to the right pos
    sentiments = sentiments.loc[sentiments.pos.isin(keep_pos)]
    sentiments['sent'] = (sentiments.sent == 'positive') * 1.

    # TODO: double-check that this makes sense (the maximum specifically)
    if group:
        sentiments = (
            sentiments[['word', 'sent']].groupby('word').max().reset_index()
        )

    return sentiments[['word', 'sent']]
    

def parse_senticnet(fpath):
    '''
    Parse the senticnet data
    Source: http://sentic.net/downloads/

    Data format:
        Words are given a real-valued score in (-1, 1)

    Args
        fpath : str
            Filepath of data
    '''
    sentiments = pd.read_csv(fpath, sep=r'\s+')
    sentiments.columns = ['word', 'polarity', 'sent']
    
    # TODO: data is naturally trimodal with cut-points at -0.5 and +0.5 
    # (possibly -0.25, +0.25); should help with 'neutral' designation

    return sentiments[['word', 'sent']]


def parse_vader(fpath, group_mean=False, group_multinomial=False):
    '''
    Parse the VADER data
    Source: https://github.com/cjhutto/vaderSentiment

    Data format:
        Words have 10 scores from human reviewers, integers in {-4, -3, ... 3, 4}, where
        0 is neutral
    '''
    sentiments = pd.read_csv(fpath, sep='\t', names=['word', 'sent', 'sd', 'scores'])
    
    if group_mean:
        return sentiments[['word', 'sent']]
    
    sentiments['sent'] = (sentiments.scores
                        .replace(re.compile(r'[\[\]]'), '')
                        .str
                        .split(',')
                        .apply(lambda x: [int(i) for i in x])
    )
    sentiments = (sentiments['sent'].apply(lambda x: pd.Series(x))
                                    .stack()
                                    .reset_index(level=1, drop=True)
                                    .to_frame('sent')
                                    .join(sentiments[['word']], how='left')
    )
    if group_multinomial:
        grouped = ( 
            sentiments.groupby('word')['sent']
                      .apply(lambda x: np.bincount(np.array(x) + 4, minlength=9) + 0.)
                      .to_frame()
                      .reset_index()
        )
        return grouped.loc[grouped.sent.apply(np.sum) <= 10] # gets rid of some junk
    
    return sentiments[['word', 'sent']]


def parse_huliu(pos_fpath, neg_fpath, group=False):
    '''
    Parse the Hu and Liu KDD-2004 Data
    Source: http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html

    Data format:
        Words are hard-classified as either positive or negative.

    Args:
        pos_fpath: str
            Filepath for positive data
        neg_fpath: str
            Filepath for negative data
    '''
    sent_pos = pd.read_csv(pos_fpath, comment=';', names=['word'], encoding='latin1')
    sent_neg = pd.read_csv(neg_fpath, comment=';', names=['word'], encoding='latin1')
    sent_pos['sent'] = 1.
    sent_neg['sent'] = 0.

    sentiments = pd.concat([sent_pos, sent_neg])

    # TODO: again, make sure this logic holds up
    if group:
        sentiments = (
            sentiments[['word', 'sent']].groupby('word').mean().reset_index()
        )
    return sentiments

def parse_general_inquirer(fpath, group=False):
    '''
    Parse the General Inquirer data
    Source: http://www.wjh.harvard.edu/~inquirer/spreadsheet_guide.htm

    Data format:
        Words are hard-classified as either positive or negative. We exclude words
        missing scores, rather than counting them as neutral #TODO: is this a good idea?

    Args
        fpath : str
            Filepath of data
    '''
    sentiments = pd.read_csv(fpath, usecols=['Entry', 'Positiv', 'Negativ'])
    sentiments.columns = ['word', 's_pos', 's_neg']
    sentiments['word'] = sentiments.word.str.lower()

    not_missing = (~sentiments.s_pos.isnull()) | (~sentiments.s_neg.isnull())
    sentiments = sentiments.loc[not_missing] # Assumption made here
    sentiments['sent'] = (sentiments['s_pos'] == 'Positiv') * 1.

    # this allows us to collapse duplicates
    sentiments['word'] = sentiments.word.replace({re.compile('#[0-9]+'): ''})

    # TODO: again, make sure this logic holds up
    if group:
        sentiments = (
            sentiments[['word', 'sent']].groupby('word').mean().reset_index()
        )
    return sentiments[['word', 'sent']]

def parse_vae(fpath, sent_cols, from_vae_only=False):
    '''
    Parse the learned combined data in the same format as others
        
    Data format:
        Words have a positive, negative, or neutral real-valued score.

    Args
        fpath : str
            Filepath of data
        sent_cols: list of ['alpha_1', 'alpha_2', 'alpha_3']
            How to sort each score -- does 2 correspond to negative, for example
    '''
    sentiments = pd.read_csv(fpath, index_col=0)    
    if from_vae_only:
        sentiments = sentiments.loc[sentiments.from_vae]

    sentiments['sent'] = sentiments[sent_cols].apply(tuple, axis=1)
    sentiments = sentiments.reset_index().rename(columns={'index':'word'})
    return sentiments[['word', 'sent']]

def read_all_sentiment_data(fpaths, vader_multinomial=False, sentiwordnet_group=False):
    
    sentiment_full = pd.concat([
        parse_sentiwordnet(fpaths['sentiwordnet'], group=sentiwordnet_group).assign(source='sentiwordnet'),
        parse_mpqa(fpaths['mpqa']).assign(source='mpqa'),
        parse_senticnet(fpaths['senticnet']).assign(source='senticnet'),
        parse_vader(fpaths['vader'], group_multinomial=vader_multinomial).assign(source='vader'),
        parse_huliu(*fpaths['huliu']).assign(source='huliu'),
        parse_general_inquirer(fpaths['general_inquirer']).assign(source='general_inquirer'),
    ], axis=0, ignore_index=True)
    return sentiment_full


if __name__ == '__main__':
    pass
