import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import pyro
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist

import evaluation as se
from sentiments import read_all_sentiment_data, FPATHS


class FullDecoder(nn.Module):
    '''
    From latent score to label
    '''
    def __init__(self, latent_dim, hidden=16):
        super().__init__()
        self.linear_mpqa1 = nn.Linear(latent_dim, hidden)
        self.linear_mpqa2 = nn.Linear(hidden, 1)

        self.linear_huliu1 = nn.Linear(latent_dim, hidden)
        self.linear_huliu2 = nn.Linear(hidden, 1)

        self.linear_inquirer1 = nn.Linear(latent_dim, hidden)
        self.linear_inquirer2 = nn.Linear(hidden, 1)

        self.linear_vader1 = nn.Linear(latent_dim, hidden)
        self.linear_vader2 = nn.Linear(hidden, 9)

        self.linear_senticnet1 = nn.Linear(latent_dim, hidden)
        self.linear_senticnet_loc = nn.Linear(hidden, 1)
        self.linear_senticnet_scale = nn.Linear(hidden, 2)

        self.linear_sentiwordnet1 = nn.Linear(latent_dim, hidden)
        self.linear_sentiwordnet_loc = nn.Linear(hidden, 2)
        self.linear_sentiwordnet_scale = nn.Linear(hidden, 1)
        
        self.elu = nn.ELU()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z, dataset):
        '''
        Take a latent draw as input, and output rho to parameterize the emission dists
        '''
        if dataset == 'mpqa':
            hidden = self.elu(self.linear_mpqa1(z))
            return self.sigmoid(self.linear_mpqa2(hidden)) # params a bernoulli
            
        if dataset == 'huliu':
            hidden = self.elu(self.linear_huliu1(z))
            return self.sigmoid(self.linear_huliu2(hidden)) # params a bernoulli

        if dataset == 'general_inquirer':
            hidden = self.elu(self.linear_inquirer1(z))
            return self.sigmoid(self.linear_inquirer2(hidden)) # params a bernoulli

        if dataset == 'vader':
            hidden = self.elu(self.linear_vader1(z))
            return self.softmax(self.linear_vader2(hidden)) # params a categorical 

        if dataset == 'senticnet':
            hidden = self.elu(self.linear_senticnet1(z))
            loc = self.tanh(self.linear_senticnet_loc(hidden))
            scale = self.relu(self.linear_senticnet_scale(hidden)) + 0.001
            return loc, scale # params a normal

        if dataset == 'sentiwordnet':
            hidden = self.elu(self.linear_sentiwordnet1(z))
            loc = self.tanh(self.linear_sentiwordnet_loc(hidden))
            scale = torch.tensor([[0.01, 0.], [0., 0.01]])
            return loc, scale # params a multivariate normal


class FullEncoder(nn.Module):
    '''
    From a sentiment label to an omega (which are summed to form beta)
    '''
    def __init__(
        self, latent_dim, hidden=16, output_activation=None, vader_multinomial=False
        ):
        super().__init__()

        self.linear_mpqa1 = nn.Linear(1, hidden)
        self.linear_mpqa2 = nn.Linear(hidden, latent_dim)
        
        self.linear_huliu1 = nn.Linear(1, hidden)
        self.linear_huliu2 = nn.Linear(hidden, latent_dim)
        
        self.linear_inquirer1 = nn.Linear(1, hidden)
        self.linear_inquirer2 = nn.Linear(hidden, latent_dim)

        vader_dim = 9 if vader_multinomial else 1
        self.linear_vader1 = nn.Linear(vader_dim, hidden)
        self.linear_vader2 = nn.Linear(hidden, latent_dim)
        
        self.linear_senticnet1 = nn.Linear(1, hidden)
        self.linear_senticnet2 = nn.Linear(hidden, latent_dim)

        self.linear_sentiwordnet1 = nn.Linear(2, hidden)
        self.linear_sentiwordnet2 = nn.Linear(hidden, latent_dim)

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.output_activation = output_activation

    def forward(self, sent, dataset):
        '''
        Take a sentiment label as input, and produce an omega term
        '''

        if dataset == 'mpqa':
            hidden = self.elu(self.linear_mpqa1(sent))
            return self.output_activation(self.linear_mpqa2(hidden))
        if dataset == 'huliu':
            hidden = self.elu(self.linear_huliu1(sent))
            return self.output_activation(self.linear_huliu2(hidden))
        if dataset == 'general_inquirer':
            hidden = self.elu(self.linear_inquirer1(sent))
            return self.output_activation(self.linear_inquirer2(hidden))
        if dataset == 'vader':
            hidden = self.elu(self.linear_vader1(sent))
            return self.output_activation(self.linear_vader2(hidden))
        if dataset == 'senticnet':
            hidden = self.elu(self.linear_senticnet1(sent))
            return self.output_activation(self.linear_senticnet2(hidden))
        if dataset == 'sentiwordnet':
            hidden = self.elu(self.linear_sentiwordnet1(sent))
            return self.output_activation(self.linear_sentiwordnet2(hidden))


class FullVAE(nn.Module):
    def __init__(
        self,
        vocab,
        word_counts,
        latent_dim=3,
        hidden=16,
        smoothing=1.,
        encoder_activation=None,
        vader_multinomial=False,
        prime_priors=True,
    ):
        super().__init__()
        self.vocab = vocab
        self.latent_dim = latent_dim
        self.smoothing = smoothing
        self.vader_multinomial = vader_multinomial

        self.encoder = FullEncoder(
            latent_dim,
            hidden=hidden,
            output_activation=encoder_activation,
            vader_multinomial=vader_multinomial,
        )
        self.decoder = FullDecoder(latent_dim, hidden=hidden)

        self.word_counts = word_counts
        self.betas = None
        self.alpha_prior = torch.tensor(
            np.full((len(word_counts), latent_dim), smoothing), 
            dtype=torch.float
        )

        # goose the priors
        if prime_priors:
            self.assign_priors(
                [
                'terrorism', 'terrorist', 'slavery', 'rape', 'kill', 'murder',
                'brutal', 'evil', 'abusive', 'cancer', 'sickly',
                ],
                0
            )
            self.assign_priors(
                [
                'foetal', 'portuguese', 'attic', 'freeway', 'reach',
                'warsaw', 'backed', 'killick', 'drink', 'tangerine',
                ],
                latent_dim // 2
            )
            self.assign_priors(
                [
                'superb', 'sensational', 'reputable', 'spiffing', 'freedom',
                'charming', 'ilu', 'amazement', 'humility', 'flawless',
                ],
                latent_dim - 1
            )

    def assign_priors(self, words, sent_idx):
        '''
        For a given set of words and an index, assign all counts to that element in
        the prior parameters
        '''
        for word in words:
            word_idx = self.vocab[word]
            self.alpha_prior[word_idx, :] = self.smoothing
            self.alpha_prior[word_idx, sent_idx] = (
                self.smoothing + self.word_counts[word_idx]
            )
            
    def model(self, data):
        '''
        The generative distribution
        '''
        pyro.module("decoder", self.decoder)

        # sample all the priors simulaneously
        with pyro.iarange("score_sample", len(self.vocab)):
            z = pyro.sample(f'latent_scores',
                dist.Dirichlet(self.alpha_prior),
            )
 
        datasets = data.source.unique()
        # loop through the datasets
        for i in pyro.irange("data_loop", len(datasets)):
            dataset = datasets[i]
            subset = data.loc[data.source == dataset]

            sent = torch.tensor(subset.sent.values.tolist(), dtype=torch.float)
            if len(sent.shape) == 1:
               sent = sent.unsqueeze(-1)
            z_word = z[subset.word_id.values]
            rho = self.decoder.forward(z_word, dataset)

            if dataset in ['mpqa', 'huliu', 'general_inquirer']:
                pyro.sample(f"obs_{dataset}", dist.Bernoulli(rho), obs=sent)

            if dataset == 'vader':
                if self.vader_multinomial:
                    pyro.sample(
                        f"obs_{dataset}",
                        dist.Multinomial(probs=rho, total_count=10),
                        obs=sent,
                    )
                else:
                    n = rho.size(0)
                    batch = n // 20
                    for j in pyro.irange("vader_chunks", 20):
                        pyro.sample(
                            f"obs_{dataset}_{j}",
                            dist.Categorical(rho[j*batch:(j+1)*batch,:]),
                            obs=sent + 4.
                        )

            if dataset == 'senticnet':
                loc, scale = rho
                pyro.sample(f"obs_{dataset}", dist.Normal(loc, scale), obs=sent)

            if dataset == 'sentiwordnet':
                loc, scale = rho
                pyro.sample(
                    f"obs_{dataset}", dist.MultivariateNormal(loc, scale),
                    obs=sent
                )

            
    def guide(self, data):
        '''
        The variational distribution
        '''
        pyro.module("encoder", self.encoder)
        
        # These betas are learned in training
        self.betas = torch.zeros((len(self.vocab), self.latent_dim)) + self.smoothing

        # encode the sentiment scores
        datasets = data.source.unique()
        for i in pyro.irange("data_loop", len(datasets)):
            dataset = datasets[i]
            subset = data.loc[data.source == dataset]
            
            sent = torch.tensor(subset.sent.values.tolist(), dtype=torch.float)
            if len(sent.shape) == 1:
                sent = sent.unsqueeze(-1)

            # sum the omegas
            self.betas[subset.word_id.values] += self.encoder.forward(sent, dataset)

        with pyro.iarange("score_sample", len(self.vocab)):
            pyro.sample(
                f"latent_scores", dist.Dirichlet(self.betas)
            )


def generate_sentiment_data(vae, normed=False, as_dict=False, latent_dim=3):
    """
    Create the sentiment dictionary from the learned vae parameters
    """
    sent_dict = {}

    # use parameters learned by the vae 
    for word, word_idx in vae.vocab.items():
        sent_dict[word] = vae.betas[word_idx].detach().numpy()
    
    if as_dict:
        return sent_dict

    data = pd.DataFrame.from_dict(
        sent_dict,
        orient='index',
        columns=[f'alpha_{i}' for i in range(1, latent_dim + 1)],
    )
    data['from_vae'] = True # holdover

    return data


if __name__ == '__main__':
    # read sentiment data
    sentiment_data = read_all_sentiment_data(
        FPATHS, vader_multinomial=True, sentiwordnet_group=True
    )
    
    sentiment_data = sentiment_data[['word', 'source', 'sent']]
    sentiment_data = sentiment_data.sample(frac=1, random_state=101).reset_index(drop=True)
    sentiment_data = sentiment_data.loc[~sentiment_data.word.isnull()]
    word_counts = (
        sentiment_data.groupby('word', as_index=False)
                      .count()[['word', 'source']]
                      .rename(columns={'source': 'counts'})
    )

    words_to_keep = ~word_counts.word.str.contains('_')
    word_counts = word_counts.loc[words_to_keep].reset_index(drop=True)
    sentiment_data = sentiment_data.loc[sentiment_data.word.isin(word_counts.word)]
    sent_vocab = dict(zip(word_counts.word.tolist(), word_counts.index.tolist()))
    sentiment_data['word_id'] = [sent_vocab[word] for word in sentiment_data.word]

    # clear the param store in case we're in a REPL
    pyro.clear_param_store()
    vae = FullVAE(
        vocab=sent_vocab,
        word_counts=word_counts.counts.values,
        latent_dim=3,
        hidden=32,
        encoder_activation=nn.Softmax(dim=-1),
        vader_multinomial=True,
    )

    # setup the optimizer
    adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
    optimizer = Adam(adam_params)

    svi = SVI(vae.model, vae.guide, optimizer, loss=Trace_ELBO())
    # enable validation (e.g. validate parameters of distributions)
    # pyro.enable_validation(True)
 
    n_steps = 10000
    save = True
    losses = []
    accuracies = [0]
    eval_every = 100
    eval_steps_before_stop = 20
    print("Training begins")
    
    for i in range(n_steps):
        losses.append(svi.step(sentiment_data))
        print(f'{i} best acc at {np.argmax(accuracies) * eval_every} '
              f'best loss at {np.argmin(losses)}', end='\r')
        
        if i % eval_every == 0 and i > 0:
            sent_data = generate_sentiment_data(vae, as_dict=True)
            scorer = lambda text, data : se.score_sent(text, data, normalize=False)

            # just evaluate on some of the data
            evaluation_data  = {}
            evaluation_data['yelp'] = se.gen_yelp_data(
                se.yelp_train, sent_data, scorer, limit_to=int(1e4), balance=True
            )
            evaluation_data['semeval'] = se.gen_semeval_data(
                se.semeval_train, sent_data, scorer, limit_to=None, balance=True
            )
            evaluation_data['acl'] = se.gen_acl_data(
                se.peerread_acl_train, sent_data, scorer, limit_to=248, merge=True
            )

            mean_accuracy, n_examples = 0, 0
            for eval_name, eval_data in evaluation_data.items():
                x, y = eval_data
                x_train, x_dev, y_train, y_dev = se.train_test_split(
                    x, y, random_state=11235, test_size=0.1
                )
                logit = se.LogisticRegression()
                logit.fit(x_train, y_train)
                pred = logit.predict(x_dev)
                accuracy = np.mean(pred == y_dev)
                mean_accuracy += accuracy * len(y)
                n_examples += len(y)
                print(f'{eval_name}: {accuracy}')
            
            accuracies.append(mean_accuracy / n_examples)
            print(f'Mean accuracy is {accuracies[-1]}\n')

        if save and i >= 50 and accuracies[-1] >= np.max(accuracies):
            generate_sentiment_data(vae).to_csv('sent_dict.csv')
        
        if np.argmax(accuracies) <= (len(accuracies) - eval_steps_before_stop):
            print(f'{eval_steps_before_stop} steps w/o accuracy improvement, stopping')
            break
