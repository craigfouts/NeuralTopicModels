import numpy as np
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
import pyro.ops.stats as stats
import torch
import torch.nn.functional as F
import torch.nn as nn
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, TraceMeanField_ELBO
from pyro.optim import Adam
from scipy.cluster.vq import kmeans, vq
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from torch.nn import Parameter
from tqdm import tqdm

# Utilities

def activation(act='softplus', **kwargs):
    if act == 'softplus':
        return nn.Softplus(**kwargs)
    elif act == 'relu':
        return nn.ReLU(**kwargs)
    elif act == 'sigmoid':
        return nn.Sigmoid(**kwargs)
    elif act == 'softmax':
        return nn.Softmax(dim=-1)
    raise NotImplementedError(f'Activation function"{act}" not supported.')

def mlp(layers, bias=True, act='softplus', final_act=None, batch_norm=True, affine=True, dropout=0.):
    n_layers = len(layers) - 1
    for i in range(1, n_layers + 1):
        yield nn.Linear(layers[i - 1], layers[i], bias=bias)
        if i < n_layers:
            yield activation(act if i < n_layers else final_act)
        else:
            if batch_norm:
                yield nn.BatchNorm1d(layers[i], affine=affine)
            if final_act:
                yield activation(final_act)
            yield nn.Dropout(dropout)

def count(x, vocab_size):
    x_counts = torch.zeros((x.shape[0], vocab_size))
    for i in range(x.shape[0]):
        idx, counts = torch.unique(x[i], return_counts=True)
        for j in range(idx.shape[0]):
            x_counts[i, idx[j].to(torch.int32)] = counts[j].to(torch.int32)
    return x_counts

def build(x, n_words, vocab_size, vocab_steps=10, return_counts=False):
    codebook, _ = kmeans(x[:, 4:], vocab_size, vocab_steps)
    neighbors = NearestNeighbors(n_neighbors=n_words).fit(x[:, :4])
    _, neighbor_idx = neighbors.kneighbors(x[:, :4])
    data = np.zeros((x.shape[0], n_words), dtype=np.int32)
    for i in range(x.shape[0]):
        data[i], _ = vq(x[neighbor_idx, 4:][i], codebook)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    if return_counts:
        return data_tensor, count(data_tensor, vocab_size)
    return data_tensor

def perplexity(log_prob, n_docs, n_words):
    score = torch.exp(-log_prob/(n_docs*n_words))
    return score

def coherence(x, labels):
    n_topics = labels.unique().shape[0]
    total = 0
    for k in range(n_topics):
        docs, = torch.where(labels == k)
        occurrences = (x[docs] != 0).sum(0)
        idx = torch.combinations(torch.arange(occurrences.shape[0]), r=2).T
        co_occurrences = occurrences[idx[0]] + occurrences[idx[1]]
        score = co_occurrences/(occurrences[idx[0]]*occurrences[idx[1]])
        total += torch.nan_to_num(torch.log(score), posinf=0.).sum()
    return total

class GaussianSoftmax(dist.Normal):
    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            return F.softmax(torch.normal(self.loc.expand(shape), self.scale.expand(shape)), dim=-1)

# Basic VAE

class VAEEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        encoding_layers = (input_dim, hidden_dim)
        latent_layers = (hidden_dim, latent_dim)
        self.encoding_net = nn.Sequential(*list(mlp(encoding_layers, final_act='softplus', batch_norm=False, dropout=.2)))
        self.loc_net = nn.Sequential(*list(mlp(latent_layers)))
        self.scale_net = nn.Sequential(*list(mlp(latent_layers)))

    def forward(self, x):
        y = self.encoding_net(x)
        z_loc = self.loc_net(y)
        z_scale = (.5*self.loc_net(y)).exp()
        return z_loc, z_scale

class VAEDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        layers = (latent_dim, hidden_dim, output_dim)
        self.net = nn.Sequential(*list(mlp(layers)))
    
    def forward(self, z):
        x = self.net(z)
        return x
    
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, scale=.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.scale = scale
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)
        self.loss_log_ = []

    def _model(self, x):
        pyro.module('decoder', self.decoder)
        with pyro.plate('data', x.shape[0]):
            z_loc = x.new_zeros((x.shape[0], self.latent_dim))
            z_scale = x.new_ones((x.shape[0], self.latent_dim))
            z = pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1))
            x_loc = self.decoder(z)
            x_scale = self.scale*torch.ones_like(x_loc)
            pyro.sample('obs', dist.Normal(x_loc, x_scale).to_event(1), obs=x)

    def _guide(self, x):
        pyro.module('encoder', self.encoder)
        with pyro.plate('data', x.shape[0]):
            z_loc, z_scale = self.encoder(x)
            pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1))

    def fit(self, X, n_steps=100, learning_rate=1e-2):
        optim = Adam({'lr': learning_rate})
        elbo = Trace_ELBO(max_iarange_nesting=4)
        svi = SVI(self._model, self._guide, optim, elbo)
        for _ in tqdm(range(n_steps)):
            loss = svi.step(X)
            self.loss_log_.append(loss)
        return self
    
    def transform(self, X):
        z_loc, z_scale = self.encoder(X)
        z = dist.Normal(z_loc, z_scale).sample()
        return z
    
# VAE Topic Model

class VAETMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        encoding_layers = (input_dim, hidden_dim, hidden_dim)
        latent_layers = (hidden_dim, latent_dim)
        self.encoding_net = nn.Sequential(*list(mlp(encoding_layers, final_act='softplus', batch_norm=False, dropout=.2)))
        self.loc_net = nn.Sequential(*list(mlp(latent_layers, affine=True)))
        self.scale_net = nn.Sequential(*list(mlp(latent_layers, affine=True)))

    def forward(self, x):
        y = self.encoding_net(x)
        z_loc = self.loc_net(y)
        z_scale = (.5*self.scale_net(y)).exp()
        return z_loc, z_scale

class VAETMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        layers = (latent_dim, hidden_dim, output_dim)
        self.dropout = nn.Dropout(.2)
        self.net = nn.Sequential(*list(mlp(layers, bias=False, final_act='softmax', affine=True)))
    
    def forward(self, z):
        y = self.dropout(z)
        x = self.net(y)
        return x
    
class VAETM(nn.Module):
    def __init__(self, n_topics, n_words, vocab_size, hidden_dim):
        super().__init__()
        self.n_topics = n_topics
        self.n_words = n_words
        self.vocab_size = vocab_size
        self.encoder = VAETMEncoder(vocab_size, hidden_dim, n_topics)
        self.decoder = VAETMDecoder(n_topics, hidden_dim, vocab_size)
        self.X_ = None
        self.loss_log_ = []

    def _model(self, x):
        pyro.module('decoder', self.decoder)
        with pyro.plate('data', x.shape[0]):
            z_loc = x.new_zeros((x.shape[0], self.n_topics))
            z_scale = x.new_ones((x.shape[0], self.n_topics))
            z = pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1))
            probs = self.decoder(z)
            pyro.sample('obs', dist.Multinomial(self.n_words, probs), obs=x)

    def _guide(self, x):
        pyro.module('encoder', self.encoder)
        with pyro.plate('data', x.shape[0]):
            z_loc, z_scale = self.encoder(x)
            pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1))

    def fit(self, X, n_steps=100, learning_rate=1e-2):
        _, self.X_ = build(X, self.n_words, self.vocab_size, return_counts=True)
        optim = Adam({'lr': learning_rate})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self._model, self._guide, optim, elbo)
        for _ in tqdm(range(n_steps)):
            loss = svi.step(self.X_)
            self.loss_log_.append(loss)
        return self
    
    def transform(self, _=None):
        dists_loc, dists_scale = self.encoder(self.X_)
        dists = dist.Normal(dists_loc, dists_scale).sample()
        return dists.detach()
    
# Product of Experts VAE Topic Model

class PVAETMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        encoding_layers = (input_dim, hidden_dim, hidden_dim)
        latent_layers = (hidden_dim, latent_dim)
        self.encoding_net = nn.Sequential(*list(mlp(encoding_layers, final_act='softplus', batch_norm=False, dropout=.2)))
        self.loc_net = nn.Sequential(*list(mlp(latent_layers, affine=True)))
        self.scale_net = nn.Sequential(*list(mlp(latent_layers, affine=True)))

    def forward(self, x):
        y = self.encoding_net(x)
        z_loc = self.loc_net(y)
        z_scale = (.5*self.scale_net(y)).exp()
        return z_loc, z_scale

class PVAETMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        layers = (latent_dim, hidden_dim, output_dim)
        self.dropout = nn.Dropout(.2)
        self.net = nn.Sequential(*list(mlp(layers, bias=False, final_act='softmax', affine=True)))
    
    def forward(self, z):
        y = self.dropout(z)
        x = self.net(y)
        return x
    
class PVAETM(nn.Module):
    def __init__(self, n_topics, n_words, vocab_size, hidden_dim):
        super().__init__()
        self.n_topics = n_topics
        self.n_words = n_words
        self.vocab_size = vocab_size
        self.encoder = VAETMEncoder(vocab_size, hidden_dim, n_topics)
        self.decoder = VAETMDecoder(n_topics, hidden_dim, vocab_size)
        self.X_ = None
        self.loss_log_ = []

    def _model(self, x):
        pyro.module('decoder', self.decoder)
        with pyro.plate('data', x.shape[0]):
            z_loc = x.new_zeros((x.shape[0], self.n_topics))
            z_scale = x.new_ones((x.shape[0], self.n_topics))
            z = pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1))
            probs = self.decoder(z)
            pyro.sample('obs', dist.Multinomial(self.n_words, probs), obs=x)

    def _guide(self, x):
        pyro.module('encoder', self.encoder)
        with pyro.plate('data', x.shape[0]):
            z_loc, z_scale = self.encoder(x)
            pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1))

    def fit(self, X, n_steps=100, learning_rate=1e-2):
        _, self.X_ = build(X, self.n_words, self.vocab_size, return_counts=True)
        optim = Adam({'lr': learning_rate})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self._model, self._guide, optim, elbo)
        for _ in tqdm(range(n_steps)):
            loss = svi.step(self.X_)
            self.loss_log_.append(loss)
        return self
    
    def transform(self, _=None):
        dists_loc, dists_scale = self.encoder(self.X_)
        dists = dist.Normal(dists_loc, dists_scale).sample()
        return dists.detach()
    
# Basic Neural Topic Model

class NTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        encoding_layers = (input_dim, hidden_dim, hidden_dim)
        latent_layers = (hidden_dim, latent_dim)
        self.encoding_net = nn.Sequential(*list(mlp(encoding_layers, final_act='softplus', batch_norm=False, dropout=.2)))
        self.loc_net = nn.Sequential(*list(mlp(latent_layers, affine=False)))
        self.scale_net = nn.Sequential(*list(mlp(latent_layers, affine=False)))

    def forward(self, x):
        y = self.encoding_net(x)
        z_loc = self.loc_net(y)
        z_scale = (.5*self.scale_net(y)).exp()
        return z_loc, z_scale

class NTMDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        layers = (latent_dim, output_dim)
        self.dropout = nn.Dropout(.2)
        self.net = nn.Sequential(*list(mlp(layers, bias=False, final_act='softmax', affine=False)))
    
    def forward(self, z):
        y = self.dropout(z)
        x = self.net(y)
        return x
    
class NTM(nn.Module):
    def __init__(self, n_topics, n_words, vocab_size, hidden_dim):
        super().__init__()
        self.n_topics = n_topics
        self.n_words = n_words
        self.vocab_size = vocab_size
        self.encoder = NTMEncoder(vocab_size, hidden_dim, n_topics)
        self.decoder = NTMDecoder(n_topics, vocab_size)
        self.X_ = None
        self.perplexity_log_ = []
        self.coherence_log_ = []

    def _model(self, x):
        pyro.module('decoder', self.decoder)
        with pyro.plate('data', x.shape[0]):
            z_loc = x.new_zeros((x.shape[0], self.n_topics))
            z_scale = x.new_ones((x.shape[0], self.n_topics))
            z = pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1))
            probs = self.decoder(z)
            pyro.sample('obs', dist.Multinomial(self.n_words, probs), obs=x)

    def _guide(self, x):
        pyro.module('encoder', self.encoder)
        with pyro.plate('data', x.shape[0]):
            z_loc, z_scale = self.encoder(x)
            pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1))

    def fit(self, X, n_steps=100, learning_rate=1e-2):
        _, self.X_ = build(X, self.n_words, self.vocab_size, return_counts=True)
        optim = Adam({'lr': learning_rate})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self._model, self._guide, optim, elbo)
        for _ in tqdm(range(n_steps)):
            loss = svi.step(self.X_)
            self.perplexity_log_.append(loss)
            labels = (self.X_@self.decoder.net[0].weight).argmax(-1)
            self.coherence_log_.append(coherence(self.X_, labels))
        return self
    
    def transform(self, _=None):
        dists = self.X_@self.decoder.net[0].weight
        return dists.detach()
    
# Product of Experts Neural Topic Model

class PNTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        encoding_layers = (input_dim, hidden_dim, hidden_dim)
        latent_layers = (hidden_dim, latent_dim)
        self.encoding_net = nn.Sequential(*list(mlp(encoding_layers, final_act='softplus', batch_norm=False, dropout=.2)))
        self.loc_net = nn.Sequential(*list(mlp(latent_layers, affine=False)))
        self.scale_net = nn.Sequential(*list(mlp(latent_layers, affine=False)))

    def forward(self, x):
        y = self.encoding_net(x)
        z_loc = self.loc_net(y)
        z_scale = (.5*self.scale_net(y)).exp()
        return z_loc, z_scale

class PNTMDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        layers = (latent_dim, output_dim)
        self.dropout = nn.Dropout(.2)
        self.net = nn.Sequential(*list(mlp(layers, bias=False, final_act='softmax', affine=False)))
    
    def forward(self, z):
        y = self.dropout(z)
        x = self.net(y)
        return x
    
class PNTM(nn.Module):
    def __init__(self, n_topics, n_words, vocab_size, hidden_dim):
        super().__init__()
        self.n_topics = n_topics
        self.n_words = n_words
        self.vocab_size = vocab_size
        self.encoder = PNTMEncoder(vocab_size, hidden_dim, n_topics)
        self.decoder = PNTMDecoder(n_topics, vocab_size)
        self.X_ = None
        self.perplexity_log_ = []
        self.coherence_log_ = []

    def _model(self, x):
        pyro.module('decoder', self.decoder)
        with pyro.plate('data', x.shape[0]):
            z_loc = x.new_zeros((x.shape[0], self.n_topics))
            z_scale = x.new_ones((x.shape[0], self.n_topics))
            z = F.softmax(pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1)), -1)
            probs = self.decoder(z)
            pyro.sample('obs', dist.Multinomial(self.n_words, probs), obs=x)

    def _guide(self, x):
        pyro.module('encoder', self.encoder)
        with pyro.plate('data', x.shape[0]):
            z_loc, z_scale = self.encoder(x)
            pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1))

    def fit(self, X, n_steps=100, learning_rate=1e-2):
        _, self.X_ = build(X, self.n_words, self.vocab_size, return_counts=True)
        optim = Adam({'lr': learning_rate})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self._model, self._guide, optim, elbo)
        for _ in tqdm(range(n_steps)):
            loss = svi.step(self.X_)
            self.perplexity_log_.append(loss)
            labels = (self.X_@self.decoder.net[0].weight).argmax(-1)
            self.coherence_log_.append(coherence(self.X_, labels))
        return self
    
    def transform(self, _=None):
        dists = self.X_@self.decoder.net[0].weight
        return dists.detach()
    
# Gaussian Softmax Neural Topic Model

class GNTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        encoding_layers = (input_dim, hidden_dim, hidden_dim)
        latent_layers = (hidden_dim, latent_dim)
        self.encoding_net = nn.Sequential(*list(mlp(encoding_layers, final_act='softplus', batch_norm=False, dropout=.2)))
        self.loc_net = nn.Sequential(*list(mlp(latent_layers, affine=False)))
        self.scale_net = nn.Sequential(*list(mlp(latent_layers, affine=False)))

    def forward(self, x):
        y = self.encoding_net(x)
        z_loc = self.loc_net(y)
        z_scale = (.5*self.scale_net(y)).exp()
        return z_loc, z_scale

class GNTMDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        layers = (latent_dim, output_dim)
        self.dropout = nn.Dropout(.2)
        self.net = nn.Sequential(*list(mlp(layers, bias=False, final_act='softmax', affine=False)))
    
    def forward(self, z):
        y = self.dropout(z)
        x = self.net(y)
        return x
    
class GNTM(nn.Module):
    def __init__(self, n_topics, n_words, vocab_size, hidden_dim):
        super().__init__()
        self.n_topics = n_topics
        self.n_words = n_words
        self.vocab_size = vocab_size
        self.encoder = GNTMEncoder(vocab_size, hidden_dim, n_topics)
        self.decoder = GNTMDecoder(n_topics, vocab_size)
        self.X_ = None
        self.perplexity_log_ = []
        self.coherence_log_ = []

    def _model(self, x):
        pyro.module('decoder', self.decoder)
        with pyro.plate('data', x.shape[0]):
            z_loc = x.new_zeros((x.shape[0], self.n_topics))
            z_scale = x.new_ones((x.shape[0], self.n_topics))
            z = pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1))
            probs = self.decoder(z)
            pyro.sample('obs', dist.Multinomial(self.n_words, probs), obs=x)

    def _guide(self, x):
        pyro.module('encoder', self.encoder)
        with pyro.plate('data', x.shape[0]):
            z_loc, z_scale = self.encoder(x)
            pyro.sample('latent', GaussianSoftmax(z_loc, z_scale).to_event(1))

    def fit(self, X, n_steps=100, learning_rate=1e-2):
        _, self.X_ = build(X, self.n_words, self.vocab_size, return_counts=True)
        optim = Adam({'lr': learning_rate})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self._model, self._guide, optim, elbo)
        for _ in tqdm(range(n_steps)):
            loss = svi.step(self.X_)
            self.perplexity_log_.append(loss)
            labels = (self.X_@self.decoder.net[0].weight).argmax(-1)
            self.coherence_log_.append(coherence(self.X_, labels))
        return self
    
    def transform(self, _=None):
        dists = self.X_@self.decoder.net[0].weight
        return dists.detach()
    
# Product of Experts Gaussian Softmax Neural Topic Model

class PGNTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        encoding_layers = (input_dim, hidden_dim, hidden_dim)
        latent_layers = (hidden_dim, latent_dim)
        self.encoding_net = nn.Sequential(*list(mlp(encoding_layers, final_act='softplus', batch_norm=False, dropout=.2)))
        self.loc_net = nn.Sequential(*list(mlp(latent_layers, affine=False)))
        self.scale_net = nn.Sequential(*list(mlp(latent_layers, affine=False)))

    def forward(self, x):
        y = self.encoding_net(x)
        z_loc = self.loc_net(y)
        z_scale = (.5*self.scale_net(y)).exp()
        return z_loc, z_scale

class PGNTMDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        layers = (latent_dim, output_dim)
        self.dropout = nn.Dropout(.2)
        self.net = nn.Sequential(*list(mlp(layers, bias=False, final_act='softmax', affine=False)))
    
    def forward(self, z):
        y = self.dropout(z)
        x = self.net(y)
        return x
    
class PGNTM(nn.Module):
    def __init__(self, n_topics, n_words, vocab_size, hidden_dim):
        super().__init__()
        self.n_topics = n_topics
        self.n_words = n_words
        self.vocab_size = None
        self.encoder = PGNTMEncoder(vocab_size, hidden_dim, n_topics)
        self.decoder = PGNTMDecoder(n_topics, vocab_size)
        self.X_ = None
        self.perplexity_log_ = []
        self.coherence_log_ = []

    def _model(self, x):
        pyro.module('decoder', self.decoder)
        with pyro.plate('data', x.shape[0]):
            z_loc = x.new_zeros((x.shape[0], self.n_topics))
            z_scale = x.new_ones((x.shape[0], self.n_topics))
            z = F.softmax(pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1)), dim=-1)
            probs = self.decoder(z)
            pyro.sample('obs', dist.Multinomial(self.n_words, probs), obs=x)

    def _guide(self, x):
        pyro.module('encoder', self.encoder)
        with pyro.plate('data', x.shape[0]):
            z_loc, z_scale = self.encoder(x)
            pyro.sample('latent', GaussianSoftmax(z_loc, z_scale).to_event(1))

    def fit(self, X, n_steps=100, learning_rate=1e-2):
        _, self.X_ = build(X, self.n_words, self.vocab_size, return_counts=True)
        optim = Adam({'lr': learning_rate})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self._model, self._guide, optim, elbo)
        for _ in tqdm(range(n_steps)):
            loss = svi.step(self.X_)
            self.perplexity_log_.append(loss)
            labels = (self.X_@self.decoder.net[0].weight).argmax(-1)
            self.coherence_log_.append(coherence(self.X_, labels))
        return self
    
    def transform(self, _=None):
        dists = self.X_@self.decoder.net[0].weight
        return dists.detach()
    
# Gaussian Prior Product of Experts Neural Topic Model

class GPNTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        encoding_layers = (input_dim, hidden_dim, hidden_dim)
        latent_layers = (hidden_dim, latent_dim)
        self.encoding_net = nn.Sequential(*list(mlp(encoding_layers, final_act='softplus', batch_norm=False, dropout=.2)))
        self.loc_net = nn.Sequential(*list(mlp(latent_layers, affine=False)))
        self.scale_net = nn.Sequential(*list(mlp(latent_layers, affine=False)))

    def forward(self, x):
        y = self.encoding_net(x)
        z_loc = self.loc_net(y)
        z_scale = (.5*self.scale_net(y)).exp()
        return z_loc, z_scale

class GPNTMDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        layers = (latent_dim, output_dim)
        self.dropout = nn.Dropout(.2)
        self.net = nn.Sequential(*list(mlp(layers, bias=False, final_act='softmax', affine=False)))
    
    def forward(self, z):
        y = self.dropout(z)
        x = self.net(y)
        return x
    
class GPNTM(nn.Module):
    def __init__(self, n_topics, n_words, vocab_size, hidden_dim, scale=.1):
        super().__init__()
        self.n_topics = n_topics
        self.n_words = n_words
        self.vocab_size = vocab_size
        self.scale = scale
        self.encoder = GPNTMEncoder(vocab_size, hidden_dim, n_topics)
        self.decoder = GPNTMDecoder(n_topics, vocab_size)
        self.X_ = None
        self.X_locs_ = None
        self.process_log_ = []
        self.perplexity_log_ = []
        self.coherence_log_ = []

    def _process(self, x, latent_dim=2, n_steps=1000, learning_rate=1e-2, n_inducing=100, variance=2., noise=1e-2, jitter=1e-5):
        x_prior_mean = x[:, :latent_dim].clone()
        kernel = gp.kernels.RBF(latent_dim, variance=torch.tensor(variance), lengthscale=torch.ones(latent_dim))
        y = Parameter(x[:, latent_dim:].clone())
        yu = stats.resample(x_prior_mean.clone(), n_inducing)
        process = gp.models.SparseGPRegression(y, y.t(), kernel, yu, noise=torch.tensor(noise), jitter=jitter)
        process.X = pyro.nn.PyroSample(dist.Normal(x_prior_mean, self.scale).to_event())
        process.autoguide('X', dist.Normal)
        optim = torch.optim.Adam(process.parameters(), learning_rate)
        self.process_log_ = gp.util.train(process, optim, num_steps=n_steps)
        process.mode = 'guide'
        self.X_locs_ = torch.cat([process.X_loc, x], -1)
        _, self.X_ = build(self.X_locs_.detach(), self.n_words, self.vocab_size, return_counts=True)
        return self.X_, self.X_locs_

    def _model(self, x):
        pyro.module('decoder', self.decoder)
        with pyro.plate('data', x.shape[0]):
            z_loc = x.new_zeros((x.shape[0], self.n_topics))
            z_scale = x.new_ones((x.shape[0], self.n_topics))
            z = F.softmax(pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1)), dim=-1)
            probs = self.decoder(z)
            pyro.sample('obs', dist.Multinomial(self.n_words, probs), obs=x)

    def _guide(self, x):
        pyro.module('encoder', self.encoder)
        with pyro.plate('data', x.shape[0]):
            z_loc, z_scale = self.encoder(x)
            pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1))

    def fit(self, X, n_steps=100, learning_rate=1e-2, gp_dim=2, gp_steps=1000, gp_rate=1e-2, gp_inducing=100, gp_variance=2.):
        Y = torch.tensor(X, dtype=torch.float32)
        self._process(Y, gp_dim, gp_steps, gp_rate, gp_inducing, gp_variance)
        optim = Adam({'lr': learning_rate})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self._model, self._guide, optim, elbo)
        for _ in tqdm(range(n_steps)):
            loss = svi.step(self.X_)
            self.perplexity_log_.append(loss)
            labels = (self.X_@self.decoder.net[0].weight).argmax(-1)
            self.coherence_log_.append(coherence(self.X_, labels))
        return self
    
    def transform(self, _=None):
        dists = self.X_@self.decoder.net[0].weight
        return dists.detach()
    
# Latent Dirichlet Allocation 
    
class PyroLDA():
    def __init__(self, n_topics, n_words, vocab_size, batch_size=100):
        self.n_topics = n_topics
        self.n_words = n_words
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        
        self.X_ = None
        self.topic_words_posterior_ = None
        self.doc_topics_posterior_ = None
        self.loss_log_ = []

    def _model(self, data=None):
        n_words, n_docs, vocab_size = *data.shape, data.unique().shape[-1]
        with pyro.plate('topics', self.n_topics):
            topic_words = pyro.sample('topic_words', dist.Dirichlet(torch.ones(vocab_size)/vocab_size))
        with pyro.plate('documents', n_docs, self.batch_size) as idx:
            if data is not None:
                data = data[:, idx]
            doc_topics = pyro.sample('doc_topics', dist.Dirichlet(torch.ones(self.n_topics)/self.n_topics))
            with pyro.plate('words', n_words):
                word_topics = pyro.sample('word_topics', dist.Categorical(doc_topics), infer={'enumerate': 'parallel'})
                data = pyro.sample('doc_words', dist.Categorical(topic_words[word_topics]), obs=data)
        
    def _guide(self, data):
        n_docs, vocab_size = data.shape[-1], data.unique().shape[-1]
        self.topic_words_posterior_ = pyro.param('topic_words_posterior', lambda: torch.ones(self.n_topics, vocab_size), constraint=constraints.greater_than(.5))
        self.doc_topics_posterior_ = pyro.param('doc_topics_posterior', lambda: torch.ones(n_docs, self.n_topics), constraint=constraints.greater_than(.5))
        with pyro.plate('topics', self.n_topics):
            pyro.sample('topic_words', dist.Dirichlet(self.topic_words_posterior_))
        with pyro.plate('documents', n_docs, self.batch_size) as idx:
            data = data[:, idx]
            pyro.sample('doc_topics', dist.Dirichlet(self.doc_topics_posterior_[idx]))

    def fit(self, X, n_steps=100, learning_rate=1e-1):
        self.X_ = build(X, self.n_words, self.vocab_size).T
        optim = Adam({'lr': learning_rate})
        elbo = TraceEnum_ELBO(max_plate_nesting=2)
        svi = SVI(self._model, self._guide, optim, elbo)
        for _ in tqdm(range(n_steps)):
            loss = svi.step(self.X_)
            self.loss_log_.append(loss)
        return self
    
    def transform(self, _=None):
        doc_topics = pyro.sample('doc_topics', dist.Dirichlet(self.doc_topics_posterior_))
        return doc_topics
    
# Spatial LDA
    
class GibbsSLDA(BaseEstimator, TransformerMixin):
    def __init__(self, n_topics, n_docs=150, vocab_size=25, vocab_steps=10, sigma=1., alpha=None, beta=None):
        self.n_topics = n_topics
        self.n_docs = n_docs
        self.vocab_size = vocab_size
        self.vocab_steps = vocab_steps
        self.sigma = sigma
        self.alpha = 1/n_docs if alpha is None else alpha
        self.beta = 1/n_topics if beta is None else beta

        self.library_ = None
        self.doc_imgs_ = None
        self.doc_locs_ = None
        self.doc_topic_counts_ = None
        self.topic_word_counts_ = None
        self.likelihood_log_ = []

    def _featurize(self, imgs, locs, markers):
        mask = (cdist(imgs, imgs) == 0).astype(np.int32)
        dists = cdist(locs, locs)
        weights = mask*np.exp(-(dists/self.sigma)**2)
        features = weights.T@markers
        return features

    def _shuffle(self, words):
        docs = np.random.choice(self.n_docs, (words.shape[0], 1))
        topics = np.random.choice(self.n_topics, (words.shape[0], 1))
        self.doc_topic_counts_ = np.zeros((self.n_docs, self.n_topics), dtype=np.int32)
        self.topic_word_counts_ = np.zeros((self.n_topics, self.vocab_size), dtype=np.int32)
        for d in range(self.n_docs):
            idx, counts = np.unique(topics[docs == d], return_counts=True)
            self.doc_topic_counts_[d, idx.astype(np.int32)] = counts
        for k in range(self.n_topics):
            idx, counts = np.unique(words[topics == k], return_counts=True)
            self.topic_word_counts_[k, idx.astype(np.int32)] = counts
        return docs, topics

    def _build(self, X):
        imgs, locs, markers = X[:, :1], X[:, 1:3], X[:, 3:]
        self.n_docs = np.unique(imgs).shape[0]*self.n_docs
        doc_idx = np.random.permutation(X.shape[0])[:self.n_docs]
        self.doc_imgs_, self.doc_locs_ = imgs[doc_idx], locs[doc_idx]
        features = self._featurize(imgs, locs, markers)
        codebook, _ = kmeans(features, self.vocab_size, self.vocab_steps)
        words = vq(features, codebook)[0][None].T
        docs, topics = self._shuffle(words)
        self.library_ = np.concatenate([imgs, locs, words, docs, topics], -1)
        return self.library_
    
    def _sample_doc(self, img, loc, topic, eta=1e-100, maximize=False):
        mask = (self.doc_imgs_ == img).astype(np.int32).T[0]
        doc_probs = mask*self.sigma**2/(((loc - self.doc_locs_)**2).sum(-1) + eta)
        topic_probs = self.doc_topic_counts_[:, topic] + self.alpha
        topic_probs /= (self.doc_topic_counts_ + self.alpha).sum(-1)
        probs = doc_probs*topic_probs/(doc_probs*topic_probs).sum()
        doc = np.argmax(probs) if maximize else np.random.choice(self.n_docs, p=probs)
        return doc, probs[doc]

    def _sample_topic(self, word, doc, maximize=False):
        topic_probs = self.doc_topic_counts_[doc] + self.alpha
        topic_probs /= (self.doc_topic_counts_[doc] + self.alpha).sum()
        word_probs = self.topic_word_counts_[:, word] + self.beta
        word_probs /= (self.topic_word_counts_ + self.beta).sum(-1)
        probs = topic_probs*word_probs/(topic_probs*word_probs).sum()
        topic = np.argmax(probs) if maximize else np.random.choice(self.n_topics, p=probs)
        return topic, probs[topic]
    
    def _sample(self, img, loc, word, old_doc, old_topic, maximize=False):
        new_doc, doc_likelihood = self._sample_doc(img, loc, old_topic, maximize=maximize)
        new_topic, topic_likelihood = self._sample_topic(word, old_doc, maximize=maximize)
        likelihood = doc_likelihood + topic_likelihood
        return new_doc, new_topic, likelihood
    
    def _decrement(self, word, doc, topic):
        self.doc_topic_counts_[doc, topic] -= 1
        self.topic_word_counts_[topic, word] -= 1
        return self.doc_topic_counts_, self.topic_word_counts_
    
    def _increment(self, word, doc, topic):
        self.doc_topic_counts_[doc, topic] += 1
        self.topic_word_counts_[topic, word] += 1
        return self.doc_topic_counts_, self.topic_word_counts_

    def _step(self):
        self.likelihood_log_.append(0.)
        for i in range(self.library_.shape[0]):
            img, loc, (word, doc, topic) = self.library_[i, :1], self.library_[i, 1:3], self.library_[i, 3:].astype(np.int32)
            self._decrement(word, doc, topic)
            doc, topic, likelihood = self._sample(img, loc, word, doc, topic)
            self._increment(word, doc, topic)
            self.library_[i, -2:] = doc, topic
            self.likelihood_log_[-1] += likelihood
        return self.likelihood_log_[-1]
    
    def fit(self, X, n_steps=100, verbose=1):
        self._build(X)
        for i in tqdm(range(n_steps)) if verbose == 1 else range(n_steps):
            likelihood = self._step()
            if verbose == 2:
                print('step', i, 'likelihood:', likelihood)
        return self
    
    def transform(self, _=None):
        topics = np.zeros(self.library_.shape[0], dtype=np.int32)
        for i in range(self.library_.shape[0]):
            word, doc = self.library_[i, 3:5].astype(np.int32)
            topics[i], _ = self._sample_topic(word, doc, True)
        return topics
