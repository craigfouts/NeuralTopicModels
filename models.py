import numpy as np
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.ops.stats as stats
import torch
import torch.nn.functional as F
import torch.nn as nn
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.optim import Adam
from scipy.cluster.vq import kmeans, vq
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
    codebook, _ = kmeans(x[:, 2:], vocab_size, vocab_steps)
    neighbors = NearestNeighbors(n_neighbors=n_words).fit(x)
    _, neighbor_idx = neighbors.kneighbors(x)
    data = np.zeros((x.shape[0], n_words), dtype=np.int32)
    for i in range(x.shape[0]):
        data[i], _ = vq(x[neighbor_idx, 2:][i], codebook)
    data_tensor = torch.tensor(data, dtype=torch.float32)
    if return_counts:
        return data_tensor, count(data_tensor, vocab_size)
    return data_tensor

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
        self.encoder = VAETMEncoder(vocab_size, hidden_dim, n_topics)
        self.decoder = VAETMDecoder(n_topics, hidden_dim, vocab_size)
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
        optim = Adam({'lr': learning_rate})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self._model, self._guide, optim, elbo)
        for _ in tqdm(range(n_steps)):
            loss = svi.step(X)
            self.loss_log_.append(loss)
        return self
    
    def transform(self, X):
        dists_loc, dists_scale = self.encoder(X)
        dists = dist.Normal(dists_loc, dists_scale).sample()
        return dists
    
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
        self.encoder = VAETMEncoder(vocab_size, hidden_dim, n_topics)
        self.decoder = VAETMDecoder(n_topics, hidden_dim, vocab_size)
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
        optim = Adam({'lr': learning_rate})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self._model, self._guide, optim, elbo)
        for _ in tqdm(range(n_steps)):
            loss = svi.step(X)
            self.loss_log_.append(loss)
        return self
    
    def transform(self, X):
        dists_loc, dists_scale = self.encoder(X)
        dists = dist.Normal(dists_loc, dists_scale).sample()
        return dists
    
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
        self.encoder = NTMEncoder(vocab_size, hidden_dim, n_topics)
        self.decoder = NTMDecoder(n_topics, vocab_size)
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
        optim = Adam({'lr': learning_rate})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self._model, self._guide, optim, elbo)
        for _ in tqdm(range(n_steps)):
            loss = svi.step(X)
            self.loss_log_.append(loss)
        return self
    
    def transform(self, X):
        dists = X@self.decoder.net[0].weight
        return dists
    
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
        self.encoder = PNTMEncoder(vocab_size, hidden_dim, n_topics)
        self.decoder = PNTMDecoder(n_topics, vocab_size)
        self.loss_log_ = []

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
        optim = Adam({'lr': learning_rate})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self._model, self._guide, optim, elbo)
        for _ in tqdm(range(n_steps)):
            loss = svi.step(X)
            self.loss_log_.append(loss)
        return self
    
    def transform(self, X):
        dists = X@self.decoder.net[0].weight
        return dists
    
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
        self.encoder = GNTMEncoder(vocab_size, hidden_dim, n_topics)
        self.decoder = GNTMDecoder(n_topics, vocab_size)
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
            pyro.sample('latent', GaussianSoftmax(z_loc, z_scale).to_event(1))

    def fit(self, X, n_steps=100, learning_rate=1e-2):
        optim = Adam({'lr': learning_rate})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self._model, self._guide, optim, elbo)
        for _ in tqdm(range(n_steps)):
            loss = svi.step(X)
            self.loss_log_.append(loss)
        return self
    
    def transform(self, X):
        dists = X@self.decoder.net[0].weight
        return dists
    
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
        self.encoder = PGNTMEncoder(vocab_size, hidden_dim, n_topics)
        self.decoder = PGNTMDecoder(n_topics, vocab_size)
        self.loss_log_ = []

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
        optim = Adam({'lr': learning_rate})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self._model, self._guide, optim, elbo)
        for _ in tqdm(range(n_steps)):
            loss = svi.step(X)
            self.loss_log_.append(loss)
        return self
    
    def transform(self, X):
        dists = X@self.decoder.net[0].weight
        return dists
    
# Gaussian Prior Product of Experts Neural Topic Model

class PPGSTMEncoder(nn.Module):
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

class PPGSTMDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        layers = (latent_dim, output_dim)
        self.dropout = nn.Dropout(.2)
        self.net = nn.Sequential(*list(mlp(layers, bias=False, final_act='softmax', affine=False)))
    
    def forward(self, z):
        y = self.dropout(z)
        x = self.net(y)
        return x
    
class PPGSTM(nn.Module):
    def __init__(self, n_topics, n_words, vocab_size, hidden_dim, scale=.1):
        super().__init__()
        self.n_topics = n_topics
        self.n_words = n_words
        self.vocab_size = vocab_size
        self.scale = scale
        self.encoder = PPGSTMEncoder(vocab_size, hidden_dim, n_topics)
        self.decoder = PPGSTMDecoder(n_topics, vocab_size)
        self.X_ = None
        self.library_ = None
        self.process_log_ = []
        self.loss_log_ = []

    def _process(self, x, gp_steps=1000, noise=torch.tensor(1e-2), jitter=1e-5):
        x_prior_mean = x.clone()
        kernel = gp.kernels.RBF(2, lengthscale=torch.ones(2))
        y = Parameter(x_prior_mean.clone())
        yu = stats.resample(x_prior_mean.clone(), 100)
        process = gp.models.SparseGPRegression(y, y.t(), kernel, yu, noise=noise, jitter=jitter)
        process.X = pyro.nn.PyroSample(dist.Normal(x_prior_mean, self.scale).to_event())
        process.autoguide('X', dist.Normal)
        self.process_log_ = gp.util.train(process, num_steps=gp_steps)
        process.mode = 'guide'
        self.X_ = process.X_loc
        _, self.library_ = build(self.X_.detach(), self.n_words, self.vocab_size, return_counts=True)
        return self.X_, self.library_

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

    def fit(self, X, n_steps=100, learning_rate=1e-2, gp_steps=1000):
        self._process(X, gp_steps)
        optim = Adam({'lr': learning_rate})
        elbo = TraceMeanField_ELBO()
        svi = SVI(self._model, self._guide, optim, elbo)
        for _ in tqdm(range(n_steps)):
            loss = svi.step(self.library_)
            self.loss_log_.append(loss)
        return self
    
    def transform(self, _=None):
        dists = self.library_@self.decoder.net[0].weight
        return dists
