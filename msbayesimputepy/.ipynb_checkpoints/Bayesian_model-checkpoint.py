from pyro.nn import PyroModule
import pyro.distributions as dist
import pyro

import sys
import torch

from .sigmoid_distribution import DropoutNormalCDF

class Model(PyroModule):
    def __init__(self, D: int = None, N: int = None, K: int = None):
        """a Bayesian network for imputation

        Parameters
        ----------
        D: int
            feature dimension
        N: int
            observation dimension
        K: int
            factor number
        """
        
        super().__init__()
        self.D = D # The original dimension
        self.N = N # Number of samples
        self.K = K # The latent dimension
        self.tune_miss_id = torch.tensor([], dtype = torch.int)
        self.predON = False
        
    def detect_missing(self, X = None):
        """Method to detect indexes of missing values and obsered values"""
        
        self.miss_id = X.reshape(-1).isnan().nonzero(as_tuple = True)[0]
        self.obs_id = (X.reshape(-1).isnan() == 0).nonzero(as_tuple = True)[0]

    def define_tau_prior(self):
        """Method to define precision prior for Normal likelihood"""
        
        tau_global =  pyro.sample("tau_global", dist.Gamma(1e-14, 1e-14)).reshape(-1) if self.tau is None else self.tau
        tau_loc = pyro.sample("tau_loc", dist.TransformedDistribution(dist.Normal(tau_global, 1).expand([self.D]).to_event(1), 
                                                                                dist.transforms.SoftplusTransform())) 
        tau_rate = pyro.sample("tau_rate", dist.Gamma(1, 1).expand([self.D]).to_event(1))
        tau = pyro.sample("tau", dist.Gamma(tau_loc * tau_rate, tau_rate).to_event(1)) 
        t_tau = 1/tau ** 0.5
        
        return t_tau
        
    def define_dropout_prior(self, option = None):
        """Method to define dropout prior for dropout model"""
        
        rho_rate = pyro.sample("rho_rate", dist.Gamma(100, 10))
        rho = pyro.sample("rho", dist.Normal(self.rho, 1/rho_rate).to_event(1)).type(torch.float64)   
        
        zeta_loc = pyro.sample("zeta_loc", dist.TransformedDistribution(dist.Normal(self.zeta, 1).to_event(1), 
                                                                        dist.transforms.SoftplusTransform())) 
        zeta_rate = pyro.sample("zeta_rate", dist.Gamma(200, 10).expand([self.D]).to_event(1))
        zeta = pyro.sample("zeta", dist.Gamma(zeta_loc * zeta_rate, zeta_rate).to_event(1)).type(torch.float64) 
            
        return [rho, zeta]

    def update_dropout_Bayes(self, X, tau, option = None):
        """Method to update dropout model using Bayesian model"""

        t_tau = 0 if self.tau is None else 1 / tau ** 0.5
        Y = torch.zeros(self.D, self.N)
        Y.reshape(-1)[self.obs_id] = 1

        rho_rate = pyro.sample("rho_rate2", dist.Gamma(100, 10).expand([self.D]).to_event(1))
        rho_sample = pyro.sample("rho2", dist.Normal(self.rho, 1/rho_rate).to_event(1)).type(torch.float64) 

        zeta_loc = pyro.sample("zeta_loc2", dist.TransformedDistribution(dist.Normal(self.zeta, 1).to_event(1), 
                                                                         dist.transforms.SoftplusTransform())) 
        zeta_rate = pyro.sample("zeta_rate2", dist.Gamma(200, 10).expand([self.D]).to_event(1))
        zeta_sample = pyro.sample("zeta2", dist.Gamma(zeta_loc * zeta_rate, zeta_rate).to_event(1)).type(torch.float64)  
        
        rho_sample = rho_sample.reshape(self.D, 1).repeat(1, self.N)
        zeta_sample = zeta_sample.reshape(self.D, 1).repeat(1, self.N)

        prob = 1 - DropoutNormalCDF(rho_sample, zeta_sample).prob(X)
        
        with pyro.plate("sample_all2", self.N * self.D):
            pyro.sample("data2", dist.Bernoulli(prob.reshape(-1)), obs = Y.reshape(-1))
         
    def forward(self, X = None, rho = None, zeta = None, tau = None, option = False):
        """forward step"""

        # hierarchical priors, w & z & intercept #
        self.detect_missing(X)
        self.rho, self.zeta, self.tau = rho, zeta, tau
        t_tau = self.define_tau_prior()

        with pyro.plate("factor", self.K):
            self.epsilon = pyro.sample("epsilon", dist.Beta(0.5, 0.5))
            with pyro.plate("feature", self.D):
                self.lambda_local = pyro.sample("lambda_local", dist.HalfCauchy(1))
                self.w_priors = pyro.sample("w", dist.Normal(0, self.epsilon * self.lambda_local))
            with pyro.plate("sample", self.N):
                self.z_priors = pyro.sample("z", dist.Normal(0, 1))

        self.w_priors = self.w_priors.view(self.D, self.K, 1)
        self.z_priors = self.z_priors.T.view(1, self.K, self.N)
        self.inter_dif = pyro.sample("inter_n", dist.Normal(0, X.nanmean(axis = 0).std()).expand([self.N]).to_event(1))
        self.inter = pyro.sample("inter", dist.Normal(X.nanmean(axis = 1), 1).to_event(1))
        loc = torch.einsum("...ikj,...ikj->...ij", self.w_priors, self.z_priors).view(self.D, self.N) + \
              self.inter.reshape(self.D, 1) + self.inter_dif.reshape(1, self.N)
                
        # observed values #
        with pyro.plate("sample_obs", len(self.obs_id)):
            pyro.sample("obs", dist.Normal(loc.reshape(-1)[self.obs_id], t_tau.reshape(self.D, 1).repeat(1, self.N).reshape(-1)[self.obs_id]), 
                        obs = X.reshape(-1)[self.obs_id])

        Y = X.clone().detach()
        Y[Y == Y] = 1
        Y = torch.nan_to_num(Y, nan = 0)
        
        # missing values #
        ## fixed
        if option == "fixed": 
            zeta = zeta.reshape(self.D, 1).repeat(1, self.N) if not isinstance(zeta, float) else torch.ones(self.D, self.N) * zeta
            rho = rho.reshape(self.D, 1).repeat(1, self.N) if not isinstance(rho, float) else torch.ones(self.D, self.N) * rho
                
            if not self.predON:
                tune_miss_mask = (abs(1 - DropoutNormalCDF(rho, zeta).prob(loc) - Y) <= 0.95).reshape(-1)
                tune_miss_mask[self.obs_id] = False
                self.tune_miss_id = tune_miss_mask.nonzero(as_tuple = True)[0]
            else:
                self.tune_miss_id = self.miss_id
                    
            with pyro.plate("sample_miss", len(self.tune_miss_id)):
                pyro.sample("miss", DropoutNormalCDF(rho.reshape(-1)[self.tune_miss_id], zeta.reshape(-1)[self.tune_miss_id]), 
                            obs = loc.reshape(-1)[self.tune_miss_id])

        ## feature-wise
        elif option == "featureWise":
            rho, zeta = self.define_dropout_prior(option)
            zeta = zeta.reshape(self.D, 1).repeat(1, self.N).type(torch.float64) 
            rho = rho.reshape(self.D, 1).repeat(1, self.N).type(torch.float64)  
            
            prob = 1 - DropoutNormalCDF(rho, zeta).prob(loc)
            with pyro.plate("sample_all", self.N * self.D):
                pyro.sample("data", dist.Bernoulli(prob.reshape(-1)), obs = Y.reshape(-1))
        
        ## the option exists but doesn't fit any values above        
        elif option is not None and option != "None" and option != "none":
            sys.exit("Incorrect option, please check the argument.")
