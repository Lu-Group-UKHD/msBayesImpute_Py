import pyro.distributions as dist
import torch

# cumulative density function of normal distribution (descending)
class DropoutNormalCDF(dist.Normal):
    """cumulative density of normal function distribution"""
    
    def __init__(self, loc, scale, validate_args = None):
        super().__init__(loc, scale, validate_args = validate_args)
    def prob(self, value):
        return 1 - self.cdf(value)
    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return self.prob(value).log()
    def expand(self, batch_shape, _instance = None):
        new = self._get_checked_instance(DropoutNormalCDF, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(DropoutNormalCDF, new).__init__(new.loc, new.scale, validate_args = None)
        new._validate_args = self._validate_args
        return new