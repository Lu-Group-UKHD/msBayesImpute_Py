from scipy.stats import trim_mean
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from pyro.params.param_store import ParamStoreDict
from pyro.infer.autoguide.initialization import init_to_value
import pyro

import torch
import numpy as np
import pandas as pd
from time import time

from .sigmoid_distribution import DropoutNormalCDF
from .Bayesian_model import Model

class msBayesImpute():
    def __init__(self, n_components = None, convergence_mode = "fast", drop_factor_threshold = 0.01):
        """a Bayesian network for imputation

        Parameters
        ----------
        n_components: int
            user can fix the number of components on their own; By default, the model can automatically filter out the insignificant factors 
        convergence_mode: dict
            two convergence options: fast by default, and slow
        drop_factor_threshold: float
            a threshold to filter out significant latent factors;By default, the values is 0.01
        """

        self.option = "alternating_featureWise"
        self.convergence_mode = convergence_mode
        self.convergence_threshold = {"fast": 1e-2, "slow": 5e-3}
        self.n_components = n_components
        self.drop_factor_threshold = drop_factor_threshold
    
    def impute_mindet(self, X):
        """Method to impute by MinDet method"""
        
        X_impute = X.clone().detach()
        lowQuantile_samples = torch.nanquantile(X, axis = 0, q = 0.01)
        
        for col in range(X.shape[1]):
            X_impute[:, col] = torch.nan_to_num(X_impute[:, col], nan = lowQuantile_samples[col]) 
            
        return X_impute
        
    def initialize_dropout(self, X, X_impute):
        """Method to calculate the dropout parameters using original missing data and MinDet-based imputed data"""
        
        if X.isnan().reshape(-1).sum() > 0:
            meanVal = X_impute.nanmean(axis = 1)
            missProb = X.isnan().sum(axis = 1) / X.shape[1]
            
            rho_ini = trim_mean(meanVal[missProb > 0], 0.2)
            sigma20 = trim_mean((rho_ini - meanVal[missProb > 0]) ** 2, 0.2)
            zeta_ini = 1 / (sigma20 ** 0.5)
            
            fun = lambda x: sum((missProb - DropoutNormalCDF(x[0], x[1]).prob(meanVal)) ** 2)
            opt_res = minimize(fun, np.array([rho_ini, zeta_ini]), method = 'nelder-mead', bounds = ((1e-3, None), (1e-3, None)))
            
            return opt_res.x
            
        else:
            return [None, None]
        
    def initialize_pca(self, X_impute, n_components = 2):
        """Method to principle component analysis"""
        
        n_components = np.min(X_impute.shape)
        pca = PCA(n_components = n_components)
        inter = X_impute.mean(axis = 1)[:, None]
        X_impute_centered_t =  X_impute.T - inter.T
        pca.fit(X_impute_centered_t)
        weight, score = pca.components_, pca.transform(X_impute_centered_t)
        factors, var = self.assess_factors(X_impute, 
                                           torch.tensor(weight.T, dtype = torch.float64), 
                                           torch.tensor(score, dtype = torch.float64),
                                           threshold = 0.05)
        rec = np.matmul(score[:, factors], weight[factors, :])
        noise = (X_impute_centered_t - rec).T
        return [noise, weight, score]

    def detect_factors(self, X, rho, zeta, tau, threshold, start_t, verbose, drop_factor_threshold):
        """Method to detect true numbers when decreasing factor numbers using simple submodel"""
         
        D, N = X.shape[0], X.shape[1]
        old_Nr = np.min([D, N]) + 1 if np.min([D, N]) <= 20 else int(np.min([D, N])/2) + 1
        new_Nr = old_Nr - 1
        option = self.option
        init_dict = None
        losses = []
        while old_Nr > new_Nr:
            if verbose: 
                print("\033[94m  - Current factor number is: %d\033[0m"%new_Nr)
            
            pyro.clear_param_store()
            self.define_model(D = D, N = N, K = new_Nr, init_dict = init_dict)
            self.option = option.split("_", 1)[1] if option is not None and "_" in option else option
            losses = self.core(X, rho, zeta, tau, threshold, self.option, start_t = start_t, losses = losses, verbose = verbose)

            if new_Nr == 1:
                break

            samples = self.get_params(X, 100)
            factors, var = self.assess_factors(X, samples["w"], samples["z"], drop_factor_threshold)
            factors = factors if len(factors) > 0 else sorted(range(len(var)), key = var.__getitem__)[::-1][:1]
            old_Nr = new_Nr
            new_Nr = len(factors)
            init_dict = self.guide()
            init_dict["w"], init_dict["z"] = init_dict["w"][:, factors], init_dict["z"][:, factors]
            init_dict["epsilon"] = init_dict["epsilon"][factors]
            init_dict["lambda_local"] = init_dict["lambda_local"][:, factors]

        if verbose:
            print("\033[94m  The final number is %d.\033[0m"%new_Nr)
        self.option = option
        return losses

    def assess_factors(self, X, w, z, threshold):
        """Method to assess and determing factor numbers"""
        
        D, N, K = w.shape[0], z.shape[0], w.shape[1]
        w = w.reshape(D, K, 1)
        z = z.T.reshape(1, K, N)
        inter = X.nanmean(axis = 1)
        
        X_center = X - inter.reshape(D, 1)
        tot_var = torch.nansum(X_center ** 2)
        factors = []
        var = []
        for n in range(K):
            res_var = torch.nansum((X_center - torch.matmul(w[:, n, :], z[:, n, :])) ** 2)
            var += [1 - res_var/tot_var]
            
            if 1 - res_var/tot_var >= threshold: 
                factors += [n] # modification
        
        return [factors, var]
    
    def define_model(self, D = None, N = None, K = None, init_dict = None):
        if D and N and K:
            self.model = Model(D = D, N = N, K = K)
            
        if init_dict is not None:
            self.guide = pyro.infer.autoguide.AutoDiagonalNormal(self.model, init_to_value(values = init_dict))
        else: 
            self.guide = pyro.infer.autoguide.AutoDiagonalNormal(self.model)
            
        optimizer = pyro.optim.Adam({"lr": 5e-2})
        elbo = pyro.infer.Trace_ELBO()
        self.svi = pyro.infer.SVI(model = self.model, guide = self.guide, optim = optimizer, loss = elbo)
        
    def train(self, input, verbose = True, log = False):
        """Run steps, including initialisation of dropout, pca, factors numbers(simple model) 
        and if the option is advanced submodel, implement alternating models"""
        
        if isinstance(input, pd.DataFrame):
            X = torch.tensor(input.to_numpy(), dtype = torch.float64)
        else:
            X = torch.tensor(input, dtype = torch.float64)

        if ((X.isnan() == 0).sum(axis = 1) == 0).any():
            print("The input data contains completely missing features. The msBayesImpute requires as least one observation per feature!")
            return
        if not X.isnan().reshape(-1).any():
            print("The input data is complete without missingness.")
            return
            
        pyro.clear_param_store()
        threshold = self.convergence_threshold[self.convergence_mode]
        D, N = X.shape[0], X.shape[1]
        option = self.option
        if verbose:
            print("\033[1mModel option: %s, convergence mode: %s, shinrakge: HorseShoe\033[0m"%(option, self.convergence_mode))
        
        # initialization precedure #
        X_impute = self.impute_mindet(X)
        
        ## 1. dropout
        rho, zeta = self.initialize_dropout(X, X_impute)
        if verbose:
            print("\n1. Initialize dropout curve: ", "rho - %.2f, zeta - %.2f"%(rho, zeta))
                
        try:
            ## 2. factor numbers
            start_t = time()
            self.start_e = 1
            if self.n_components is None:
                if verbose:
                    print("2. Initialize training and optimize number of factors.")
                losses = self.detect_factors(X, rho, zeta, None, threshold, start_t, verbose, self.drop_factor_threshold)
            else:
                if verbose:
                    print("2. Initialize training with %s factors."%self.n_components)
                self.define_model(D = D, N = N, K = self.n_components, init_dict = None)
                self.option = option.split("_", 1)[1] if option is not None and "_" in option else option
                losses = self.core(X, rho, zeta, None, threshold, self.option, start_t = start_t, verbose = verbose)
    
            ## 3. core of training
            if option is not None and "_" in option:
                self.start_e = 1
                if verbose:
                    print("3. Start final model training.")
                losses = self.core(X, rho, zeta, None, threshold, option, 30_000, start_t, losses, verbose)
    
        except ValueError as e:
            pyro.clear_param_store()
            if verbose:
                print("\n\033[1mRelaunch model using pca initialisation\033[0m")
            self.option = option
            if verbose:
                print("\n1. Initialize dropout curve: ", "rho - %.2f, zeta - %.2f"%(rho, zeta))
        
            ## 2. pca
            noise, _, _ = self.initialize_pca(X_impute)
            if verbose:
                print("2. Initialize factorization model.")
            tau = (1 / noise.var(axis = 1)).mean()
            
            ## 3. factor numbers
            
            self.start_e = 1
            if self.n_components is None:
                if verbose:
                    print("3. Initialize training and optimize number of factors.")
                losses = self.detect_factors(X, rho, zeta, tau, threshold, start_t, verbose, self.drop_factor_threshold)
            else:
                if verbose:
                    print("3. Initialize training with %s factors."%self.n_components)
                self.define_model(D = D, N = N, K = self.n_components, init_dict = None)
                self.option = option.split("_", 1)[1] if option is not None and "_" in option else option
                losses = self.core(X, rho, zeta, tau, threshold, self.option, start_t = start_t, verbose = verbose)
    
            ## 4. core of training
            if option is not None and "_" in option:
                self.start_e = 1
                if verbose:
                    print("4. Start final model training.")
                losses = self.core(X, rho, zeta, tau, threshold, option, 30_000, start_t, losses, verbose)
        
        if verbose:        
            sec = time() - start_t
            min = sec // 60
            sec %= 60
            print("\n  Finally, elapsed time=%02d:%02d"%(min, sec))
            print("\033[1m\nModel training finished!\033[0m\n")
        if log:
            return losses

    def core(self, X, rho, zeta, tau, threshold, option = None, epochs = 30_000, start_t = 0, losses = [], verbose = True):  
        """core code of running simple model and advanced model
        
        Parameters
        ----------
        X: torch.Tensor
            original data with missingness
        rho: float
            initial loc parameter for dropout model
        zeta: float
            initial slope parameter for dropout model
        tau: float
            precision for likelihood
        threshold: float
            a value to judge if the model is converged
        option: str
            described in the __init__ method
        epochs: int
            running iterations
        start_t: int
            the start time point of launching model
        losses: list
            a list of ELBO log
        verbose: logical
            If set to True, the model's progress will be displayed; if False, all messages will be suppressed.

        Returns
        -------
        list
            a list of errors in iterations
        """
        
        D, N = X.shape[0], X.shape[1]
        X_update = None
        print_losses = True
        convergence_token, convergence_token_flexible, convergence_token_dropout, convergence_token_fixed = 1, 1, 1, 3
        self.option = option.split("_", 1)[1] if option is not None and "_" in option else option
        epoch_dropout = epoch_fixed = 20_000
        if option == "featureWise":
            rho, zeta = torch.ones(D) * torch.tensor(rho), torch.ones(D) * torch.tensor(zeta)
            
        for epoch in range(self.start_e, epochs + 1):
            ### refine model
            if option == "alternating_featureWise":
                # step 1: fixed
                if convergence_token_flexible < 3 and epoch <= 10_000:
                    if self.option == "featureWise": 
                        samples = self.get_params(X)
                        rho_update, zeta_update = samples["rho"].reshape(-1), samples["zeta"].reshape(-1)
                        self.option = "fixed"
                        if verbose:
                            print("\033[91m  Step 1, refine factorization model.\033[0m")
                        init_dict = self.guide()
                        for key in ["rho_rate", "zeta_rate", "zeta_loc", "rho", "zeta"]:
                            init_dict.pop(key)
                        pyro.clear_param_store()
                        self.define_model(init_dict = init_dict)
                    loss = self.svi.step(X, rho_update, zeta_update, tau, self.option)
                    losses.append(loss / (D * N))
                    convergence_token_flexible = self.update_convergence(losses, convergence_token_flexible, threshold) if len(losses) > 1 else 1
                
                # step 2: dropout
                elif convergence_token_dropout < 3 and epoch < epoch_dropout:
                    if X_update is None:
                        epoch_dropout = epoch + 10_000
                        print_losses = False
                        if verbose:
                            print("\033[91m  Step 2, refine feature-wise dropout curves.\033[0m")
                        tau_update = self.get_params(X)["tau"].reshape(-1)
                        X_update = torch.tensor(self.predict(X).to_numpy(), dtype = torch.float64)
                        optimizer = pyro.optim.Adam({"lr": 5e-2})
                        elbo = pyro.infer.Trace_ELBO()
                        guide_dropout = pyro.infer.autoguide.AutoDiagonalNormal(self.model.update_dropout_Bayes)
                        svi_dropout = pyro.infer.SVI(model = self.model.update_dropout_Bayes, 
                                                     guide = guide_dropout,
                                                     optim = optimizer,
                                                     loss = elbo)
                        GLOBAL_STORE = pyro.poutine.runtime._PYRO_PARAM_STORE
                        self.setup_params(ParamStoreDict())
                        losses2 = []
                    loss2 = svi_dropout.step(X_update, tau_update, option)
                    losses2.append(loss2 / (D * N))
                    convergence_token_dropout = self.update_convergence(losses2, convergence_token_dropout, threshold) if len(losses2) > 1 else 1
                    if len(losses2) > 1 and verbose:
                        self.print_progress(epoch, losses2, start_t, "LOSS2")
                
                # step 3: fixed
                else: 
                    if X_update is not None:
                        epoch_fixed = epoch + 10_000
                        print_losses = True
                        predictive = pyro.infer.Predictive(model = self.model.update_dropout_Bayes, guide = guide_dropout, num_samples = 100)
                        samples = predictive(X_update, tau_update, option)
                        rho_update, zeta_update = samples["rho2"].mean(axis = 0).reshape(-1), samples["zeta2"].mean(axis = 0).reshape(-1)
                        X_update = None
                        self.setup_params(GLOBAL_STORE)
                        if verbose:
                            print("\033[91m  Step 3, refine factorization model.\033[0m")
                        self.model.tune_miss_id = torch.tensor([], dtype = torch.int)
                    if convergence_token == 3 or epoch == epoch_fixed:
                        break
                    loss = self.svi.step(X, rho_update, zeta_update, tau, self.option)
                    losses.append(loss / (D * N))
                    convergence_token = self.update_convergence(losses, convergence_token, threshold)
    
            ### initialise model
            else: 
                loss = self.svi.step(X, rho, zeta, tau, self.option)
                losses.append(loss / (D * N))
                convergence_token = self.update_convergence(losses, convergence_token, threshold) if len(losses) > 1 else 1
                if convergence_token == 3:
                    break
                    
            if print_losses and verbose:
                self.print_progress(epoch, losses, start_t)
            if epoch == epochs and "_" in option:
                print("After %d epochs, the msBayesImpute hasn't reached convergence."%(epoch))
        self.start_e = epoch
        return losses
        
    def print_progress(self, epoch, losses, start_t, name = "LOSS"):
        sec = time() - start_t
        min = sec // 60
        sec %= 60
        
        if epoch == 1:
            print("  Epoch 1: elapsed time=%02d:%02d, %s=%.6f"%(min, sec, name, losses[-1]))
        else:
            if (epoch <= 1000 and epoch % 100 == 0) or (epoch > 1000 and epoch % 1000 == 0):
                print("  Epoch %d: elapsed time=%02d:%02d, deltaLOSS=%.6f (%.5f%%)"%
                      (epoch, min, sec, losses[-2] - losses[-1], 100 * abs((losses[-1] - losses[-2]) / losses[0])))

    def update_convergence(self, losses, convergence_token, threshold):
        """Method to update convergence token"""
        
        if 100 * abs((losses[-1] - losses[-2]) / losses[0]) < threshold:
            convergence_token += 1
        else:
            convergence_token = 1
            
        return convergence_token

    def setup_params(self, paramstore):
        """Method to set up pyro parameters in storage"""
        
        # Replace all imports of the global param store with our version.
        pyro.poutine.runtime._PYRO_PARAM_STORE = \
            pyro.primitives._PYRO_PARAM_STORE = \
            pyro.nn.module._PYRO_PARAM_STORE = \
            paramstore
        # Replace static use of the imported param store.
        pyro.primitives._param = \
            pyro.poutine.runtime.effectful(paramstore.get_param, type="param")
    
    def get_params(self, X, num_samples = 100, output = None):
        """Method to extract all pyro parameters in storage"""
        
        self.model.predON = True
        if isinstance(X, pd.DataFrame): 
            X = torch.tensor(X.to_numpy(), dtype = torch.float64)        
        predictive = pyro.infer.Predictive(model = self.model, guide = self.guide, num_samples = num_samples)
        samples = predictive(X, self.model.rho, self.model.zeta, self.model.tau, self.option)
        for key in samples.keys():
            samples[key] = torch.squeeze(samples[key].mean(axis = 0)) if key != "w" and key != "z" else samples[key].mean(axis = 0)
            if output == "numpy" or output == "matrix":
                samples[key] = samples[key].numpy()
        
        self.model.predON = False
        return samples

    def predict(self, X, replace = True):
        """Method to predict data after the model training is finished"""
        
        index = columns = None
        if isinstance(X, pd.DataFrame): 
            index, columns = X.index, X.columns
            X = torch.tensor(X.to_numpy(), dtype = torch.float64)
            
        D, N, K, obs_id = self.model.D, self.model.N, self.model.K, self.model.obs_id
        samples = self.get_params(X)
        X_impute = torch.einsum("...ikj,...ikj->...ij", samples["w"].reshape(D, K, 1), samples["z"].T.reshape(1, K, N)) + \
        samples["inter"].reshape(D, 1) + samples["inter_n"].reshape(1, N)
        
        if replace:
            X_impute.reshape(-1)[obs_id] = X.reshape(-1)[obs_id]
        index, columns = [range(X_impute.shape[0]), range(X_impute.shape[1])] if index is None or columns is None else [index, columns]
        df_impute = pd.DataFrame(data = X_impute, index = index, columns = columns)
            
        return df_impute

    def calc_rmsd(self, X_complete, X_impute, id = None):
        """Method to calculate root mean squared deviation between reference and predicted data"""
        
        X_complete = torch.tensor(X_complete.to_numpy(), dtype = torch.float64) if isinstance(X_complete, pd.DataFrame) else X_complete
        X_impute = torch.tensor(X_impute.to_numpy(), dtype = torch.float64) if isinstance(X_impute, pd.DataFrame) else X_impute
        if id is not None:
            x1 = X_complete.reshape(-1)[id].detach().numpy()
            x2 = X_impute.reshape(-1)[id].detach().numpy()
        else:
            x1 = X_complete.reshape(-1).detach().numpy()
            x2 = X_impute.reshape(-1).detach().numpy()
        return np.sqrt(sum((x1 - x2) ** 2) / len(x1))