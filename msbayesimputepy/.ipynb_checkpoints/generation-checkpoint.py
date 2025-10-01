from scipy.stats import norm, truncnorm
import numpy as np
import pandas as pd

def gen_data(n_features = 200, n_samples = 50, n_factors = 5, theta_z = 1, theta_w = 1, 
            tau = 0, tau_mode = "perProt", shape_noise = 1.5, scale_noise = 1.5, reference_df = False,
            alpha_row = 0, sd_row = 1, alpha_col = 20, sd_col = 2):
    """generate data artificially by mimicing weight and factor matrix

    Parameters
    ----------
    n_features: int
        a number of features
    n_samples: int
        a number of samples/observations
    n_factors: int
        a number of factors
    theta_z: float
        a probability of factor actication 
    theta_w: float
        a probability of weight actication 
    tau: float
        the precision for likelihood
    tau_mode: str
        a character that can take the value of "perFeature" or None. "perFeature" allows the noise to varied to degrees defined
        by "shape_noise" and "scale_noise" in a per-feature manner or by reference_df.
    shape_noise: float
        the shape parameter of gamma distribution
    scale_noise: float
        the scale parameter of gamma distribution
    reference_df: 
        a reference dataframe is considered when creating noise
    alpha_row: float
        intercepts are added to row level using a normal distribution. alpha_row is the loc parameter of this distribution.
    sd_row: 
        intercepts are added to row level using a normal distribution. sd_row is the scale parameter of this distribution.
    alpha_col: 
        intercepts are added to column level using a normal distribution. alpha_col is the loc parameter of this distribution.
    sd_col: 
        intercepts are added to column level using a normal distribution. sd_col is the loc parameter of this distribution.
        
    Returns
    ----------
    a dictionary contains data, S_z, Z, S_w, W, feature_mean, sample_mean, tau
    """
    
    #generate factor activation and factor value matrix
    S_z = np.random.binomial(1, theta_z, n_samples * n_factors).reshape(n_samples, n_factors) #factor activation matrix
    Z = np.random.normal(0, 1, n_samples * n_factors).reshape(n_samples, n_factors) #factor value matrix
        
    #generate weight activation and weight matrix
    S_w = np.random.binomial(1, theta_w, n_features * n_factors).reshape(n_features, n_factors) #weight activation matrix
    W = np.random.normal(0, 1, n_features * n_factors).reshape(n_features, n_factors) #weight value matrix
        
    # generate the real value matrix without any noise
    mu = np.matmul((S_z * Z), (S_w * W).T)
    
    # add row-wise intercept
    if alpha_row is not None:
        sample_mean = np.random.normal(alpha_row, sd_row, mu.shape[0])
        mu = mu + sample_mean.reshape(mu.shape[0], 1)
    else:
        sample_mean = None
    
    # add column-wise intercept
    if alpha_col is not None:
        feature_mean = np.random.normal(alpha_col, sd_col, mu.shape[1])
        mu = mu + feature_mean.reshape(1, mu.shape[1])
    else:
        feature_mean = None
    
    #add Gaussian noise with the sd = sqrt(1/tau)
    tau_vec = False
    if tau_mode == "perProt":
        data = mu.copy()
        if isinstance(reference_df, pd.core.frame.DataFrame):
            reference_df = reference_df[reference_df.notna().sum(axis = 1) >= 2]
            shape, loc, scale = gamma.fit(reference_df.std(axis = 1, skipna = True))
            std_vec = np.random.gamma(shape, 1, n_features) * scale + loc
            tau_vec = 1/(std_vec ** 2)
        else:
            tau_vec = tau + np.random.gamma(shape_noise, scale_noise, n_features) # loc = shape_noise * scale_noise or shape_noise/rate_noise
            
        for i in range(n_features):
            data[:, i] = data[:, i] + np.random.normal(0, np.sqrt(1/tau_vec[i]), n_samples)
        data = data.T
        
    else:
        tau_vec = tau
        data = (mu + np.random.normal(0, np.sqrt(1/tau), mu.size).reshape(mu.shape)).T 
    data = pd.DataFrame(data)
    data.index = ["feature_" + str(i + 1) for i in range(data.shape[0])]
    data.columns = ["sample_" + str(j + 1) for j in range(data.shape[1])]
         
    return {"data": data,
            "S_z": S_z, "Z": Z, 
            "S_w": S_w, "W": W, 
            "feature_mean": feature_mean, 
            "sample_mean": sample_mean, 
            "tau": tau_vec}


def gen_prob_miss(X, rho, zeta, model, rho_sd = 1, zeta_sd = 1, subSample = 0, filter_threshold = 0):
    """generate missing values artificially by a probabilistic dropout method

    Parameters
    ----------
    X: DataFrame
        a complete input matrix, with features in rows and samples in columns
    rho: float
        loc parameter for dropout model
    zeta:float
        slope parameter for the drop-out model
    model: str
        a character that can take the value of "global", "perFeature" or "perSample". "global" indicates generating missing values for 
        all measurements using the same probabilistic. "perFeature" and "perSample" allow the drop-out model parameter to varied to degrees defined
        by "rho_sd" and "zeta_sd" in a per-feature or per-sample manner. 
    rho_sd: float
        the degree (standard deviation in a normal distribution) the loc of the drop-out model can vary.
    zeta_sd: float
        the degree (standard deviation in a normal distribution) the slope of the drop-out model can vary. 
    subSample: int
        if value is > 0 and < 1, features will be subsetted randomly by this percentage. 
    filter_threshold: int
        the cutoff for removing proteins within missingness

    Returns
    ----------
    outputDic: dict
        a dictionary contains X_miss, X_complete, dropMat, rho_vec, zeta_vec
    """
    
    outputDic = {}
    if subSample > 0 and subSample < 1:
        X = X.iloc[random.sample(range(X.shape[0]), int(X.shape[0] * subSample))]
    
    # generate the drop-out probability matrix for each observation
    if model == "global":
        X_drop = norm.sf(X, rho, zeta)
        dropMat = pd.DataFrame(X_drop, index = X.index, columns = X.columns)
        
    elif model == "perFeature":
        rho_vec = norm.rvs(rho, rho_sd, X.shape[0])
        zeta_vec = truncnorm.rvs((0 - zeta) / zeta_sd, np.inf, zeta, zeta_sd, X.shape[0])
        dropMat = np.array([norm.sf(X.iloc[i], rho_vec[i], zeta_vec[i]) for i in range(X.shape[0])])
            
        dropMat = pd.DataFrame(dropMat, index = X.index, columns =  X.columns)
        outputDic["rho_vec"] = np.array([X.index, rho_vec])
        outputDic["zeta_vec"] = np.array([X.index, zeta_vec])
        
    elif model == "perSample":
        rho_vec = norm.rvs(rho, rho_sd, X.shape[1])
        zeta_vec = truncnorm.rvs((0 - zeta) / zeta_sd, np.inf, zeta, zeta_sd, X.shape[1])
        dropMat = np.array([norm.sf(X.iloc[:, i], rho_vec[i], zeta_vec[i]) for i in range(X.shape[1])]).T
            
        dropMat = pd.DataFrame(dropMat, index = X.index, columns =  X.columns)
        outputDic["rho_vec"] = np.array([X.columns, rho_vec])
        outputDic["zeta_vec"] = np.array([X.columns, zeta_vec])
    
    # create missing values randomly based on the drop-out probability
    missMask = np.array([bool(np.random.choice([True, False], size = 1, p = [val, 1-val])) for val in dropMat.values.flatten()])
    
    missMask = missMask.reshape(-1, X.shape[1])
    X_miss = X.copy()
    X_complete = X.copy()
    X_miss[missMask] = np.nan
    
    #it is possible that all values will become NA for a certain protein or sample, they need to be removed
    X_miss = X_miss.loc[X_miss.notna().sum(axis = 1) > filter_threshold, X_miss.notna().sum(axis = 0) > 0]
    X_complete = X_complete.loc[X_miss.index, X_miss.columns]
    dropMat = dropMat.loc[X_miss.index, X_miss.columns]
    outputDic["X_miss"] = np.round(X_miss, 5)
    outputDic["X_complete"] = np.round(X_complete, 5)
    outputDic["dropMat"] = np.round(dropMat, 5)

    return outputDic