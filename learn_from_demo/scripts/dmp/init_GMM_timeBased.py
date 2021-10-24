import numpy as np

def init_GMM_timeBased(sIn, param):
    # Parameters
    params_diagRegFact = 1E-4
    # divide the phase variable into several pieces
    TimingSep = np.linspace(np.amin(sIn), np.amax(sIn), param.nbStates+1)

    Priors = np.zeros(param.nbStates)
    Mu = np.zeros(param.nbStates)
    Sigma = np.zeros(param.nbStates)

    for i in range(param.nbStates):
        # find the time points of each phase variable section
        idtmp = np.argwhere((sIn >= TimingSep[i]) & (sIn < TimingSep[i+1]))
        Priors[i] = np.size(idtmp)  # the number in each section
        Mu[i] = np.mean(sIn[idtmp])  # calculate the mean value of each section
        Sigma[i] = np.var(sIn[idtmp])  # calculate the variance of each section
        # optional regularization term to avoid numerical instability
        Sigma[i] = Sigma[i] + np.eye(1)*params_diagRegFact
    Priors = Priors / np.sum(Priors)

    class GMM_model:
        priors = Priors
        mu = Mu
        sigma = Sigma
        std = np.sqrt(Sigma)
        params_diagregfact = params_diagRegFact

    gmm_model = GMM_model()

    return gmm_model
    # print(param.Priors)

