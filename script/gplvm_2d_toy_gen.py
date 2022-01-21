import numpy as np
import matplotlib.pyplot as plt
from GPy.models import BayesianGPLVM
from sklearn.datasets import make_moons



if __name__ == "__main__":
	np.random.seed(42); plt.ion(); plt.style.use('ggplot')
	n = 500
	Y, c = make_moons(n, noise=0.1)
	model = BayesianGPLVM(Y, 2, init='PCA')
	model.optimize_restarts(10, optimizer='lbfgs')
 
	E_ygY, V_ygY = model.predict(np.random.normal(size=(500, 2)))
	Y_samp = E_ygY + np.random.normal(0, np.sqrt(V_ygY))
	plt.scatter(Y_samp[:, 0], Y_samp[:, 1], label='sample')
	plt.scatter(Y[:, 0], Y[:, 1], alpha=0.3, label='data')
	plt.legend()
