# Mixtures

A python package for working with Gaussian Mixture Models. You can either generate your own data and fit models to help understand how the models work, or you can work directly with another non-synthetic data source.
# Install

```
git clone https://github.com/blazickjp/mixtures.git
cd mixtures
pip install .
```

# Examples
## Finite Gaussian Mixture Models

Define a mixture model and generate sample data according to the model. We can then fit a mixture model to the data and compare our fitted model to actuals.

```python
import mixtures
import numpy as np

gmm = mixtures.FiniteGMM(k = 4, mu = [0,4,8,16], sigma = [1,1,2,3], phi = [.2,.2,.2,.4])
gmm.data_gen(1000)
gmm.plot_data(alpha = .5, bins = 50)
```

<p align="center">
  <img src="images/mixture4.png" width="400" height="300" title="hover text">
</p>

```python
gmm.gibbs(iters=500, burnin=50)
gmm.plot_results(alpha = .5, bins=50)
```

<p align="center">
  <img src="images/gmm_fitted.png" width="400" height="300" title="hover text">
</p>

We can also generate and fit multitvariate data by simply providing multivariate parameters to the model. The ```FiniteGMM``` class will detect the shape of the inputs and set ```self.multivariate = True```. This tells the class to take different actions on the inputs and uses a collapsed gibbs sampler for fitting the model.

```python
# Define parameters for K=3 Mixture of Multivariate Gaussians

phi = [.3, .5, .2]
mu = np.array([[13,5], [0,-2], [-14,3]])
cov_1 = np.array([[2.0, 0.3], [0.3, 0.5]])
cov_2 = np.array([[3.0,.4], [.4,3.0]])    
cov_3 = np.array([[1.7,-.7], [-.7,1.7]])
cov = np.stack((cov_1, cov_2, cov_3), axis = 0)

gmm = mixtures.FiniteGMM(3, mu, cov, phi)
```

Now we can use the same functionality as in the univariate case.

```python
gmm.data_gen(1000)
gmm.plot_data(alpha = .5)
```

<p align="center">
  <img src="images/m_variate_data.png" width="400" height="300" title="hover text">
</p>

```python
gmm.gibbs(iters=20)
gmm.plot_results(alpha = .5)
```
<p align="center">
  <img src="images/m_variate_fitted.png" width="400" height="300" title="hover text">
</p>

## Infinite Gaussian Mixture Models

The API for working with Infitine Mixture Models is exactly the same except we call ```InfitineGMM``` instead of ```FiniteGMM```. The Infinite GMM model will fit both multivariate and univarite data. The ```InfiniteGMM``` class inherits from the ```FiniteGMM``` class so you get the sample plotting features as in the finite class.

```python
gmm = mixtures.InfiniteGMM(4, [0,4,8,16], [1,1,2,3], [.2,.2,.2,.4])
gmm.data_gen(1000)
gmm.gibbs(initial_k=3,iters=50)
gmm.plot_results()
```

<p align="center">
  <img src="images/infinitegmm.png" width="400" height="300" title="hover text">
</p>