# mixtures


## Examples

```python
gmm = mixtures.FiniteGMM(4, [0,4,8,16], [1,1,2,3], [.2,.2,.2,.4])
gmm.data_gen(1000)
gmm.plot_data(alpha = .5)
```

![mixture.png]

```python
gmm.gibbs(iters=500)
gmm.plot_results(alpha = .5)
```