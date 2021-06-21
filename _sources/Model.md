# Estimating Motion Flow for Blur Removal


## Heterogeneous Motion Blur Model


$$Y = K ∗ X + N$$


* Y = unscharfes Bild
* K = heterogene motion blur kernel map mit verschiedenen motion blur kernel für jeden Pixel in X
* ∗ = allgemeiner convolution operator
* X = latentes scharfes Bild
* N = zusätzliches Bildrauschen


$$Y(i, j) = \sum\limits_{i´, j´} K_{(i, j)} (i´, j´) X (i + i´, j +j´)$$


* K(i, j) repräsentiert den Kernel aus K, der auf den Pixel (i, j) zentriert ist


$$y = H(K)x + n;$$


* vec() = vectorises a matrix
* y = vec(Y) 
* x = vec(X)
* n = vec(n)
* $H(K) ∈ \mathbb{R}^{PQ✖PQ}$ (each row corresponds to a blur kernel located at each pixel)

```python

```
