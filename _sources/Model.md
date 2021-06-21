# Estimating Motion Flow for Blur Removal


## Heterogeneous Motion Blur Model


$$Y = K ∗ X + N$$


* Y = unscharfes Bild
* K = heterogene motion blur kernel map mit verschiedenen motion blur kernel für jeden Pixel in X
* ∗ = allgemeiner convolution operator
* X = latentes scharfes Bild
* N = zusätzliches Bildrauschen


$$Y(i, j) = \sum\limits_{i´, j´} K~(i, j)~ - \frac{1}{x}$$

```python

```
