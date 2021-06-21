# Deblurring Theorien


## Konventionelles blind image deblurring

Bei den konventionellen blind image deblurring Methoden ist eine der grunsätzlichen Annahmen, dass die Bewegungsunschärfe spatially uniform ist und nicht heterogenous.  
In diesen Theorien wurden bereits mehrere Ansätze genauer betrachtet, um die Bewegungsunschärfe zu entfernen. Dazu gehören unter anderem:  
* total variational regularizer
* Gaussian scale mixture priors
* L1/L2-norms, L0-norms
* dark channel based regularizers  

Zusätzlich wurden mehrere Estimator untersucht, um robustere und zuverlässige Kernel zu berechnen. Mögliche Estimator sind:  
* edge-extraction-based maximum-a-posteriori (MAP)
* gradient activation based MAP
* variational Bayesian methods

Die Theorien, welche sich dieser Methoden annehmen, sind dabei aber sehr stark abhängig von den ersten Annahmen und Priors. Der praktische Einsatz dieser Theorien wird hierduch eingeschränkt.


## Spatially-varying blur removal



## Learning based motion blur removing
