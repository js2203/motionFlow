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



Um mit räumlich variierender Unschärfe umzugehen, werden flexiblere blur Modelle vorgeschlagen. 
Ein Ansatz ist ein projektives projective motion path model, welches ein unscharfes Bild als gewichtete Summe einer Menge von 
transformierten scharfen Bildern darstellt.  
Ein weiterer Ansatz ist, die Kamerabewegugn als motion density function für non-uniform blur zu modellieren. 
Um die Unschärfe, die durch die Bewegung von Objekten verursacht wird, zu behandeln, segmentieren einige Methoden Bilder in Bereiche mit unterschiedlichen Arten von Unschärfe und sind somit stark von einer akkuraten Segmentierung eines unscharfen Bildes abhängig. 
Ein pixelweises lineares Bewegungsmodell wurde ebenfalls eingeführt, um mit heterogener Bewegungsunschärfe umzugehen. 
Obwohl die Bewegung als lokal linear angenommen wird, gibt es keine Annahmen über die latente Bewegung, 
was es flexibel genug macht, um einen großen Bereich möglicher Bewegungen zu behandeln. 


## Learning based motion blur removing


more flexible and efficient blue removal  
discriminative methods for non-blind deconvolution based on  
- Gaussian CRF  
- multi-layer perceptron (MLP)  
- deep convolution neural network  

most relevant work is a method based on CNN and patch-level blur type classification  
focuses on estimating the motion flow from single blurry image  

