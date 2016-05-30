# sisfft

A Python implementation of sisFFT and aFFT-C.

See also: an R implementation https://github.com/huonw/sisfft

Example of using sisFFT to compute a pvalue, along with doing a raw
convolution power:

```Python
import sisfft, numpy

log_pmf = numpy.log([0.1, 0.3, 0.2, 0.4])
L = 10 # 10-fold convolution
s0 = 25 # threshold
error_limit = 1e-3

# compute the pvalue of s0 directly
print repr(numpy.exp(sisfft.log_pvalue(log_pmf, s0, L, error_limit)))
# => 0.044086067199999975

# now compute the full 10-fold convolution (delta limit of zero means no truncation)
log_full_conv = sisfft.log_convolve_power(log_pmf, L, error_limit, 0.0)

# use this to compute the pvalue
pvalue = sum(numpy.exp(log_full_conv)[s0:])
print repr(pvalue)
# => 0.044086067199999877
```

Example of aFFT-C:

```Python
import sisfft, numpy

# two pmfs
pmf1 = [0.1, 0.3, 0.2, 0.4]
pmf2 = [0.25, 0.25, 0.25, 0.25]

log_pmf1 = numpy.log(pmf1)
log_pmf2 = numpy.log(pmf2)

beta = 1e-3 # accuracy parameter

# Compute pmf1 * pmf2 in two ways:
# first, with numpy's built in convolve
direct = numpy.convolve(pmf1, pmf2)
# and then with aFFT-C
via_afftc = sisfft.log_convolve(log_pmf1, log_pmf2, beta)

print direct
# => array([ 0.025,  0.1  ,  0.15 ,  0.25 ,  0.225,  0.15 ,  0.1  ])
print numpy.exp(via_afftc)
# => array([ 0.025,  0.1  ,  0.15 ,  0.25 ,  0.225,  0.15 ,  0.1  ])

# They're not exactly the same, but aFFT-C is well within the error bounds:
rel_error = (numpy.exp(via_afftc) - direct) / direct
print rel_error
# => array([ -4.16333634e-16,  -6.93889390e-16,   1.85037171e-16,
#             0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
#            -2.77555756e-16])
```
