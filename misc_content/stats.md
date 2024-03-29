## Traditional Hypothesis testing
[Cheat sheet 1](https://cheatography.com/mmmmy/cheat-sheets/hypothesis-testing-cheatsheet/)

[Cheat sheet 2](https://web.mit.edu/~csvoss/Public/usabo/stats_handout.pdf)

* create H0 (null) and H1 (alternate).
* Define significance level (eg. 0.05). If two-tailed then 0.05 is 0.025 each side
* Find z-score
* Find test statistic
* sufficient evidence to reject the null or not?

## stats to consider
Methods:
* measures of central tendency
	* Mean, Mode, median.
* Skewness
* Variance & covariance
* Correlations
* Discrete distributions
	* Poisson,equiprobable, Binomial ( X ~ B(10,0.6) ) 
* Continuous distributions
	* Gaussian, uniform, chi-squared, exponential, students t, logistic.
* CLT - use sampling distribution of the means to approximate normal dist where mean is the same as the population mean, usually sample size n>30


---------------


# 365 Data Science Probability course notes

## Combinatorics
### permutations
num ways to arrange elements
## Bayes rule
finding causal relationship between symptoms


# Discrete dists
finitely many discrete outcomes

1. equiprobable - uniform
E(x) has no real value
Each outcome is equallty likely so mean and variance is uninterpretable
2. t/f - bernoulli dist
X ~ Bern(p) - p is prob success
outcomes over several follow binomial
1 trial and 2 outcomes
3. similar experiment several time sin a row- binomial
Binomial is a sequence of bernoulli
B(n,p)
X ~ B(10,0.6)
4. poisson - how unusual an event is in a given dist


# continuous dists
1. normal
2. students t
3. chi-squared - non negative, skewed left,
4. hypothesis testing for goodness of fit
5. exponential distribution - exponential events
6. logistic dist - for forecasting

# Data Science and probability
Monte Carlo Simulations

====================

# Distributions
Scribblings from testing distfit in Python.


* https://stackoverflow.com/questions/6615489/fitting-distributions-goodness-of-fit-p-value-is-it-possible-to-do-this-with/16651524#16651524
* https://pypi.org/project/distfit/
* https://glowingpython.blogspot.com/2012/07/distribution-fitting-with-scipy.html
* https://pythonhealthcare.org/2018/05/03/81-distribution-fitting-to-data/ 
* https://machinelearningmastery.com/a-gentle-introduction-to-normality-tests-in-python/ https://stackoverflow.com/questions/37487830/how-to-find-probability-distribution-and-parameters-for-real-data-python-3 
* https://greenteapress.com/thinkstats2/html/index.html
* https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python#
* Is it parametric or non-parametric
* Doane's for correct number of histogram bins
* Quantile-Quantile plot
