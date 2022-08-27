[HOME](README.md)

# Regularization
> [video link](https://www.youtube.com/watch?v=Q81RR3yKn30)

> [part 2 link](https://www.youtube.com/watch?v=NGf0voTMlcs)

## Ridge regression (L2 regularization)
Note that SS  means Sum of the Square residuals.

1. As usual, fit a line using least squares.
2. When the number of measurements is small then the SS is very low. This line is high in Variance and overfitted.
3. Ridge Regression introduces a Bias so that it doesn't fit the training data as well, significantly dropping variance.

We can solve where number parameters is much higher than the number of data points available using the ridge regression penalty and Cross Validation.

So Ridge regression is = Least Squares + ridge regression penalty. It reduces variance and provides better long term predictions.

## Lasso Regression (L1 regularization)
absolute value of slope instead of squaring. The lambda penalty is
acquired with cross-validation.

Both regression methods are very similar, and reduce sensitivity
to very small training sets.

# The differences
When lambda = 0 then the least squares line is the same as the lasso
regression line. Ridge cannot shrink the slope to 0, just
asymptotically close.

Lasso regression can remove useless variables, but is slightly worse
when most variables are useful.
