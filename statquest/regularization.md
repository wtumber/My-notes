[HOME](README.md)

# Regularization
> [video link](https://www.youtube.com/watch?v=Q81RR3yKn30)
> [part 2 link](https://www.youtube.com/watch?v=NGf0voTMlcs)

## Ridge regression
1. As usual, fit a line using least squares.
2. When the number of measurements is small then the SS is very low. This line is high in Variance and overfitted.
3. Ridge Regression introduces a Bias so that it doesn't fit the training data as well, significantly dropping variance.

We can solve where number parameters is much higher than the number of data points available using the ridge regression penalty and Cross Validation.

## Lasso regression