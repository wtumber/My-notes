[HOME](README.md)

# Gradient Boosting
> [video link](https://www.youtube.com/watch?v=3CC4N4z3GJc)

> [part 2 link](https://www.youtube.com/watch?v=2xudPOBz-vs)

> [part 3 link](https://www.youtube.com/watch?v=jxuNLH5dXCs)

> [part 4 link](https://www.youtube.com/watch?v=StWY5QWMXCw)

# Gradient Boosting
Most common way gradient boosting is used to predict a continuous value, like
weight

## Gradient Boosting for regression
* Gradient boosting starts with a single leaf - representing an initial guess - the average value
* Gradient boost then builds a tree based on the errors of a previous tree.

### The most common configuration
1. first guess is the average value.
2. Build a tree based on the guess - using differences between observed and predicted - which is a pseudo-residual.
3. This second tree predicts these residuals.
4. You can restrict the total number of leaves - typically between 8 and 32
5. Combine the original leaf with the new tree.
6. Gradient boosting deals with overfitting (low bias, high variance) using a learning rate to scale the contribution of the new tree.
7. This results in small steps in the right direction, giving better predictions with a test set.
8. Repeat the process with another tree, using new pseudo-residuals.
9. If a leaf has multiple samples replace the residuals with their averages
10. All of the trees have their learning weights scaled equally,
11. Continue chaining trees until max specified trees included or new trees do not reduce residuals.
12. To predict, start with the initial (mean) prediction and scale value by each tree.
