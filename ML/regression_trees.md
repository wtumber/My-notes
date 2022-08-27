[HOME](README.md)

# Regression trees
> [video link](https://www.youtube.com/watch?v=g9c66TUylZ4)

* A type of decision tree.
* Each leaf in the tree represents a numeric value.
* The tree can bin off the values on the graph and predicts the average value in that bin if the new point is on that branch.
* Each leaf corresponds to a cluster of values.
* Trees become useful as the number of predictors increases.

## Building a regression tree
* Find a squared residual for all points based on a root node prediction.
* Use the residual to quantify prediction quality.
* We can use a residual graph to visualise the sum of squared residuals, and then work to minimise the residual for the root node.
* Select the value with the smallest residual and use that as the root node.
* Repeat this process for each new branch (using only the points below the root node threshold) to decide whether to make further branches until the residuals are low enough to be satisfied.
* Prevent overfitting by only creating new branches when there are a number of observations greater than a minimum number in that branch area.

### With multiple predictors
* Find candidates for each predictor columns by finding the lowest residual for each column, and choosing the one with the lowest overall residual.