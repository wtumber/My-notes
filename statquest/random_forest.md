[HOME](README.md)

# Random Forests
> [video link](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
> [part 2 link](https://www.youtube.com/watch?v=sQ870aTKqiM)

* Built from decision trees
* Decision trees are not flexible when it comes to new samples, which is why we use random forests.

## Making a random forest.
* Create a boostrap dataset by taking random samples from the dataset. Samples do not have to be unique - rows can be reselected.
* Create decision tree for the bootstrapped data, only considering "n" predictor columns from the dataset.
* Repeat this process hundreds of times to make a variety of decision trees.
* When exposed to new data, run it through each tree, keeping track for all trees. We then use this to conclude a value.
* Bootstrapping the data plus using the aggregate is **Bagging**.
* There exists an out-of-bag dataset where rows are not selected for boostrap datasets. We can use the out-of-bag samples to measure the proportion of samples which were correctly predicted by the random forest. This value is the **out-of-bag error**.
* We can compare the out-of-bag error for "n" predictor columns on each bootstrapped decision tree and evaluate the accuracy, choosing the "n" that gives the highest accuracy.

## Missing data and clustering
* If data is missing make an initial guess for the missing values that could be bad, then refine the guess.
* Determine similarity by building random forests and running the data down them. When samples end up at the same leaf nodes they are similar so add 1 on the proximity matrix in the corresponding row-row cell to show they are similar. 
* Divide the values in the proximity matrix by the total number of trees.
* Calculate the weighted frequency of the missing values, using proximity values as the weight, and the frequency of that value occuring. The proximity value is the proximity values in the rows where the value occurs. Divide the weight by the sum of proximities for the sample which is missing data for the weighted frequency.
* Decide the value of the missing value based on the value with the heigher weighted average, or take the weighted average if continuous (7:30 on video 2).
* Repeat this process over and over until the missing values converge.

We can draw a heatmap of the proximity matrix and an MDS plot.

With a binary missing value duplicate the row, fill in with both True and False, then make a prediction using the decision trees. Whichever value is correct most often wins.