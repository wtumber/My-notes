[HOME](SQ_home.md)

# Linear Regression
> [video link](https://www.youtube.com/watch?v=nk2CQITm_eo) 

* Fit a line with least-squares.
* calculate R-squared.
* Calculate a p-value for R-squared.

* R^2 can reduce as the number of params increases because these parameters could reduce R^2 by complete chance, therefore making the model appear better, so use adjusted R^2 when the number of params increases.

* The p-value for R^2 comis derived from F - the variation that is explained by the fit divided by the variation that is not explained by fit.

# Multiple Regression
> [video link](https://www.youtube.com/watch?v=zITIFTsivN8)

Simple regression is fitting a line to data. Multiple regression is fitting a plane to data, just adding additional factors (dimensions) to the equation.

* We can still use R^2 for multiple regression, just adjusted for additional params.
* We can calculate p-values in the same way too.
* If the difference in R^2 between simple and multiple regression is big and the P-value difference is small then the new dimension is useful.

# t-tests and ANOVA
> [video link](https://www.youtube.com/watch?v=NF5_btOaCig)

* Uses a Design Matrix.

1. Find the overall mean, then find SS residuals around the mean.
2. Fit a line to the data - just fitting a line for each group - control and mutant group - this will just be the mean line.
3. We can combine the two mean lines into one by calculating the group mean plus the residual for each point. 
4. This makes a design matrix (6:58).
5. We can then use this to calculate F.
6. This can give us a P-value.

# Design Matrices
> [video link](https://www.youtube.com/watch?v=2UYx-qjJGSs)

Design matrices essentially turn on or off the term for each group so that each value from the group is just the residual and the mean in relation to that group.

We can use design matrices to account for a batch effect.