[HOME](SQ_home.md)

# Cross Validation
> [video link](https://www.youtube.com/watch?v=fSytzGwwBVw) 

We can use Cross Validation to compare ML methods and get a sense of how they work in practice.

We need the data to:
1. Train the ML methods.
2. Test the ML methods.

We can split the dataset into train and test to do this, but how do we know that this split will work well?

We can use cross validation to test all blocks so that we don't have to worry about this. The number of blocks is the value n in **n-fold cross validation**. We can use all samples as block, testing each sample individually, something called **Leave One Out Cross Validation**.

We can use cross validation to find tuning parameters too.