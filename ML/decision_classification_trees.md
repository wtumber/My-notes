[HOME](README.md)

# Decision and Classification Trees
> [video link](https://www.youtube.com/watch?v=_L39rN6gz7Y) 

## Basic concepts
* Statement is used to make a decision based on if it is true or false.
* Can be used for regression (e.g. less than 100mm length and more than 100mm length), and classification.
* start at the top and work down
* TRUE goes LEFT, FALSE goes right
* Top of tree is root node
* Internal node/branches have arrows in and arrows out
* Leaf nodes have arrows in but not out

## Starting a tree
You can quantify the impurity of leaves from a root node to decide what to use as the root. An example impurity is the *Gini* impurity. You calculate the impurity for each leaf and then take a weighted average for an overal impurity value for that root node.

When the root node column is a numeric value instead of boolean you must calculate the gini impurity for each numeric value, then test that value as the root node and find the weighted value.

Select a value with the lowest impurity as the root node.

Repeat this process with each branch to continue lowering the impurity.

We can prune or limit a tree to minimise overfitting.
