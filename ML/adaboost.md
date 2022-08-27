[HOME](README.md)

# AdaBoost
> [video link](https://www.youtube.com/watch?v=LsK-xG1cLYA) 

We will be combining adaboost with decision trees.

* In a forest of adaboost trees, the trees are usually just a node and two leaves, a **stump**.
* Stumps are bad at making accurate classifications, so are **weak learners**.
* Each tree in a random forest has an equal vote in classification. In adaboost this is not the case.
* In AdaBoost, in a forest of stumps, the order of stump creation is important, as the errors of each stump informs the future stump.


## Example
1. Give each sample a weight to indicate how important each sample is. 
2. At first, the weight is equal.
3. Decide the variable that does the best job at classifying, picking the one with the lowest Gini as the first stump.
4. Stumps importance/ final weight is determined by how well it does at classifying the samples.
5. **Total Error** is the sum of the weights associated with teh incorrectly classified samples, which connects the the **Amount of Say**.
6. **AoS** is 0.5*log((1-TE)/TE) where TE is Total Error. A small error term is added to ensure this is not 0 or 1.
7. Modify the sample weights by emphasising the incorrectly classified sample.
8. Increase sample weight by current sample weight * e^Amount of Say
9. Decrese other samples by current sample weight * e^-Amount of Say
10. Normalise the new sample weights.
11. Repeat with the next stump, using the weighted Gini index, or by making a new sample dataset that samples each row by the amount of their sample weight. If using the latter method then on the sample dataset reset the sample weights.
12.  Make classifications by combining the amount of say for each stump that is making a prediction and choose the prediction value with the largest total say.