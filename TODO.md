1. No separate learning rate, I've register the learning rate of recurrent weights, not sure, about the effectiveness...
    I change the registeration code and find the result does not change, therefore the registration's modification of learning rate is not effective. 
2. Check the number of parameters, number of MLP layers...
3. Check the dropout
4. Check the structure. (layer normalization / skip, stuffs like this)

5. Current performance of 120M over large learning is against the expectation. 