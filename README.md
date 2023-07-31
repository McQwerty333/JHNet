# JHNet
Multi-Layer Perceptron built with Numpy  
Inspired by The Independent Code(Youtube)  

Evolution:  
1. neural_network.py :  
   One layer dense, from n inputs to 1 output.  Incapable of training to solve XOR due to lack of biases.  Successful on linear decision problems such as OR and X1, where X1 is simply the first input.
2. JHNet_01.py :  
   With biases and sigmoid activation function, a single dense layer is now able to train to solve OR, AND, and XOR.  Sigmoid and Softmax activation functions are implemented and working, as well as two functions to calculate the gradient, derivative of mean squared error and derivative of categorical cross entropy.  
3. 
