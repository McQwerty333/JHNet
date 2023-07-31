# JHNet
Multi-Layer Perceptron built with Numpy  
Inspired by The Independent Code  

Evolution:  
1. neural_network.py :  
   One layer dense, from n inputs to 1 output.  Incapable of training to solve XOR due to lack of biases.  Successful on linear decision problems such as OR and X1, where X1 is simply the first input.
2. JHNet_01.py :  
   With biases and sigmoid activation function, a single dense layer is now able to train to solve OR, AND, and XOR.  Sigmoid and Softmax activation functions are implemented and working, as well as two functions to calculate the gradient, derivative of mean squared error and derivative of categorical cross entropy.  
3. JHNet_02.py :
   Added a network class to allow multiple dense layers to be built and trained together and simplify execution.  Applies sigmoid activation after each dense layer.  
  
Over the next few iterations, the option to choose hidden layer as well as output layer activations was added to the network builder, as well as choice of gradient calculation function.  With sigmoid on hidden layers, softmax on output, and derivative of cross categorical entropy as gradient, a network with 1 hidden layer of 10 nodes was able to memorize 10 samples from the mnist handwritten digits dataset. Memorize, not learn, since with only 10 samples, it would be unable to generalize correct results to any other unseen images.  

4. JHNet_06.py , pandas_03.py  , mnist_train.csv  
   The first usage of full mnist training dataset, which is shuffled and split into 50,000 training samples and 10,000 validation samples in pandas_03.py.  Create a network by defining the size of each layer, choosing activation for hidden layers and output, and a gradient function.  Training via one iteration over the full batch is slow but yields roughly 75% accuracy on both training and validation.  Stochastic gradient descent over the full dataset in one iteration is much faster, though yields about 60% accuracy.  Minibatch gradient descent is the fastest, and in one iteration results range from about 55% to 75% accuracy.  
   
I experimented deeply with these training methods, trying several epochs with and without data shuffling, as well as experimenting with bootstrap style sampling.  Large batch training climbed to about 78% accuracy over 1000 epochs, while other methods peaked at around 75% accuracy.

At this point, I moved on to experimenting with gradient descent optimizers.  I worked on implementing optimizers both on this network, and another project of mine, the gradient descent optimizer visualizer.  The visualizer is great for seeing how different optimizers perform on various 3D surfaces, to help understand their strengths and weaknesses.  Formulas and intuition provided by Raimi Karim.  

5. JHNet_17.py  
   10 optimizers are implemented in Dense Layer backpropagation.  Xavier Glorot initialization is also used, which improves speed of training to 70% accuracy to just 1 epoch for most proficient optimizers.  With data class integrated, as well as new tester class to find average accuracy across training sessions.  As training large batch is very slow, most hyperparameters were adjusted based on minibatch training.  Learning rates found to be successful are listed next to each optimizer.  Best performers on average were momentum, Nesterov momentum, Adam, Nadam, Adamax, and AMSgrad, averaging about 76% accuracy on 5 epochs.  Some would peak around 80% during training.  Unoptimized (vanilla) gradient descent achieves roughly 70% accuracy on average.  The network is still using sigmoid on hidden, softmax on output, and derivative of cross categorical entropy for gradient.  

There is still much improvment to be made.  Promising ideas include regularization via dropout, batch normalization, or weight decay, as well as additive white gaussian noise and random translation on training samples to make it capable of recognizing handwriting samples from outside of the mnist set, as mnist is size-normalized and centered.  Another direction to travel is towards a convolutional neural net.  A more simple solution may be trying different activations, such as ReLu, or more nodes in hidden layers.
