import numpy as np


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : (1 / (1 + np.exp(-x)))  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid
                    

    def train(self, features, targets):
        ''' Train the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Implement forward pass here 
         
            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.matmul(X, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.matmul(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation
         
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        
        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.matmul(self.weights_hidden_to_output, error)
        
        # TODO: Backpropagated error terms - Replace these values with your calculations.
        output_error_term = error
        
        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)
        
        # Weight step (input to hidden)
        delta_weights_i_h += hidden_error_term * X[:,None]
        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs[:,None]
        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features 
        
            Arguments
            ---------
            features: 1D array of feature values
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer 
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
# initial

#iterations = 100
#learning_rate = 0.1
#hidden_nodes = 2
output_nodes = 1

# 1st trial

#Progress: 100.0% ... Training loss: 0.130 ... Validation loss: 0.232
#Progress: 100.0% ... Training loss: 0.218 ... Validation loss: 0.393
#iterations = 5000
#learning_rate = 0.2
#hidden_nodes = 2

#Progress: 100.0% ... Training loss: 0.221 ... Validation loss: 0.375
#iterations = 5000
#learning_rate = 0.25
#hidden_nodes = 2

#Progress: 100.0% ... Training loss: 0.194 ... Validation loss: 0.288
#iterations = 5000
#learning_rate = 0.4
#hidden_nodes = 2

##Progress: 100.0% ... Training loss: 0.119 ... Validation loss: 0.205
#iterations = 5000
#learning_rate = 0.2
#hidden_nodes = 4

#Progress: 100.0% ... Training loss: 0.243 ... Validation loss: 0.405
#iterations = 5000
#learning_rate = 0.1
#hidden_nodes = 8

#Progress: 100.0% ... Training loss: 0.136 ... Validation loss: 0.253
#iterations = 5000
#learning_rate = 0.2
#hidden_nodes = 8

#Progress: 100.0% ... Training loss: 0.120 ... Validation loss: 0.244
#iterations = 5000
#learning_rate = 0.2
#hidden_nodes = 16

#Progress: 100.0% ... Training loss: 0.072 ... Validation loss: 0.165
#Progress: 100.0% ... Training loss: 0.077 ... Validation loss: 0.180
#iterations = 10000
#learning_rate = 0.2
#hidden_nodes = 16

#Progress: 100.0% ... Training loss: 0.080 ... Validation loss: 0.182
#iterations = 10000
#learning_rate = 0.2
#hidden_nodes = 24

#Progress: 100.0% ... Training loss: 0.150 ... Validation loss: 0.269
#iterations = 5000
#learning_rate = 0.2
#hidden_nodes = 32

# 1st release
#Progress: 100.0% ... Training loss: 0.095 ... Validation loss: 0.161
#Progress: 100.0% ... Training loss: 0.067 ... Validation loss: 0.167
#iterations = 10000
#learning_rate = 0.2
#hidden_nodes = 32

#Progress: 100.0% ... Training loss: 0.068 ... Validation loss: 0.152
#iterations = 10000
#learning_rate = 0.25
#hidden_nodes = 32

#Progress: 100.0% ... Training loss: 0.125 ... Validation loss: 0.272
#iterations = 5000
#learning_rate = 0.2
#hidden_nodes = 64

#Progress: 100.0% ... Training loss: 0.061 ... Validation loss: 0.152
#iterations = 10000
#learning_rate = 0.2
#hidden_nodes = 64

#Progress: 100.0% ... Training loss: 0.071 ... Validation loss: 0.156
#iterations = 15000
#learning_rate = 0.15
#hidden_nodes = 64

# 2nd trial

#Progress: 100.0% ... Training loss: 0.975 ... Validation loss: 1.367
#iterations = 5000
#learning_rate = 1
#hidden_nodes = 32

#Progress: 39.0% ... Training loss: 0.974 ... Validation loss: 1.367
#iterations = 5000
#learning_rate = 1
#hidden_nodes = 64

#Progress: 100.0% ... Training loss: 0.730 ... Validation loss: 1.183
#iterations = 3000
#learning_rate = 0.9
#hidden_nodes = 32

#Progress: 100.0% ... Training loss: 0.752 ... Validation loss: 1.217
#iterations = 2000
#learning_rate = 0.8
#hidden_nodes = 32

#Progress: 100.0% ... Training loss: 0.533 ... Validation loss: 0.618
#iterations = 3000
#learning_rate = 0.7
#hidden_nodes = 32

#Progress: 100.0% ... Training loss: 0.155 ... Validation loss: 0.333
#iterations = 3000
#learning_rate = 0.6
#hidden_nodes = 32

#Progress: 100.0% ... Training loss: 0.092 ... Validation loss: 0.184
#iterations = 3000
#learning_rate = 0.5
#hidden_nodes = 32

#Progress: 100.0% ... Training loss: 0.088 ... Validation loss: 0.213
#Progress: 100.0% ... Training loss: 0.078 ... Validation loss: 0.161
#Progress: 100.0% ... Training loss: 0.078 ... Validation loss: 0.161
iterations = 5000
learning_rate = 0.5
hidden_nodes = 32

#Progress: 100.0% ... Training loss: 0.099 ... Validation loss: 0.222
#iterations = 3000
#learning_rate = 0.4
#hidden_nodes = 24

#Progress: 100.0% ... Training loss: 0.080 ... Validation loss: 0.197
#iterations = 3000
#learning_rate = 0.4
#hidden_nodes = 32
