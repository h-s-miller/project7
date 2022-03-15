# BMI 203 Project 7: Neural Network


# Importing Dependencies
import numpy as np
from typing import List, Dict, Tuple, Union
from numpy.typing import ArrayLike


# Neural Network Class Definition
class NeuralNetwork:
    """
    This is a neural network class that generates a fully connected Neural Network.

    Parameters:
        nn_arch: List[Dict[str, float]]
            This list of dictionaries describes the fully connected layers of the artificial neural network.
            e.g. [{'input_dim': 64, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation:': 'sigmoid'}] will generate a
            2 layer deep fully connected network with an input dimension of 64, a 32 dimension hidden layer
            and an 8 dimensional output.
        lr: float
            Learning Rate (alpha).
        seed: int
            Random seed to ensure reproducibility.
        batch_size: int
            Size of mini-batches used for training.
        epochs: int
            Max number of epochs for training.
        loss_function: str
            Name of loss function.

    Attributes:
        arch: list of dicts
            This list of dictionaries describing the fully connected layers of the artificial neural network.
    """
    def __init__(self,
                 nn_arch: List[Dict[str, Union[int, str]]],
                 lr: float,
                 seed: int,
                 batch_size: int,
                 epochs: int,
                 loss_function: str):
        # Saving architecture
        self.arch = nn_arch
        # Saving hyperparameters
        self._lr = lr
        self._seed = seed
        self._epochs = epochs
        self._loss_func = loss_function
        self._batch_size = batch_size
        # Initializing the parameter dictionary for use in training
        self._param_dict = self._init_params()

    def _init_params(self) -> Dict[str, ArrayLike]:
        """
        DO NOT MODIFY THIS METHOD!! IT IS ALREADY COMPLETE!!

        This method generates the parameter matrices for all layers of
        the neural network. This function returns the param_dict after
        initialization.

        Returns:
            param_dict: Dict[str, ArrayLike]
                Dictionary of parameters in neural network.
        """
        # seeding numpy random
        np.random.seed(self._seed)
        # defining parameter dictionary
        param_dict = {}
        # initializing all layers in the NN
        for idx, layer in enumerate(self.arch):
            layer_idx = idx + 1
            input_dim = layer['input_dim']
            output_dim = layer['output_dim']
            # initializing weight matrices
            param_dict['W' + str(layer_idx)] = np.random.randn(output_dim, input_dim) * 0.1
            # initializing bias matrices
            param_dict['b' + str(layer_idx)] = np.random.randn(output_dim, 1) * 0.1
        return param_dict

    def _single_forward(self,
                        W_curr: ArrayLike,
                        b_curr: ArrayLike,
                        A_prev: ArrayLike,
                        activation: str) -> Tuple[ArrayLike, ArrayLike]:
        """
        This method is used for a single forward pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            activation: str
                Name of activation function for current layer.

        Returns:
            A_curr: ArrayLike
                Current layer activation matrix.
            Z_curr: ArrayLike
                Current layer linear transformed matrix.
        """
        print('A previous shape {}'.format(A_prev.shape))
        print('W curr.T shape {}'.format(W_curr.T.shape))
        print('b_curr.T shape {}'.format(b_curr.T.shape))
        ## Calculate Z=A*W+b ##
        # 16x64 * 64x96 + 16x1
        Z_curr=np.add(A_prev.dot(W_curr.T),b_curr)
        
        ## Activation ##
        if activation=='sigmoid':
            A_curr=self._sigmoid(Z_curr)
        elif activation=='relu':
            A_curr=self._relu(Z_curr)
        else:
            raise ValueError("{} is not a defined activation function".format(activation))
        
        return A_curr, Z_curr
        

    def forward(self, X: ArrayLike) -> Tuple[ArrayLike, Dict[str, ArrayLike]]:
        """
        This method is responsible for one forward pass of the entire neural network.

        Args:
            X: ArrayLike
                Input matrix with shape [batch_size, features].

        Returns:
            output: ArrayLike
                Output of forward pass.
            cache: Dict[str, ArrayLike]:
                Dictionary storing Z and A matrices from `_single_forward` for use in backprop.
        """
        
        ### instantiate cache ###
        cache={}
        
        ## first layer is input ##
        A_prev=X
        cache['A0']=A_prev #save X in cache
        
        for idx,layer in enumerate(self.arch):
 
            
            ## load the current weights and biases ##
            layer_idx=idx+1
            W_curr=self._param_dict['W'+str(layer_idx)]
            b_curr=self._param_dict['b'+str(layer_idx)]
            
            ## calculate forward pass ##
            A_curr, Z_curr = self._single_forward(W_curr, b_curr, A_prev, layer['activation'])
            
            ## save A and Z ##
            cache['A'+str(layer_idx)]=A_curr
            cache['Z'+str(layer_idx)]=Z_curr
            
            ## update ##
            A_prev= A_curr
        
        return A_curr, cache
            

    def _single_backprop(self,
                         W_curr: ArrayLike,
                         b_curr: ArrayLike,
                         Z_curr: ArrayLike,
                         A_prev: ArrayLike,
                         dA_curr: ArrayLike,
                         activation_curr: str) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        This method is used for a single backprop pass on a single layer.

        Args:
            W_curr: ArrayLike
                Current layer weight matrix.
            b_curr: ArrayLike
                Current layer bias matrix.
            Z_curr: ArrayLike
                Current layer linear transform matrix.
            A_prev: ArrayLike
                Previous layer activation matrix.
            dA_curr: ArrayLike
                Partial derivative of loss function with respect to current layer activation matrix.
            activation_curr: str
                Name of activation function of layer.

        Returns:
            dA_prev: ArrayLike
                Partial derivative of loss function with respect to previous layer activation matrix.
            dW_curr: ArrayLike
                Partial derivative of loss function with respect to current layer weight matrix.
            db_curr: ArrayLike
                Partial derivative of loss function with respect to current layer bias matrix.
        """
        ## get dims of A ##
        dim = A_prev.shape[1]

        ## derivative of the activation ##
        if activation_curr == 'relu':
            dZ_curr = self._relu_backprop(Z_curr)
        elif activation_curr == 'sigmoid':
            dZ_curr = self._sigmoid_backprop(Z_curr)

        ## backprop calulcation ##
        dW_curr = np.dot(W_curr.T, dZ_curr) # partial Z/ partial W_L
        dA_prev = np.dot(dZ_curr, A_prev.T) / A_dims # partial a_L/ partial Z
        db_curr = np.sum(dZ_curr, axis = 1, keepdims = True) / A_dims # partial

        return dA_prev, dW_curr, db_cur

    def backprop(self, y: ArrayLike, y_hat: ArrayLike, cache: Dict[str, ArrayLike]):
        """
        This method is responsible for the backprop of the whole fully connected neural network.

        Args:
            y (array-like):
                Ground truth labels.
            y_hat: ArrayLike
                Predicted output values.
            cache: Dict[str, ArrayLike]
                Dictionary containing the information about the
                most recent forward pass, specifically A and Z matrices.

        Returns:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from this pass of backprop.
        """
        ### define grad_dict ###
        grad_dict = {}

        ### intialize dA curr with the loss function ###
        if self._loss_func == 'mse':
            dA_curr = self._mean_squared_error_backprop(y, y_hat)
        elif self._loss_func == 'bce':
            dA_curr = self._binary_cross_entropy_backprop(y, y_hat)

        ### go thru layers in reverse order ###
        for i, layer in reversed(list(enumerate(self.arch))):
            idxLayer = i + 1

            ### get the weights and biases ###
            weight_curr = self._param_dict['W' + str(idxLayer)]
            bias_curr = self._param_dict['b' + str(idxLayer)]
            
            ### load Z and A ##
            Z_curr =  cache['Z' + str(idxLayer)]
            A_prev =  cache['A' + str(idxLayer - 1)]

            ### backpropograte ##
            dA_prev, dW_curr, db_curr = self._single_backprop(weight_curr,
                                        bias_curr, Z_curr, A_prev, dA_curr,
                                        layer['activation'])

            ### save results ###
            grad_dict['dW' + str(idxLayer)] = dW_curr
            grad_dict['db' + str(idxLayer)] = db_curr

            ## move backwards ##
            dA_curr = dA_prev

        return grad_dict

    def _update_params(self, grad_dict: Dict[str, ArrayLike]):
        """
        This function updates the parameters in the neural network after backprop. This function
        only modifies internal attributes and thus does not return anything

        Args:
            grad_dict: Dict[str, ArrayLike]
                Dictionary containing the gradient information from most recent round of backprop.

        Returns:
            None
        """
        for i, layer in enumerate(self.arch):
            idxLayer = i + 1

            ### W= W - lr*dW ###
            self._param_dict['W' + str(idxLayer)] -= self._lr * grad_dict['dW' + str(idxLayer)]
            
            ### b= b - lr*db ###
            self._param_dict['b' + str(idxLayer)] -=self._lr * grad_dict['db' + str(idxLayer)]

    def fit(self,
            X_train: ArrayLike,
            y_train: ArrayLike,
            X_val: ArrayLike,
            y_val: ArrayLike) -> Tuple[List[float], List[float]]:
        """
        This function trains the neural network via training for the number of epochs defined at
        the initialization of this class instance.
        Args:
            X_train: ArrayLike
                Input features of training set.
            y_train: ArrayLike
                Labels for training set.
            X_val: ArrayLike
                Input features of validation set.
            y_val: ArrayLike
                Labels for validation set.

        Returns:
            per_epoch_loss_train: List[float]
                List of per epoch loss for training set.
            per_epoch_loss_val: List[float]
                List of per epoch loss for validation set.
        """
        ### keep track of losses ##
        per_epoch_loss_train = []
        per_epoch_loss_val = []

        ### loop thru epochs ###
        for e in range(self._epochs):
            ## shuffle the data so we get different splits when splitting into mini batches ##
            idx = np.random.permutation(len(X_train))
            X_train_shuffle,y_train_shuffle = X_train[idx], y_train[idx]

            
            ### spllit into batches ###
            X_batch = np.array_split(X_train_shuffle, self._batch_size)
            y_batch = np.array_split(y_train_shuffle, self._batch_size)

            ### keep track of losses per batch ###
            loss_train = []
            loss_val = []
            
            ### loop thru batches ###
            for X, y in zip(X_batch, y_batch):

                ### Forward pass: get y_hat--predicted labels ###
                y_hat, cache = self.forward(X)

                ### calculate loss of predicted labels and save output###
                if self._loss_func == 'mse':
                    train_loss = self._mean_squared_error(y, y_hat)
                elif self._loss_func == 'bce':
                    train_loss = self._binary_cross_entropy(y, y_hat)
        
                loss_train.append(train_loss)

                ### backpropogation ###
                grad_dict = self.backprop(y, y_hat, cache)
                self._update_params(grad_dict)

                ### predict labels of validation set ###
                y_hat_val = self.predict(X_val)

                ### calculate loss on those predictions and save output ###
                if self._loss_func == 'mse':
                    val_loss = self._mean_squared_error(y_val, y_hat_val)
                elif self._loss_func == 'bce':
                    val_loss = self._binary_cross_entropy(y_val, y_hat_val)

                loss_val.append(val_loss)

            ### average loss across batches ###
            per_epoch_loss_train.append(np.mean(loss_train))
            per_epoch_loss_val.append(np.mean(loss_val))

        return per_epoch_loss_train, per_epoch_loss_val


    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        This function returns the prediction of the neural network model.

        Args:
            X: ArrayLike
                Input data for prediction.

        Returns:
            y_hat: ArrayLike
                Prediction from the model.
        """
        ##prediction is just one forward pass with new data##
        y_hat, cache = self.forward(X)
        
        return y_hat

    def _sigmoid(self, Z: ArrayLike) -> ArrayLike:
        """
        Sigmoid activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        ## calculate sigmoid, sigmoid = (1+e^(-x))^(-1) ##
        nl_transform = 1 / (1 + np.exp(-Z))
        return nl_transform 

    def _relu(self, Z: ArrayLike) -> ArrayLike:
        """
        ReLU activation function.

        Args:
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            nl_transform: ArrayLike
                Activation function output.
        """
        ### calculate relu = {x if x>0, 0 o/w} ###
        nl_transform=np.maximum(0,Z)
        return nl_transform

    def _sigmoid_backprop(self, dA: ArrayLike, Z: ArrayLike):
        """
        Sigmoid derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        dZ=self.sigmoid(Z)*(1-self._sigmoid(Z))
        return dZ

    def _relu_backprop(self, dA: ArrayLike, Z: ArrayLike) -> ArrayLike:
        """
        ReLU derivative for backprop.

        Args:
            dA: ArrayLike
                Partial derivative of previous layer activation matrix.
            Z: ArrayLike
                Output of layer linear transform.

        Returns:
            dZ: ArrayLike
                Partial derivative of current layer Z matrix.
        """
        if Z>0:
            dZ=1
        else:
            dZ=0
        return dZ
        

    def _binary_cross_entropy(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Binary cross entropy loss function.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            loss: float
                Average loss over mini-batch.
        """
        ### bound predictions b/c gonna log them ###
        y_hat=np.clip(y_pred, 0.00001, 0.99999)
        n = len(y_hat)
        
        ### BCE = -1/N sum_1^N(y_i*P(y_i)+(1-y_i)*P(1-y_i))) ###
        not_y = np.ones(n)-y
        prob_not_y = np.ones(n)-y_hat 
        loss = -np.mean((y*np.log(y_hat))+((not_y)*np.log(prob_not_y)))
        return loss

    def _binary_cross_entropy_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Binary cross entropy loss function derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        dA=-((y/y_hat) - ((1-y)/(1-y_hat))) / len(y_hat)
        return dA

    def _mean_squared_error(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Mean squared error loss.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.

        Returns:
            loss: float
                Average loss of mini-batch.
        """
        ### MSE= mean((y_hat-y)^2) ###
        loss=np.mean((y_hat-y)**2)
        return loss

    def _mean_squared_error_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        Mean square error loss derivative.

        Args:
            y_hat: ArrayLike
                Predicted output.
            y: ArrayLike
                Ground truth output.

        Returns:
            dA: ArrayLike
                partial derivative of loss with respect to A matrix.
        """
        dA=2*(y_hat-y)/len(y_hat)
        return dA

    def _loss_function(self, y: ArrayLike, y_hat: ArrayLike) -> float:
        """
        Loss function, computes loss given y_hat and y. This function is
        here for the case where someone would want to write more loss
        functions than just binary cross entropy.

        Args:
            y: ArrayLike
                Ground truth output.
            y_hat: ArrayLike
                Predicted output.
        Returns:
            loss: float
                Average loss of mini-batch.
        """
        pass

    def _loss_function_backprop(self, y: ArrayLike, y_hat: ArrayLike) -> ArrayLike:
        """
        This function performs the derivative of the loss function with respect
        to the loss itself.
        Args:
            y (array-like): Ground truth output.
            y_hat (array-like): Predicted output.
        Returns:
            dA (array-like): partial derivative of loss with respect
                to A matrix.
        """
        pass
