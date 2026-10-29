import numpy as np
# loads numpy as np. All array work uses it.


def unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)  ## remove leading axis
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] > 1:
            grad = grad.sum(axis=axis, keepdims=True)  ## sum along the axis
    return grad





class Tensor:  ## begins the definition of the Tensor class, which is a fundamental data structure for automatic differentiation. i.e the blueprint
    def __init__(self, data, parents=(), op=""):    ## set-up method. If parents=() is not passed. It's defaaults to an empty tuple. This means that the tensor has no parents in the computational graph. The op parameter is a string that describes the operation that produced this tensor. 
        self.data =np.asarray(data, dtype=np.float64) ## passes all data to an array of 64-bit decimals
        self.grad = np.zeros_like(self.data)  ## makes an array of zeros of the same shape as the data so += works from the onset.
        self._parents = parents ## stores the tensors created this tensor in a tuple. This is used to build the computational graph for backpropagation.
        self._backward = lambda: None ## it's a one line nameless function that takes no input and retunrs nothing. This is a placeholder for the backward method that will be defined later. It will be used to compute the gradients of the tensor with respect to its parents during backpropagation.
        self.op = op ## stores the lablel of the operation that produced this tensor. This is used for debugging and visualization purposes.            

    
    
    def __add__(self, other): ## runs for a + b where a is self and b is other. It defines the addition operation for tensors.
        other = other if isinstance(other, Tensor) else Tensor(other) ## checks if other is already a Tensor. If not, it converts it into a Tensor. This allows for operations between Tensors and scalars or other data types.
        out = Tensor(self.data + other.data, parents=(self, other), op="+") ## does the forward pass of the addition operation. It creates a new Tensor out that holds the result of adding self.data and other.data. It also sets the parents of out to be self and other, and labels the operation as "+".

        def _backward(): ## it captures the results of self, other and out from the outer scope. It defines the backward pass for the addition operation. This function will be called during backpropagation to compute the gradients of self and other with respect to out.
            self.grad += unbroadcast(out.grad, self.data.shape) ## updates the gradient of self by adding the gradient of out. This is based on the chain rule of calculus, which states that the gradient of a function with respect to its inputs is the product of the gradients along the computational graph.
            other.grad += unbroadcast(out.grad, other.data.shape) ## updates the gradient of other by adding the gradient of out. This is also based on the chain rule of calculus.


        out._backward = _backward ## attach the _backward function to the out tensor. This allows the backward pass to be executed when backpropagation is performed on out.
        return out  ## gives back the result of the addition operation as a new Tensor. This allows for chaining operations together, as the result can be used in further computations.    


    
    def __mul__(self, other): ## runs for a * b where a is self and b is other. It defines the multiplication operation for tensors.
        other = other if isinstance(other, Tensor) else Tensor(other) ## checks if other is already a Tensor. If not, it converts it into a Tensor. This allows for operations between Tensors and scalars or other data types.
        out = Tensor(self.data * other.data, parents=(self, other), op="*") ## does the forward pass of the multiplication operation. It creates a new Tensor out that holds the result of multiplying self.data and other.data. It also sets the parents of out to be self and other, and labels the operation as "*".

        def _backward(): ## it captures the results of self, other and out from the outer scope. It defines the backward pass for the multiplication operation. This function will be called during backpropagation to compute the gradients of self and other with respect to out.
            self.grad += unbroadcast((other.data * out.grad), self.data.shape) ## updates the gradient of self by adding the product of other.data and out.grad. This is based on the chain rule of calculus, which states that the gradient of a function with respect to its inputs is the product of the gradients along the computational graph.
            other.grad += unbroadcast((self.data * out.grad), other.data.shape) ## updates the gradient of other by adding the product of self.data and out.grad. This is also based on the chain rule of calculus.

        out._backward = _backward ## attach the _backward function to the out tensor. This allows the backward pass to be executed when backpropagation is performed on out.
        return out  ## gives back the result of the multiplication operation as a new Tensor. This allows for chaining operations together, as the result can be used in further computations.


    def __neg__(self): ## defines the negation operation for tensors. It allows for expressions like -a, where a is a Tensor.
        out = Tensor(-self.data, parents=(self,), op="neg") ## does the forward pass of the negation operation. It creates a new Tensor out that holds the result of negating self.data. It also sets the parent of out to be self, and labels the operation as "neg".

        def _backward(): ## it captures the results of self and out from the outer scope. It defines the backward pass for the negation operation. This function will be called during backpropagation to compute the gradient of self with respect to out.
            self.grad -= out.grad ## updates the gradient of self by subtracting out.grad. This is based on the chain rule of calculus, which states that the gradient of a function with respect to its inputs is the product of the gradients along the computational graph.

        out._backward = _backward ## attach the _backward function to the out tensor. This allows the backward pass to be executed when backpropagation is performed on out.
        return out  ## gives back the result of the negation operation as a new Tensor. This allows for chaining operations together, as the result can be used in further computations.    


    def __sub__(self, other): ## defines the subtraction operation for tensors. It allows for expressions like a - b, where a and b are Tensors.                
        other = other if isinstance(other, Tensor) else Tensor(other) ## checks if other is already a Tensor. If not, it converts it into a Tensor. This allows for operations between Tensors and scalars or other data types.
        out = Tensor(self.data - other.data, parents=(self, other), op="-") ## does the forward pass of the subtraction operation. It creates a new Tensor out that holds the result of subtracting other.data from self.data. It also sets the parents of out to be self and other, and labels the operation as "-".

        def _backward(): ## it captures the results of self, other and out from the outer scope. It defines the backward pass for the subtraction operation. This function will be called during backpropagation to compute the gradients of self and other with respect to out.
            self.grad += unbroadcast(out.grad, self.data.shape) ## updates the gradient of self by adding out.grad. This is based on the chain rule of calculus, which states that the gradient of a function with respect to its inputs is the product of the gradients along the computational graph.
            other.grad -= unbroadcast(out.grad, other.data.shape) ## updates the gradient of other by subtracting out.grad. This is also based on the chain rule of calculus.

        out._backward = _backward ## attach the _backward function to the out tensor. This allows the backward pass to be executed when backpropagation is performed on out.
        return out  ## gives back the result of the subtraction operation as a new Tensor. This allows for chaining operations together, as the result can be used in further computations.


    def __truediv__(self, other): ## defines the division operation for tensors. It allows for expressions like a / b, where a and b are Tensors.                       
        other = other if isinstance(other, Tensor) else Tensor(other) ## checks if other is already a Tensor. If not, it converts it into a Tensor. This allows for operations between Tensors and scalars or other data types.
        out = Tensor(self.data / other.data, parents=(self, other), op="/") ## does the forward pass of the division operation. It creates a new Tensor out that holds the result of dividing self.data by other.data. It also sets the parents of out to be self and other, and labels the operation as "/".

        def _backward(): ## it captures the results of self, other and out from the outer scope. It defines the backward pass for the division operation. This function will be called during backpropagation to compute the gradients of self and other with respect to out.
            self.grad += (1 / other.data) * unbroadcast(out.grad, self.data.shape) ## updates the gradient of self by adding the product of (1 / other.data) and out.grad. This is based on the chain rule of calculus, which states that the gradient of a function with respect to its inputs is the product of the gradients along the computational graph.
            other.grad -= (self.data / (other.data ** 2)) * unbroadcast(out.grad, other.data.shape) ## updates the gradient of other by subtracting the product of (self.data / (other.data ** 2)) and out.grad. This is also based on the chain rule of calculus.

        out._backward = _backward ## attach the _backward function to the out tensor. This allows the backward pass to be executed when backpropagation is performed on out.
        return out  ## gives back the result of the division operation as a new Tensor. This allows for chaining operations together, as the result can be used in further computations.    




    def __pow__(self, power): ## defines the power operation for tensors. It allows for expressions like a ** b, where a is a Tensor and b is a scalar or another Tensor.
        assert isinstance(power, (int, float)), "Power must be an int or float" ## checks if the power is either an integer or a float. If not, it raises an assertion error with the message "Power must be an int or float". This ensures that the power operation is only performed with valid types.

        out = Tensor(self.data ** power, parents=(self,), op=f"**{power}") ## does the forward pass of the power operation. It creates a new Tensor out that holds the result of raising self.data to the specified power. It also sets the parent of out to be self, and labels the operation as "**{power}".

        def _backward(): ## it captures the results of self, power and out from the outer scope. It defines the backward pass for the power operation. This function will be called during backpropagation to compute the gradient of self with respect to out.
            self.grad += (power * (self.data ** (power - 1))) * out.grad ## updates the gradient of self by adding the product of (power * (self.data ** (power - 1))) and out.grad. This is based on the chain rule of calculus, which states that the gradient of a function with respect to its inputs is the product of the gradients along the computational graph.

        out._backward = _backward ## attach the _backward function to the out tensor. This allows the backward pass to be executed when backpropagation is performed on out.
        return out  ## gives back the result of the power operation as a new Tensor. This allows for chaining operations together, as the result can be used in further computations.   



    


    def backward(self): ## defines the backward method for the Tensor class.
        topo = [] ## initializes an empty list topo that will be used to store the tensors in topological order. This order ensures that each tensor is processed after its parents have been processed, allowing for correct gradient computation.
        visited = set() ## initializes an empty set visited that will be used to keep track of the tensors that have already been visited during the depth-first search (DFS) traversal of the computational graph.

        def build_topo(tensor): ## defines a helper function build_topo that performs a DFS traversal of the computational graph starting from the given tensor. It captures the results of topo and visited from the outer scope.
            if tensor not in visited: ## checks if the current tensor has not been visited yet.
                visited.add(tensor) ## marks the current tensor as visited by adding it to the visited set.
                for parent in tensor._parents: ## iterates over each parent of the current tensor.
                    build_topo(parent) ## recursively calls build_topo on each parent, ensuring that all ancestors are processed before the current tensor.
                topo.append(tensor) ## adds the current tensor to the topo list after all its parents have been processed.

        build_topo(self) ## initiates the DFS traversal by calling build_topo on self, which is the tensor on which backward() was called. This builds the topological order of tensors in topo.

        self.grad = np.ones_like(self.data) ## initializes the gradient of self to be an array of ones with the same shape as self.data. This is because we are computing gradients with respect to self, and we want to start with a gradient of 1 for backpropagation.

        for tensor in reversed(topo): ## iterates over each tensor in topo in reverse order. This ensures that we process tensors from leaves to roots, allowing gradients to flow correctly through the computational graph.
            tensor._backward() ## calls the _backward method of each tensor, which computes and accumulates gradients for its parents based on its own gradient. This effectively propagates gradients backward through the computational graph.


        
    def __radd__(self, other):
        return self + other  ## allows for right-side addition, enabling expressions like scalar + Tensor to work correctly by delegating to the __add__ method.

    def __rmul__(self, other):
        return self * other  ## allows for right-side multiplication, enabling expressions like scalar * Tensor to work correctly by delegating to the __mul__ method.  

    def __rsub__(self, other):
        return -self + other  ## allows for right-side subtraction, enabling expressions like scalar - Tensor to work correctly by negating self and adding it to other. This effectively computes the difference between other and self.


    def __rtruediv__(self, other):
        return Tensor(other) / self  ## allows for right-side division, enabling expressions like scalar / Tensor to work correctly by creating a new Tensor from other and dividing it by self. This effectively computes the reciprocal of self multiplied by other.



    def __matmul__(self, other): ## defines the matrix multiplication operation for tensors. It allows for expressions like a @ b, where a and b are Tensors.
        other = other if isinstance(other, Tensor) else Tensor(other) ## checks if other is already a Tensor. If not, it converts it into a Tensor. This allows for operations between Tensors and scalars or other data types.
        out = Tensor(self.data @ other.data, parents=(self, other), op="@") ## does the forward pass of the matrix multiplication operation. It creates a new Tensor out that holds the result of multiplying self.data and other.data using the @ operator. It also sets the parents of out to be self and other, and labels the operation as "@".

        def _backward(): ## it captures the results of self, other and out from the outer scope. It defines the backward pass for the matrix multiplication operation. This function will be called during backpropagation to compute the gradients of self and other with respect to out.
            self.grad += out.grad @ other.data.T ## updates the gradient of self by adding the product of out.grad and the transpose of other.data. This is based on the chain rule of calculus, which states that the gradient of a function with respect to its inputs is the product of the gradients along the computational graph.
            other.grad += self.data.T @ out.grad ## updates the gradient of other by adding the product of the transpose of self.data and out.grad. This is also based on the chain rule of calculus.

        out._backward = _backward ## attach the _backward function to the out tensor. This allows the backward pass to be executed when backpropagation is performed on out.
        return out  ## gives back the result of the matrix multiplication operation as a new Tensor. This allows for chaining operations together, as the result can be used in further computations.

    

    def transpose(self): ## defines the transpose operation for tensors. It allows for expressions like a.transpose(), where a is a Tensor.
        out = Tensor(self.data.T, parents=(self,), op="transpose") ## does the forward pass of the transpose operation. It creates a new Tensor out that holds the result of transposing self.data. It also sets the parent of out to be self, and labels the operation as "transpose".

        def _backward(): ## it captures the results of self and out from the outer scope. It defines the backward pass for the transpose operation. This function will be called during backpropagation to compute the gradient of self with respect to out.
            self.grad += out.grad.T ## updates the gradient of self by adding the transpose of out.grad. This is based on the chain rule of calculus, which states that the gradient of a function with respect to its inputs is the product of the gradients along the computational graph.

        out._backward = _backward ## attach the _backward function to the out tensor. This allows the backward pass to be executed when backpropagation is performed on out.
        return out  ## gives back the result of the transpose operation as a new Tensor. This allows for chaining operations together, as the result can be used in further computations.   

    @property
    def T(self): ## defines a property T that allows for accessing the transpose of a tensor    
        return self.transpose() ## returns the result of calling the transpose method on self. This allows for expressions like a.T to get the transpose of a tensor a. 



    def reshape(self, shape): ## defines the reshape operation for tensors. It allows for expressions like a.reshape(new_shape), where a is a Tensor and new_shape is a tuple specifying the desired shape.
        out = Tensor(self.data.reshape(shape), parents=(self,), op="reshape") ## does the forward pass of the reshape operation. It creates a new Tensor out that holds the result of reshaping self.data to new_shape. It also sets the parent of out to be self, and labels the operation as "reshape".

        def _backward(): ## it captures the results of self and out from the outer scope. It defines the backward pass for the reshape operation. This function will be called during backpropagation to compute the gradient of self with respect to out.
            self.grad += out.grad.reshape(self.data.shape) ## updates the gradient of self by adding out.grad reshaped back to the original shape of self.data. This is based on the chain rule of calculus, which states that the gradient of a function with respect to its inputs is the product of the gradients along the computational graph.

        out._backward = _backward ## attach the _backward function to the out tensor. This allows the backward pass to be executed when backpropagation is performed on out.
        return out  ## gives back the result of the reshape operation as a new Tensor. This allows for chaining operations together, as the result can be used in further computations.

    def sum(self, axis=None, keepdims=False): ## defines the sum operation for tensors. It allows for expressions like a.sum(axis=0), where a is a Tensor and axis specifies the dimension along which to sum.
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), parents=(self,), op="sum") ## does the forward pass of the sum operation. It creates a new Tensor out that holds the result of summing self.data along the specified axis. It also sets the parent of out to be self, and labels the operation as "sum".

        def _backward(): ## it captures the results of self and out from the outer scope. It defines the backward pass for the sum operation. This function will be called during backpropagation to compute the gradient of self with respect to out.
            g = out.grad ## stores the gradient of out in a variable grad.
            if axis is not None and not keepdims: ## checks if keepdims is False.
                g = np.expand_dims(g, axis=axis) ## if keepdims is False, it expands the dimensions of grad along the specified axis to match the shape of self.data.
            self.grad += np.broadcast_to(g, self.data.shape) ## updates the gradient of self by adding grad broadcasted to match the shape of self.data. This is based on the chain rule of calculus, which states that the gradient of a function with respect to its inputs is the product of the gradients along the computational graph.

        out._backward = _backward ## attach the _backward function to the out tensor. This allows the backward pass to be executed when backpropagation is performed on out.
        return out  ## gives back the result of the sum operation as a new Tensor. This allows for chaining operations together, as the result can be used in further computations.


    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else self.data.shape[axis] ## calculates the number of elements along the specified axis. If axis is None, it uses the total number of elements in self.data.
        return self.sum(axis=axis, keepdims=keepdims) * (1.0    / n) ## returns the mean of self.data by dividing the sum by the number of elements. It calls the sum method and scales the result by 1/n. This allows for expressions like a.mean(axis=0) to compute the mean along a specified axis.  


    def exp(self): ## defines the exponential operation for tensors. It allows for expressions like a.exp(), where a is a Tensor.
        out = Tensor(np.exp(self.data), parents=(self,), op="exp") ## does the forward pass of the exponential operation. It creates a new Tensor out that holds the result of applying the exponential function to self.data. It also sets the parent of out to be self, and labels the operation as "exp".

        def _backward(): ## it captures the results of self and out from the outer scope. It defines the backward pass for the exponential operation. This function will be called during backpropagation to compute the gradient of self with respect to out.
            self.grad += out.data * out.grad ## updates the gradient of self by adding the product of out.data and out.grad. This is based on the chain rule of calculus, which states that the gradient of a function with respect to its inputs is the product of the gradients along the computational graph.

        out._backward = _backward ## attach the _backward function to the out tensor. This allows the backward pass to be executed when backpropagation is performed on out.
        return out  ## gives back the result of the exponential operation as a new Tensor. This allows for chaining operations together, as the result can be used in further computations. 


    def log(self): ## defines the natural logarithm operation for tensors. It allows for expressions like a.log(), where a is a Tensor.
        out = Tensor(np.log(self.data), parents=(self,), op="log") ## does the forward pass of the logarithm operation. It creates a new Tensor out that holds the result of applying the natural logarithm function to self.data. It also sets the parent of out to be self, and labels the operation as "log".

        def _backward(): ## it captures the results of self and out from the outer scope. It defines the backward pass for the logarithm operation. This function will be called during backpropagation to compute the gradient of self with respect to out.
            self.grad += (1 / self.data) * out.grad ## updates the gradient of self by adding the product of (1 / self.data) and out.grad. This is based on the chain rule of calculus, which states that the gradient of a function with respect to its inputs is the product of the gradients along the computational graph.

        out._backward = _backward ## attach the _backward function to the out tensor. This allows the backward pass to be executed when backpropagation is performed on out.
        return out  ## gives back the result of the logarithm operation as a new Tensor. This allows for chaining operations together, as the result can be used in further computations.   


    def relu(self): ## defines the ReLU (Rectified Linear Unit) activation function for tensors. It allows for expressions like a.relu(), where a is a Tensor.
        out = Tensor(np.maximum(0, self.data), parents=(self,), op="ReLU") ## does the forward pass of the ReLU operation. It creates a new Tensor out that holds the result of applying the ReLU function to self.data. It also sets the parent of out to be self, and labels the operation as "ReLU".

        def _backward(): ## it captures the results of self and out from the outer scope. It defines the backward pass for the ReLU operation. This function will be called during backpropagation to compute the gradient of self with respect to out.
            self.grad += (self.data > 0) * out.grad ## updates the gradient of self by adding the product of (self.data > 0) and out.grad. This effectively passes gradients only for positive input values, as ReLU is zero for negative inputs.

        out._backward = _backward ## attach the _backward function to the out tensor. This allows the backward pass to be executed when backpropagation is performed on out.
        return out  ## gives back the result of the ReLU operation as a new Tensor. This allows for chaining operations together, as the result can be used in further computations.    


    def tanh(self): ## defines the hyperbolic tangent activation function for tensors. It allows for expressions like a.tanh(), where a is a Tensor.
        out = Tensor(np.tanh(self.data), parents=(self,), op="tanh") ## does the forward pass of the tanh operation. It creates a new Tensor out that holds the result of applying the tanh function to self.data. It also sets the parent of out to be self, and labels the operation as "tanh".

        def _backward(): ## it captures the results of self and out from the outer scope. It defines the backward pass for the tanh operation. This function will be called during backpropagation to compute the gradient of self with respect to out.
            self.grad += (1 - out.data ** 2) * out.grad ## updates the gradient of self by adding the product of (1 - out.data ** 2) and out.grad. This is based on the derivative of the tanh function, which is 1 - tanh^2(x).

        out._backward = _backward ## attach the _backward function to the out tensor. This allows the backward pass to be executed when backpropagation is performed on out.
        return out  ## gives back the result of the tanh operation as a new Tensor. This allows for chaining operations together, as the result can be used in further computations.    


    def sigmoid(self): ## defines the sigmoid activation function for tensors. It allows for expressions like a.sigmoid(), where a is a Tensor. 
        out = Tensor(1 / (1 + np.exp(-self.data)), parents=(self,), op="sigmoid") ## does the forward pass of the sigmoid operation. It creates a new Tensor out that holds the result of applying the sigmoid function to self.data. It also sets the parent of out to be self, and labels the operation as "sigmoid".

        def _backward(): ## it captures the results of self and out from the outer scope. It defines the backward pass for the sigmoid operation. This function will be called during backpropagation to compute the gradient of self with respect to out.
            self.grad += (out.data * (1 - out.data)) * out.grad ## updates the gradient of self by adding the product of (out.data * (1 - out.data)) and out.grad. This is based on the derivative of the sigmoid function, which is sigmoid(x) * (1 - sigmoid(x)).

        out._backward = _backward ## attach the _backward function to the out tensor. This allows the backward pass to be executed when backpropagation is performed on out.
        return out  ## gives back the result of the sigmoid operation as a new Tensor. This allows for chaining operations together, as the result can be used in further computations. 


    def log_softmax(self, axis=-1): ## defines
        m = self.data.max(axis=axis, keepdims=True) ## computes the maximum value along the specified axis, keeping the dimensions for broadcasting. This is used for numerical stability in the softmax computation.
        shifted = self.data - m ## shifts the input data by subtracting the maximum value. This helps prevent overflow in the exponential calculations.
        lse = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True)) ## computes the log-sum-exp (LSE) of the shifted data along the specified axis. This is a key step in calculating the log-softmax.  
        out = Tensor(shifted - lse, parents=(self,), op="log_softmax") ## creates a new Tensor out that holds the result of the log-softmax operation. It sets the parent of out to be self and labels the operation as "log_softmax".

        def _backward(): ## it captures the results of self and out from the outer scope. It defines the backward pass for the log-softmax operation. This function will be called during backpropagation to compute the gradient of self with respect to out.
            p = np.exp(out.data) ## computes the softmax probabilities by exponentiating the log-softmax output. This is used to calculate the gradient of the log-softmax function.
            self.grad += (out.grad - np.sum(out.grad, axis=axis, keepdims=True) * p) ## updates the gradient of self by adding the difference between out.grad and the sum of out.grad multiplied by the softmax probabilities. This is based on the derivative of the log-softmax function.

        out._backward = _backward ## attach the _backward function to the out tensor. This allows the backward pass to be executed when backpropagation is performed on out.
        return out  ## gives back the result of the log-softmax operation as a new Tensor. This allows for chaining operations together, as the result can be used in further computations.
    


def cross_entropy(logits, targets):
    N, C = logits.data.shape ## unpacks the shape of the logits tensor into N (number of samples) and C (number of classes).
    one_hot_targets = np.zeros_like(logits.data) ## creates a one-hot encoded version of the targets tensor with the same shape as logits.data, initialized to zeros.
    m = np.zeros((N, C)) ## creates a temporary matrix m of zeros with the same shape as logits.data. This will be used to construct the one-hot encoding of the targets.
    labels = np.array(targets, dtype=int) ## converts the targets into a numpy array of integers. This ensures that the target labels are in the correct format for indexing.
    m[np.arange(N), labels] = 1
    logp = logits.log_softmax(axis=1) ## computes the log-softmax of the logits tensor along the class dimension (axis=1). This gives the log probabilities for each class.
    return -(logp * Tensor(m)).sum(axis=1).mean() ## computes the cross-entropy loss by taking the negative sum of the product of the one-hot encoded targets and the log probabilities, averaging over all samples. This gives a scalar loss value that can be used for optimization.  


def grad_check(f, inputs, eps=1e-6):
    for t in inputs:
        t.grad = np.zeros_like(t.data)  # Reset gradients to zero before backward pass  
    out = f(*inputs)
    R = np.random.default_rng(0).standard_normal(out.data.shape)  # fixed random weight for reproducibility
    L = (out * Tensor(R)).sum()  # Use the sum of the output for gradient checking
    L.backward()  # Compute gradients using backpropagation
    analytic = [t.grad.copy() for t in inputs]  # Store the analytic gradients
    errors = []
    for t, g_an in zip(inputs, analytic):
        g_num = np.zeros_like(t.data)
        for idx in np.ndindex(t.data.shape):
            old = t.data[idx]
            t.data[idx] = old + eps
            plus = (f(*inputs).data * R).sum()
            t.data[idx] = old - eps
            minus = (f(*inputs).data * R).sum()
            t.data[idx] = old  # Restore original value
            g_num[idx] = (plus - minus) / (2 * eps)
        error = np.linalg.norm(g_an - g_num) / max(np.linalg.norm(g_an) + np.linalg.norm(g_num), 1e-12)
        errors.append(error)
    return errors       

