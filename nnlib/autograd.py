import numpy as np
# loads numpy as np. All array work uses it.

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
            self.grad += out.grad ## updates the gradient of self by adding the gradient of out. This is based on the chain rule of calculus, which states that the gradient of a function with respect to its inputs is the product of the gradients along the computational graph.
            other.grad += out.grad ## updates the gradient of other by adding the gradient of out. This is also based on the chain rule of calculus.


        out._backward = _backward ## attach the _backward function to the out tensor. This allows the backward pass to be executed when backpropagation is performed on out.
        return out  ## gives back the result of the addition operation as a new Tensor. This allows for chaining operations together, as the result can be used in further computations.    


    def __mul__(self, other): ## runs for a * b where a is self and b is other. It defines the multiplication operation for tensors.
        other = other if isinstance(other, Tensor) else Tensor(other) ## checks if other is already a Tensor. If not, it converts it into a Tensor. This allows for operations between Tensors and scalars or other data types.
        out = Tensor(self.data * other.data, parents=(self, other), op="*") ## does the forward pass of the multiplication operation. It creates a new Tensor out that holds the result of multiplying self.data and other.data. It also sets the parents of out to be self and other, and labels the operation as "*".

        def _backward(): ## it captures the results of self, other and out from the outer scope. It defines the backward pass for the multiplication operation. This function will be called during backpropagation to compute the gradients of self and other with respect to out.
            self.grad += other.data * out.grad ## updates the gradient of self by adding the product of other.data and out.grad. This is based on the chain rule of calculus, which states that the gradient of a function with respect to its inputs is the product of the gradients along the computational graph.
            other.grad += self.data * out.grad ## updates the gradient of other by adding the product of self.data and out.grad. This is also based on the chain rule of calculus.

        out._backward = _backward ## attach the _backward function to the out tensor. This allows the backward pass to be executed when backpropagation is performed on out.
        return out  ## gives back the result of the multiplication operation as a new Tensor. This allows for chaining operations together, as the result can be used in further computations.



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