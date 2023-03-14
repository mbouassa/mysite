### Your Neural Network is Secretly a Kernel Machine

Arunav Gupta, Rohit Mishra, Mehdi Bouassami William Luu

## Introduction

While neural networks have been successful in solving many technological and scientific problems, our understanding of how they work hasn't kept up with their practical advancements. By identifying the underlying mechanics that make these systems work, we can develop new and efficient strategies to improve neural network performance. Our study focuses on the essential mechanism behind feature learning in fully connected neural networks to construct recursive feature machines (RFMs), which are kernel machines that learn features. Our findings show that RFMs outperform a range of models, including deep neural networks.

In recent years, there have been highly accurate machine learning models with billions of parameters that can solve tasks previously thought difficult. However, researchers are unsure why these models are better. One way to understand this is by studying kernel methods. Kernel functions are a type of algorithm that can solve non-linear problems with linear classifiers. Different types of kernels can transform data into higher dimensions to solve linear separability problems.

We also want to compare neural networks to recursive feature machines, which are kernel machines that learn features from deep fully connected neural networks. Recursive feature machines can capture features learned by deep networks and perform well on some tabular data. We will compare the three methods on various datasets and understand their performance differences. We will also analyze the theory behind recursive kernel machines by looking at the eigenvalues of the kernel matrix.

In summary, we're investigating why large parameter models perform better, comparing kernel methods to neural networks and recursive feature machines on text datasets, and analyzing the theory behind recursive kernel machines.

## Methods

# What Do Kernels Look Like?

Kernels take the following form: 

$$
K(x, z) = \exp\left(\frac{-|x-z|L}{σ}\right)
$$


Where L represents the distance norm and σ the kernel width.
L = 1 for the Laplacian Kernel and L = 2 for the Gaussian kernel.

In order to train a kernel, the following equation has to be solved for $$\hat{\alpha}$$:

$$\hat{\alpha} = yK(X, X)^{-1}$$

To make predictions, use the following equation:

$$\hat{y}(x) = \hat{\alpha}K(X, x)$$

# What Are Recursive Feature Machines?

RFMs have the following form:

$$
K(x, z) = \exp\left(\frac{-|x-z|M}{σ}\right)
$$       \quad Where $$|x-z|M := \sqrt{(x-z)^{T}(x-z)}$$

Thus, we can calculate the gradient as follow: 

 $$\nabla K_{M}(x, z) = \frac{Mx - Mz}{σ|x - z|M}K(x, z) $$







