### Your Neural Network is Secretly a Kernel Machine

Arunav Gupta, Rohit Mishra, Mehdi Bouassami William Luu

## Introduction

While neural networks have been successful in solving many technological and scientific problems, our understanding of how they work hasn't kept up with their practical advancements. By identifying the underlying mechanics that make these systems work, we can develop new and efficient strategies to improve neural network performance. Our study focuses on the essential mechanism behind feature learning in fully connected neural networks to construct recursive feature machines (RFMs), which are kernel machines that learn features. Our findings show that RFMs outperform a range of models, including deep neural networks.

In recent years, there have been highly accurate machine learning models with billions of parameters that can solve tasks previously thought difficult. However, researchers are unsure why these models are better. One way to understand this is by studying kernel methods. Kernel functions are a type of algorithm that can solve non-linear problems with linear classifiers. Different types of kernels can transform data into higher dimensions to solve linear separability problems.

We also want to compare neural networks to recursive feature machines, which are kernel machines that learn features from deep fully connected neural networks. Recursive feature machines can capture features learned by deep networks and perform well on some tabular data. We will compare the three methods on various datasets and understand their performance differences. We will also analyze the theory behind recursive kernel machines by looking at the eigenvalues of the kernel matrix.

In summary, we're investigating why large parameter models perform better, comparing kernel methods to neural networks and recursive feature machines on text datasets, and analyzing the theory behind recursive kernel machines.

## What Do Kernels Look Like?

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

## What Are Recursive Feature Machines?

RFMs have the following form:

$$
K(x, z) = \exp\left(\frac{-|x-z|M}{σ}\right)
$$       
Where $$|x-z|M := \sqrt{(x-z)^{T}(x-z)}$$

Thus, we can calculate the gradient as follow: 

 $$\nabla K_{M}(x, z) = \frac{Mx - Mz}{σ|x - z|M}K(x, z)$$

## How To Train an RFM?

First, let d be the number of features and let $$M := I_{dxd}$$ (1)

Then use M to train a kernel $$K_{M}$$ (2)

Update M as follow: $$ M = \frac{1}{n}\sum_{x \in X}\nabla f(x) \nabla f(x)^{T}$$ where $$ \nabla f(x) = \alpha \nabla K_{M}(X, x) $$ (3)

Finally, cross-validate and repeat 2-4 until convergence (4)

## What Are The Benefits of RFMs?

RFMs have a remarkable ability to outperform several neural networks in text classification, as the results section will demonstrate later on. Additionally, RFMs use sparse matrices, making them more data-efficient than other methods. Interestingly, M also converges to the same weights as a learned DNN does on vision tasks.

## Our Dataset

![1984, George Orwell](/assets/images/1984_cover.jpg)

Our approach involved using RFMs to predict the next word in a text dataset. To accomplish this, we utilized a PDF version of George Orwell's 1984 and extracted the raw words from it. We constructed a vocabulary of 50 alphanumeric characters and tokenized the text accordingly. Next, we encoded the characters in a one-hot format and obtained a matrix with dimensions N (number of samples) by 64 (token size) by 50 (vocabulary size). We then compared the performance of our RFM model to that of bigram/trigram models, as well as the Laplacian kernel.

## Results 

### Scaling Test

Before testing the model on text prediction, let's first look at how the performance of an RFM scales with the model and feature size. 
In order to do so, we generated 1000 random datapoints $$x_{i} \in R^{d} \rightarrow X $$ and used the following target function:
$$f(x) = 5x_1^3 + 10x_2^2 + 2x_3$$
We then train an RFM and Laplacian Kernel ("Baseline") and get the test MSE. The results are shown below

![Scaling Test](/assets/images/scaling_plot_test_with_baseline)

















