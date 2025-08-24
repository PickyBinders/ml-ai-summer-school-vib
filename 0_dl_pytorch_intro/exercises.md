# PyTorch Basics - Practice Exercises

These exercises are designed to help you get familiar with PyTorch syntax and basic operations. They follow the structure of the `pytorch_intro.ipynb` notebook.

## **Exercise 1: Tensor Creation and Operations** ⭐
**Time**: 10 minutes

Create the following tensors and perform the operations:

```python
# 1. Create these tensors:
# - A tensor with values [1, 2, 3, 4, 5]
# - A 2x3 tensor filled with zeros
# - A 3x3 tensor filled with ones
# - A 2x2 tensor with random values between 0 and 1

# 2. Perform these operations:
# - Add the first two tensors (if possible)
# - Multiply the third tensor by 5
# - Find the maximum value in the random tensor
# - Calculate the mean of the random tensor

# 3. Reshape the first tensor to 5x1 and then to 1x5
```

**Expected Output**: Show the results of each operation.

---

## **Exercise 2: Basic Tensor Math** ⭐
**Time**: 15 minutes

```python
# Create these tensors:
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

# Perform these operations and print results:
# 1. a + b
# 2. a * b (element-wise multiplication)
# 3. torch.matmul(a, b) (matrix multiplication)
# 4. a ** 2 (element-wise power)
# 5. torch.sum(a) (sum of all elements)
# 6. torch.mean(b) (mean of all elements)

# Bonus: Try a @ b (another way to do matrix multiplication)
```

---

## **Exercise 3: Autograd Practice** ⭐⭐
**Time**: 15 minutes

```python
# 1. Create a tensor x with value 3.0 and requires_grad=True
# 2. Create a tensor y with value 4.0 and requires_grad=True
# 3. Compute z = x² + y²
# 4. Call z.backward()
# 5. Print x.grad and y.grad
# 6. Verify manually: ∂z/∂x = 2x, ∂z/∂y = 2y

# Try with different values for x and y
```

---

## **Exercise 4: Simple Function with Autograd** ⭐⭐
**Time**: 20 minutes

```python
# Create a function f(x) = x³ + 2x + 1
# 1. Create a tensor x with requires_grad=True
# 2. Compute f(x)
# 3. Compute the derivative using backward()
# 4. Verify: f'(x) = 3x² + 2
# 5. Test with x = 2.0 and x = -1.0

# Bonus: Try with a list of x values and compute derivatives for each
```

---

## **Exercise 5: Building a Simple Neural Network** ⭐⭐
**Time**: 20 minutes

```python
# 1. Create a simple neural network using nn.Sequential:
#    - Input size: 5
#    - Hidden layer: 10 neurons with ReLU
#    - Output size: 1

# 2. Create a custom nn.Module class with:
#    - Input size: 5
#    - Hidden layer: 8 neurons with ReLU
#    - Output size: 1

# 3. Test both models with a random input tensor of shape (3, 5)
# 4. Print the output shapes
# 5. Count the number of parameters in each model
```

---

## **Exercise 6: Training a Simple Model** ⭐⭐⭐
**Time**: 25 minutes

```python
# Create a simple regression problem:
# 1. Generate synthetic data: y = 2x + 1 + noise
#    - Create 100 x values between 0 and 10
#    - Create corresponding y values
#    - Add some random noise

# 2. Build a simple model:
#    - One linear layer: input_size=1, output_size=1
#    - Use MSE loss
#    - Use SGD optimizer with learning_rate=0.01

# 3. Train for 100 epochs:
#    - Forward pass
#    - Compute loss
#    - Backward pass
#    - Update parameters
#    - Print loss every 20 epochs

# 4. Test the model on new data
# 5. Plot the results (x vs y and predicted y)
```

---

## **Exercise 7: DataLoader Practice** ⭐⭐
**Time**: 15 minutes

```python
# 1. Create a simple dataset:
#    - Generate 50 random feature vectors of size 3
#    - Generate corresponding labels (0 or 1)

# 2. Create a custom Dataset class:
#    - __init__ method that takes features and labels
#    - __len__ method
#    - __getitem__ method

# 3. Create a DataLoader:
#    - batch_size = 8
#    - shuffle = True

# 4. Iterate through the DataLoader and print:
#    - Batch number
#    - Batch data shape
#    - Batch labels shape
```

---

## **Exercise 8: GPU vs CPU (if available)** ⭐
**Time**: 10 minutes

```python
# 1. Check if CUDA is available
# 2. Create a large tensor (e.g., 1000x1000) on CPU
# 3. If CUDA is available, move it to GPU
# 4. Perform matrix multiplication on both devices
# 5. Compare execution times
# 6. Print the device of your tensors

# Note: This exercise only works if you have a GPU
```

---

## **Exercise 9: Common Mistakes Practice** ⭐⭐
**Time**: 15 minutes

```python
# Try these code snippets and fix the errors:

# 1. What's wrong with this?
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
z = x + y
print(z)

# 2. What's wrong with this?
model = nn.Linear(10, 1)
x = torch.randn(5, 10)
output = model(x)
print(output.shape)

# 3. What's wrong with this?
x = torch.tensor(2.0)
y = x**2
y.backward()
print(x.grad)

# 4. What's wrong with this?
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(3, 3)
y = x.to(device)
print(y.device)
```

---

## **Exercise 10: Mini Project - XOR Problem** ⭐⭐⭐
**Time**: 30 minutes

```python
# Solve the XOR problem:
# Input: (0,0) -> Output: 0
# Input: (0,1) -> Output: 1
# Input: (1,0) -> Output: 1
# Input: (1,1) -> Output: 0

# 1. Create the XOR dataset
# 2. Build a neural network with at least 2 hidden layers
# 3. Train the network
# 4. Test on all four input combinations
# 5. Print the predictions and accuracy
```

---

## **Exercise 11: Tensor Manipulation** ⭐
**Time**: 10 minutes

```python
# Create a tensor: torch.arange(12)
# 1. Reshape it to 3x4
# 2. Reshape it to 4x3
# 3. Reshape it to 2x6
# 4. Get the first row
# 5. Get the last column
# 6. Get the element at position (1, 2)
# 7. Get a 2x2 submatrix starting at position (0, 1)
```

---

## **Exercise 12: Loss Functions** ⭐⭐
**Time**: 15 minutes

```python
# 1. Create two tensors: predictions and targets
#    - predictions: [0.1, 0.9, 0.3, 0.8]
#    - targets: [0, 1, 0, 1]

# 2. Try different loss functions:
#    - MSE Loss
#    - Binary Cross Entropy Loss
#    - Cross Entropy Loss (with different shapes)

# 3. Print the loss values and explain the differences
```

---

## **Solutions and Tips**

### **General Tips:**
1. **Always check tensor shapes** with `.shape`
2. **Use `print()` to debug** your code
3. **Start simple** and build up complexity
4. **Test small examples** before scaling up

### **Common Patterns:**
```python
# Creating tensors
x = torch.tensor([1, 2, 3])
y = torch.zeros(2, 3)
z = torch.randn(3, 4)

# Basic operations
result = x + y
product = torch.matmul(x, z)

# Autograd
x = torch.tensor(2.0, requires_grad=True)
y = x**2
y.backward()
print(x.grad)

# Model creation
model = nn.Sequential(
    nn.Linear(10, 5),
    nn.ReLU(),
    nn.Linear(5, 1)
)
```

### **Debugging Tips:**
```python
# Check shapes
print(f"Tensor shape: {tensor.shape}")
print(f"Tensor dtype: {tensor.dtype}")
print(f"Tensor device: {tensor.device}")

# Check if gradients are enabled
print(f"Requires grad: {tensor.requires_grad}")

# Check model parameters
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
```

### **Expected Learning Outcomes:**
- ✅ Understand tensor creation and basic operations
- ✅ Get comfortable with autograd and gradients
- ✅ Learn to build simple neural networks
- ✅ Practice with DataLoaders and training loops
- ✅ Understand common PyTorch patterns and syntax
- ✅ Debug common issues and errors

**Remember:** The goal is to get familiar with PyTorch syntax, not to build perfect models! Experiment, make mistakes, and learn from them. 🚀 