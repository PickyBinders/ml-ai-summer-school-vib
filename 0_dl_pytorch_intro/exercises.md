# PyTorch Course Exercises

This document contains exercises for both PyTorch notebooks to help students practice and reinforce their learning.

## Notebook 1: PyTorch Basics Exercises

### Exercise 1.1: Tensor Operations
**Difficulty**: Easy
**Time**: 15 minutes

Create the following tensors and perform the operations:
```python
# Create tensors
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([[7, 8, 9], [10, 11, 12]])

# Tasks:
# 1. Add a and b
# 2. Multiply a and b element-wise
# 3. Compute matrix multiplication of a and b.T
# 4. Find the maximum value in a
# 5. Compute the mean of b
# 6. Reshape a to a 1D tensor
```

**Expected Output**: Show the results of each operation.

### Exercise 1.2: Custom Autograd Function
**Difficulty**: Medium
**Time**: 20 minutes

Create a custom function that computes f(x) = x³ + 2x² + 3x + 1 and its derivative:
```python
# Tasks:
# 1. Create a tensor x with requires_grad=True
# 2. Implement the function f(x) = x³ + 2x² + 3x + 1
# 3. Compute the derivative using backward()
# 4. Verify the derivative manually (f'(x) = 3x² + 4x + 3)
# 5. Test with multiple values of x
```

### Exercise 1.3: Simple Neural Network
**Difficulty**: Medium
**Time**: 25 minutes

Build a neural network to solve the XOR problem:
```python
# XOR truth table:
# Input: (0,0) -> Output: 0
# Input: (0,1) -> Output: 1
# Input: (1,0) -> Output: 1
# Input: (1,1) -> Output: 0

# Tasks:
# 1. Create XOR dataset
# 2. Build a neural network with at least 2 hidden layers
# 3. Train the network to learn XOR
# 4. Test the network on all four input combinations
# 5. Visualize the training loss
```

### Exercise 1.4: DataLoader Practice
**Difficulty**: Easy
**Time**: 15 minutes

Create a custom dataset and DataLoader:
```python
# Tasks:
# 1. Create a custom dataset class for a simple regression problem
# 2. Generate synthetic data: y = 2x + 1 + noise
# 3. Create DataLoader with batch_size=4
# 4. Iterate through the DataLoader and print batch shapes
# 5. Verify that data is being batched correctly
```

### Exercise 1.5: GPU vs CPU Comparison
**Difficulty**: Easy
**Time**: 10 minutes

Compare computation time between CPU and GPU:
```python
# Tasks:
# 1. Create large tensors (e.g., 1000x1000)
# 2. Perform matrix multiplication on CPU
# 3. Perform matrix multiplication on GPU (if available)
# 4. Measure and compare execution times
# 5. Calculate speedup factor
```

## Notebook 2: Protein Dimer Classification Exercises

### Exercise 2.1: Data Exploration
**Difficulty**: Easy
**Time**: 20 minutes

Explore the dimers dataset in detail:
```python
# Tasks:
# 1. Load the dimers_features.csv dataset
# 2. Create a correlation matrix heatmap for numerical features
# 3. Plot distributions of 5 most important features
# 4. Analyze the relationship between interface_area and physiological
# 5. Create a box plot showing energy differences between physiological/non-physiological
```

### Exercise 2.2: Feature Engineering
**Difficulty**: Medium
**Time**: 25 minutes

Improve the feature engineering process:
```python
# Tasks:
# 1. Create new features (ratios, differences, etc.)
# 2. Implement feature selection using correlation analysis
# 3. Try different scaling methods (MinMaxScaler, RobustScaler)
# 4. Handle any missing values appropriately
# 5. Compare model performance with original vs. engineered features
```

### Exercise 2.3: Model Architecture Experimentation
**Difficulty**: Medium
**Time**: 30 minutes

Experiment with different neural network architectures:
```python
# Tasks:
# 1. Try different numbers of hidden layers (1, 3, 5)
# 2. Experiment with different activation functions (ReLU, LeakyReLU, ELU)
# 3. Test different dropout rates (0.1, 0.3, 0.5)
# 4. Compare performance of different optimizers (Adam, SGD, RMSprop)
# 5. Find the best hyperparameter combination
```

### Exercise 2.4: Advanced Evaluation Metrics
**Difficulty**: Medium
**Time**: 20 minutes

Implement comprehensive model evaluation:
```python
# Tasks:
# 1. Calculate ROC curve and AUC score
# 2. Implement precision-recall curve
# 3. Calculate F1-score, precision, recall for each class
# 4. Create a confusion matrix visualization
# 5. Implement k-fold cross-validation
```

### Exercise 2.5: Model Interpretability
**Difficulty**: Hard
**Time**: 30 minutes

Deep dive into model interpretability:
```python
# Tasks:
# 1. Implement SHAP values for feature importance
# 2. Create partial dependence plots for top features
# 3. Analyze prediction confidence distributions
# 4. Identify and analyze misclassified samples
# 5. Create a feature importance ranking visualization
```

## Advanced Exercises

### Exercise 3.1: Transfer Learning
**Difficulty**: Hard
**Time**: 45 minutes

Apply transfer learning to the protein classification task:
```python
# Tasks:
# 1. Pre-train a model on a larger protein dataset (if available)
# 2. Fine-tune the pre-trained model on the dimers dataset
# 3. Compare performance with training from scratch
# 4. Analyze which layers benefit most from pre-training
# 5. Implement learning rate scheduling for fine-tuning
```

### Exercise 3.2: Ensemble Methods
**Difficulty**: Hard
**Time**: 40 minutes

Implement ensemble methods for improved performance:
```python
# Tasks:
# 1. Train multiple models with different architectures
# 2. Implement voting (hard and soft) for classification
# 3. Use bagging with bootstrap sampling
# 4. Implement stacking with a meta-learner
# 5. Compare ensemble performance with individual models
```

### Exercise 3.3: Custom Loss Functions
**Difficulty**: Hard
**Time**: 35 minutes

Implement custom loss functions for imbalanced data:
```python
# Tasks:
# 1. Implement focal loss for handling class imbalance
# 2. Create weighted cross-entropy loss
# 3. Implement contrastive loss for learning representations
# 4. Compare performance with standard cross-entropy
# 5. Analyze the impact on model predictions
```

## Project-Based Exercises

### Project 1: Complete ML Pipeline
**Difficulty**: Medium-Hard
**Time**: 2-3 hours

Build a complete machine learning pipeline:
```python
# Requirements:
# 1. Data preprocessing pipeline with configurable parameters
# 2. Model training with hyperparameter tuning
# 3. Model evaluation and comparison
# 4. Model saving and loading functionality
# 5. Prediction pipeline for new data
# 6. Comprehensive documentation and testing
```

### Project 2: Real-World Application
**Difficulty**: Hard
**Time**: 4-6 hours

Apply PyTorch to a real-world problem:
```python
# Choose one of the following:
# 1. Image classification with custom dataset
# 2. Time series forecasting
# 3. Natural language processing task
# 4. Reinforcement learning problem
# 5. Generative model implementation

# Requirements:
# - Use real data
# - Implement proper evaluation metrics
# - Handle data preprocessing challenges
# - Optimize for performance
# - Create a presentation of results
```

## Assessment Criteria

### For Each Exercise:
- **Correctness** (40%): Does the code work and produce expected results?
- **Efficiency** (20%): Is the implementation efficient and well-structured?
- **Understanding** (25%): Does the student understand the concepts?
- **Creativity** (15%): Does the student go beyond the basic requirements?

### Grading Scale:
- **A (90-100%)**: Excellent work, goes beyond requirements
- **B (80-89%)**: Good work, meets all requirements
- **C (70-79%)**: Satisfactory work, meets most requirements
- **D (60-69%)**: Needs improvement, meets some requirements
- **F (<60%)**: Unsatisfactory, doesn't meet requirements

## Tips for Students

1. **Start Early**: Don't wait until the last minute
2. **Read Carefully**: Make sure you understand the requirements
3. **Test Incrementally**: Test your code as you write it
4. **Document**: Add comments and explanations
5. **Ask Questions**: Don't hesitate to ask for clarification
6. **Experiment**: Try different approaches and learn from mistakes
7. **Review**: Go back and review your solutions

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Machine Learning Mastery](https://machinelearningmastery.com/)
- [Papers With Code](https://paperswithcode.com/)

Good luck with your PyTorch learning journey! 🚀 