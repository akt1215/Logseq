- [Perplexity](https://www.perplexity.ai/search/explain-how-support-vector-mac-iW1dZVjkRXi5SQCj09PH9w)
-
- Define a hyperplane that separates the dataset
  logseq.order-list-type:: number
	- The equation of the hyperplane is defined as: $\omega \cdot x + b = 0$
	  logseq.order-list-type:: number
		- $\omega$ ... weight vector
		  logseq.order-list-type:: number
		- $x$ ... input feature vector
		  logseq.order-list-type:: number
		- $b$ ... bias term (to shift the hyperplane)
		  logseq.order-list-type:: number
	- Depending on it's sign, we can classify into two labels
	  logseq.order-list-type:: number
- Maximize the margin defined as: 
  logseq.order-list-type:: number
	- Margin $=\frac{2}{||\omega||}$
		- Distance $=\frac{|\omega \cdot x + b|}{||\omega||}$ because $|\omega \cdot x + b|$ is a projection onto the $\vec{\omega}$
		  logseq.order-list-type:: number
	- This is the same as finding $\min \frac{1}{2}||\omega||^2$ subject to $y_i(\omega\cdot x_i + b) \geq 1, \forall i$
	  logseq.order-list-type:: number
		- This is convex optimization meaning it has one global minimum since its Hessian matrix (second derivative) is positive definite ($x^T Mx \geq 0$)
		  logseq.order-list-type:: number
- Soft-Margin SVM
  logseq.order-list-type:: number
	- Introduce slack variable $\xi_i \geq 0$ to allow some misclassification
	  logseq.order-list-type:: number
		- logseq.order-list-type:: number
		  $$y_i(\omega\cdot x_i + b) \geq 1-\xi_i, \forall i$$
		- If $\xi_i=0$, the point is correctly classified
		  logseq.order-list-type:: number
		- If $0<\xi_i<1$, the point lies within the margin but on the correct side of the plane
		  logseq.order-list-type:: number
		- If $\xi > 1$, the point is misclassified
		  logseq.order-list-type:: number
	- The Objective function becomes
	  logseq.order-list-type:: number
		- logseq.order-list-type:: number
		  $$\min \frac{1}{2}||\omega||^2+C\sum_{i=1}^N{\xi_i}$$
- Dual Formation for efficiency
  logseq.order-list-type:: number
	- logseq.order-list-type:: number
	  $$\max \sum_{i=1}^N{\alpha_i} - \frac{1}{2} \sum_{i=1}^N\sum_{j=1}^Ny_iy_j\alpha_i\alpha_j(x_i \cdot x_j)$$
	- where $0\leq\alpha_i\leq C, \sum_{i=1}y_i\alpha_i$
	  logseq.order-list-type:: number
	- And we can solve for $\alpha_i$ to know if the example is support vector or not
	  logseq.order-list-type:: number
	- This is come from the following using Lagrange multiplier
	  logseq.order-list-type:: number
		- logseq.order-list-type:: number
		  $$L(\omega, b, a) = \frac{1}{2}||\omega||^2 - \sum_{i=1}^N{\alpha_i[y_i(\omega\cdot x_i + b)-1]}$$
		- Minimize L with respect to \omega
		  logseq.order-list-type:: number
			- logseq.order-list-type:: number
			  $$\frac{\partial{L}}{\partial{\omega}} = \omega - \sum_{i=1}^N\alpha_i y_i x_i = 0$$
			- logseq.order-list-type:: number
			  $$\Leftrightarrow \omega = \sum_{i=i}^{N}\alpha_iy_ix_i$$
			- logseq.order-list-type:: number
			  $$\Leftrightarrow \frac{1}{2}||\omega||^2 = \frac{1}{2} \sum_{i=1}^N\sum_{j=1}^Ny_iy_j\alpha_i\alpha_j(x_i \cdot x_j)$$
		- Similarly, minimize L with respect to $b$
		  logseq.order-list-type:: number
			- logseq.order-list-type:: number
			  $$\frac{\partial{L}}{\partial{b}} = - \sum_{i=1}^N\alpha_i y_i = 0$$
			- logseq.order-list-type:: number
			  $$\Leftrightarrow -\sum_{i=1}^N{\alpha_i[y_i(\omega\cdot x_i + b)-1]} = \sum_{i=1}^N\alpha_i$$
		- Therefore, we get
		  logseq.order-list-type:: number
			- logseq.order-list-type:: number
			  $$L(\omega, b, a) = \frac{1}{2}||\omega||^2 - \sum_{i=1}^N{\alpha_i[y_i(\omega\cdot x_i + b)-1]}=\max \sum_{i=1}^N{\alpha_i} - \frac{1}{2} \sum_{i=1}^N\sum_{j=1}^Ny_iy_j\alpha_i\alpha_j(x_i \cdot x_j)$$
- Kernel Trick
  logseq.order-list-type:: number
	- Recall the decision function is 
	  logseq.order-list-type:: number
		- logseq.order-list-type:: number
		  $$f(x) = \omega^T x + b$$
	- Since we know that $$\omega = \sum_{i=i}^{N}\alpha_iy_ix_i$$, we get
	  logseq.order-list-type:: number
		- logseq.order-list-type:: number
		  $$f(x) = (\sum_{i=i}^{N}\alpha_iy_ix_i)^Tx+b$$
		- logseq.order-list-type:: number
		  $$= \sum_{i=i}^{N}\alpha_iy_i(x_i^Tx)+b$$
	- For non-linear problems, this is particularly useful as we can replace the dot product with **kernel function** $K(x_i, x_j) = \phi(x_i)^T\phi(x_j)$ where $\phi(x)$ maps data into a higher-dimension space
	  logseq.order-list-type:: number
- # Higher Dimension Application
	- One-vs-Rest  (OvR)
		- Procedure
			- Foe N classes, N separate binary classifiers are trained
			- Each classifier distinguishes one class from the others
			- Each classifier outputs a decision score, and the class with the highest decision score is chosen as the predicted label
		- Advantages
			- Computationally efficient
		- Disadvantages
			- Can struggle with imbalanced datasets because one class is compared against all others
		-
	- One-vs-One (OvO)
		- Procedure
			- For $N$ classes $\frac{N(N-1)}{2}$ binary classifiers are trained
			- Each classifier distinguishes between a pair of classes
			- Each classifier votes for one of its two classes and the class with the most votes is chosen as the predicted label
		- Advantages
			- Often more accurate than OvR
		- Disadvantages
			- Computationally expensive due to the large number of classifiers required
-
- Machine Learning Library for multi-class SVM
- ```python
  from sklearn.svm import SVC
  
  # One-vs-One strategy
  clf_ovo = SVC(decision_function_shape='ovo')
  clf_ovo.fit(X_train, y_train)
  
  # One-vs-Rest strategy
  clf_ovr = SVC(decision_function_shape='ovr')
  clf_ovr.fit(X_train, y_train)
  ```
- logseq.order-list-type:: number
- # SVM From Scratch
- logseq.order-list-type:: number
- ```python
  import numpy as np
  
  class SVM:
      def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
          """
          Initialize SVM parameters
          
          Parameters:
          - learning_rate: Step size for gradient descent
          - lambda_param: Regularization strength
          - n_iters: Number of training iterations
          """
          self.lr = learning_rate
          self.lambda_param = lambda_param
          self.n_iters = n_iters
          self.w = None
          self.b = None
      
      def fit(self, X, y):
          """
          Train the SVM using Stochastic Gradient Descent
          
          Parameters:
          - X: Training features (shape: [n_samples, n_features])
          - y: Training labels (+1 or -1)
          """
          n_samples, n_features = X.shape
          # Initialize weights and bias to zero
          self.w = np.zeros(n_features)
          self.b = 0
          
          # Stochastic Gradient Descent for optimization
          for _ in range(self.n_iters):
              for idx, x_i in enumerate(X):
                  condition = y[idx] * (np.dot(x_i, self.w) + self.b)
                  
                  if condition >= 1:
                      # Correctly classified with margin; apply only regularization term
                      dw = self.lambda_param * self.w
                      db = 0
                  else:
                      # Misclassified or within margin; update weights and bias
                      dw = self.lambda_param * self.w - y[idx] * x_i
                      db = -y[idx]
                  
                  # Update weights and bias using gradient descent
                  self.w -= self.lr * dw
                  self.b -= self.lr * db
      
      def predict(self, X):
          """
          Predict labels for input features
          
          Parameters:
          - X: Input features (shape: [n_samples, n_features])
          
          Returns:
          - Predicted labels (+1 or -1)
          """
          linear_output = np.dot(X, self.w) + self.b
          return np.sign(linear_output)
  
  # Generate a synthetic dataset for testing the SVM implementation
  def generate_data(n_samples=100):
      np.random.seed(42)
      # Class 1: Centered around (2, 2)
      X1 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
      y1 = np.ones(n_samples // 2)
      
      # Class 2: Centered around (-2, -2)
      X2 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
      y2 = -np.ones(n_samples // 2)
      
      # Combine datasets
      X = np.vstack((X1, X2))
      y = np.hstack((y1, y2))
      
      return X, y
  
  # Generate dataset and split into training and testing sets
  from sklearn.model_selection import train_test_split
  
  X, y = generate_data()
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  # Train the SVM model on the training data
  svm = SVM(learning_rate=0.01, lambda_param=0.01, n_iters=1000)
  svm.fit(X_train, y_train)
  
  # Evaluate the model on training and testing data
  def accuracy(y_true, y_pred):
      return np.mean(y_true == y_pred)
  
  train_predictions = svm.predict(X_train)
  test_predictions = svm.predict(X_test)
  
  print(f"Training Accuracy: {accuracy(y_train, train_predictions):.4f}")
  print(f"Testing Accuracy: {accuracy(y_test, test_predictions):.4f}")
  
  # Visualize the decision boundary
  import matplotlib.pyplot as plt
  
  plt.figure(figsize=(10, 6))
  plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Positive Class')
  plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Negative Class')
  
  # Plot decision boundary
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  
  xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                       np.arange(y_min, y_max, 0.1))
  Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  
  plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdBu)
  plt.xlabel('Feature 1')
  plt.ylabel('Feature 2')
  plt.title('SVM Decision Boundary')
  plt.legend()
  plt.show()
  ```
-