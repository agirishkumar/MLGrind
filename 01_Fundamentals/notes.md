# Statistical Models

**Key Concepts:**

* **Population:** The entire group of interest.
* **Sample:** A subset of the population.
* **Variables:**
    * **Independent:**  Manipulated or changed.
    * **Dependent:** Affected by changes in the independent variable.
* **Parameters:** Numerical values describing the population.
* **Statistics:** Numerical values calculated from a sample.
* **Hypothesis Testing:**  Evaluating evidence to support a claim.
    * **Null Hypothesis (H0):**  A statement of no effect or no difference.
    * **Alternative Hypothesis (H1):** A statement that contradicts the null hypothesis.
    * **p-value:** The probability of observing the data if the null hypothesis is true.
* **Statistical Significance:**  A result unlikely to occur by chance (often p-value < 0.05).
* **Confidence Intervals:** A range of values likely to contain the true parameter.


**Example: Hypothesis Testing**

Let's say we want to test if a new drug is effective in lowering blood pressure.

* **H0:** The drug has no effect on blood pressure.
* **H1:** The drug lowers blood pressure.

We collect data from a sample of patients and calculate the average blood pressure before and after taking the drug. We then use a statistical test (e.g., t-test) to calculate the p-value.

* **If p-value < 0.05:** We reject the null hypothesis and conclude that the drug is effective.
* **If p-value >= 0.05:** We fail to reject the null hypothesis and conclude that there is not enough evidence to say the drug is effective.

**Confidence Interval Example:**

A 95% confidence interval for the mean blood pressure reduction might be (5 mmHg, 10 mmHg). This means we are 95% confident that the true mean reduction in blood pressure in the population lies between 5 mmHg and 10 mmHg.

**Important Note:**
Statistical models rely on assumptions about the data. It's important to check these assumptions before drawing conclusions.



# Bayesian Statistics

**Key Concepts:**

* **Prior Distribution (P(θ)):**  Initial beliefs about parameters (θ).
    * **Informative Prior:** Strong prior knowledge.
    * **Uninformative Prior:**  Weak or vague prior knowledge.
* **Likelihood Function (P(D|θ)):** Probability of data (D) given parameters.
* **Posterior Distribution (P(θ|D)):** Updated beliefs after observing data.
* **Bayes' Theorem:** P(θ|D) = [P(D|θ) * P(θ)] / P(D) 
    * Often simplified to:  Posterior ∝ Likelihood * Prior
* **Credible Intervals:**  Intervals containing a specified percentage of posterior probability.

**Advantages of Bayesian Inference:**

* **Incorporates prior knowledge:** Can leverage expert knowledge or previous data.
* **Quantifies uncertainty:** Provides a full probability distribution over parameters.
* **Flexible and adaptable:** Can handle complex models and missing data.

**Example:**

Suppose you want to estimate the probability (θ) of a coin landing heads.

1. **Prior:** You might start with a uniform prior, assuming any value of θ between 0 and 1 is equally likely.
2. **Likelihood:** You flip the coin 10 times and observe 7 heads. The likelihood function describes the probability of getting this data given different values of θ.
3. **Posterior:** Using Bayes' theorem, you combine the prior and likelihood to get the posterior distribution, which will now be centered around a higher value of θ (reflecting the observed data).
4. **Credible Interval:**  You can calculate a 95% credible interval from the posterior distribution, which will give you a range of plausible values for θ.

**Important Note:**
Choosing appropriate priors is crucial in Bayesian analysis.


# Probability Distributions

**Discrete Distributions:**

* **Bernoulli:**
    * P(X = 1) = p (probability of success)
    * P(X = 0) = 1 - p (probability of failure)
* **Binomial:**
    * P(X = k) = (n choose k) * p^k * (1 - p)^(n - k) 
        * n: number of trials
        * k: number of successes
        * p: probability of success in a single trial
* **Poisson:**
    * P(X = k) = (λ^k * e^(-λ)) / k!
        * λ: average rate of events

**Continuous Distributions:**

* **Normal (Gaussian):**
    * f(x) = (1 / (σ * sqrt(2π))) * exp(-(x - μ)^2 / (2σ^2))
        * μ: mean
        * σ: standard deviation
* **Uniform:**
    * f(x) = 1 / (b - a) for a ≤ x ≤ b
        * a: lower bound
        * b: upper bound
* **Exponential:**
    * f(x) = λ * exp(-λx) for x ≥ 0
        * λ: rate parameter

**Important Concepts:**

* **Mean (Expected Value):** The average value of the distribution.
* **Variance:**  A measure of the spread of the distribution.
* **Standard Deviation:** The square root of the variance.

# Probability Theory

**Key Concepts:**

* **Random Experiment:** A process with uncertain outcomes.
* **Sample Space (S):** Set of all possible outcomes.
* **Event (E):** A subset of the sample space.
* **Probability (P(E)):** Likelihood of an event (0 ≤ P(E) ≤ 1).

**Formulas:**

* **Probability of an Event:**  P(E) = Number of favorable outcomes / Total number of possible outcomes
* **Probability of the Complement of an Event:** P(E') = 1 - P(E) (where E' is the event "not E")
* **Probability of the Union of Two Events:** P(A ∪ B) = P(A) + P(B) - P(A ∩ B) (where ∩ represents intersection)
* **Conditional Probability:** P(A|B) = P(A ∩ B) / P(B)  (probability of A given B)
* **Bayes' Theorem:** P(A|B) = [P(B|A) * P(A)] / P(B)

**Important Theorems and Laws:**

* **Law of Large Numbers:** As the number of trials increases, the average of the results converges to the expected value.
* **Central Limit Theorem:** The sum of a large number of independent random variables tends towards a normal distribution.

**Random Variables:**

* **Discrete Random Variable:**  Takes on a finite number of values (e.g., number of heads in coin flips).
* **Continuous Random Variable:** Takes on any value within a given range (e.g., height, weight).

# Conditional Probability

* **P(A|B):** Probability of A given B.
* **Formula:** P(A|B) = P(A and B) / P(B)

# Law of Large Numbers

* As the number of trials increases, the average of the results approaches the expected value.

# Central Limit Theorem

* The sum of many i.i.d. (independent and identically distributed) random variables approximates a normal distribution.
* Allows us to use the normal distribution for inference, even when the population is not normally distributed.
* The CLT is fundamental to statistical inference because it allows us to use the normal distribution to make inferences about population parameters, even if the population itself is not normally distributed.


# Linear Algebra (Continued)

## Matrices

* **Dimensions:** m x n (rows x columns)
* **Elements:** a_ij (element at row i, column j)
* **Types:** Square, Diagonal, Identity, Symmetric
* **Transpose:** A^T (rows and columns interchanged)

## Eigenvectors and Eigenvalues

* **Eigenvector (v):**  A non-zero vector that only scales when multiplied by a matrix.
* **Eigenvalue (λ):** The scaling factor.
* **Equation:** A * v = λ * v

## Matrix Decompositions

* **Singular Value Decomposition (SVD):** A = UΣV^T
    * **Applications:** PCA, recommendation systems, noise reduction.
* **QR Decomposition:** A = QR
    * **Applications:** Solving linear systems, least squares problems.

## Linear Transformations

* Functions that map vectors while preserving linear combinations.
* **Properties:**
    * Additivity: T(u + v) = T(u) + T(v)
    * Homogeneity: T(cu) = cT(u)
* **Examples:** Rotation, scaling, projection.

# Calculus

**Key Concepts:**

* **Limits:**  lim_(x->a) f(x) = L  (The value f(x) approaches as x approaches a)
* **Derivatives:**
    * **Notation:** f'(x), df/dx
    * **Interpretation:** Instantaneous rate of change, slope of the tangent line.
    * **Rules:** Power rule, product rule, quotient rule, chain rule.
* **Chain Rule:** d/dx f(g(x)) = f'(g(x)) * g'(x)
* **Partial Derivatives:** ∂f/∂x (derivative of f with respect to x, holding other variables constant)
* **Gradients:** ∇f = [∂f/∂x1, ∂f/∂x2, ..., ∂f/∂xn] (vector of partial derivatives)
* **Integrals:** ∫f(x) dx (area under the curve of f(x))
* **Taylor Series:**  Approximating a function using its derivatives at a point.

**Applications in Machine Learning:**

* **Optimization:** Gradient descent uses derivatives to find the minimum of a loss function.
* **Model Analysis:** Derivatives help understand how model predictions change with inputs.
* **Deep Learning:** Backpropagation uses the chain rule to calculate gradients in neural networks.

## Multivariate Calculus

* **Functions of several variables:** f(x, y, z, ...)
* **Directional derivatives:** Rate of change in any direction.
* **Multiple integrals:**  Integrating over regions in multiple dimensions.
* **Vector calculus:** Operations on vector fields (gradient, divergence, curl).

## Chain Rule

* **Formula:** If y = f(u) and u = g(x), then dy/dx = dy/du * du/dx
* **Backpropagation:** Used to calculate gradients in neural networks.

## Optimization Techniques

* **Gradient Descent:**  An iterative algorithm that uses the gradient of the objective function to update the parameters in the direction of steepest descent. 
    * **Variations:** Batch, stochastic, mini-batch.
* **Newton's Method:** Uses the Hessian matrix.
* **Conjugate Gradient:** An iterative method for solving linear systems and optimizing functions.
* **Line Search:** A method for finding the optimal step size in gradient descent.

## Hessian Matrix

* Square matrix of second-order partial derivatives.
* **Applications:**
    * Optimization (Newton's method): to find the minimum of a function.
    * Checking for saddle points:  The eigenvalues of the Hessian can indicate whether a critical point is a minimum, maximum, or saddle point.

# Information Theory

**Key Concepts:**

* **Information:**  Measure of uncertainty or surprise.The less likely an event is, the more information it conveys.
* **Entropy (H(X)):** Average uncertainty of a random variable X.It quantifies how much information is needed, on average, to describe the outcome of that variable.
    * **Formula:** H(X) = - Σ p(x) * log2(p(x)) 
        * Where p(x) is the probability of outcome x.
* **Cross-entropy (H(p, q)):** Measures the difference between two probability distributions p and q. Often used as a loss function in machine learning for classification tasks.
    * **Formula:** H(p, q) = - Σ p(x) * log2(q(x))
* **KL Divergence (D_KL(p || q)):**  Measures how much p diverges from q.
    * **Formula:** D_KL(p || q) = Σ p(x) * log2(p(x) / q(x))
* **Mutual Information (I(X; Y)):** Measures shared information between X and Y. Used for feature selection and understanding relationships between variables.
    * **Formula:** I(X; Y) = H(X) + H(Y) - H(X, Y)
* **Coding Theory:** Efficient and reliable encoding of information for transmission and storage.

**Applications in Machine Learning:**

* **Data Compression:** Finding efficient ways to represent data.
* **Feature Selection:** Identifying the most informative features in a dataset.
* **Model Building:** Understanding the information flow in complex models like neural networks.
* **Natural Language Processing:** Analyzing language and building language models.