# CSE 150A, Group Project – Milestone 2 Write-Up

## Data Exploration
The dataset contained 541909 rows and 6 columns representing the purchases made by customers of a UK-based and registered non-store online retail. This dataset has the following features.

- Description: contains a short description of the item
- Quantity: the number of the item purchased
- Invoice Date: the time of purchase
- Unit Price: cost per item
- CustomerID: a unique identifier given to each customer
- Country: country of the purchaser

From the summaries of the features it can be seen that Quantity and Unit Price have dramatic outliers; for this reason, when developing our Hidden Markov Model (HMM), we consider the logarithm of the total purchase. This is calculated by computing the total purchase price (quantity times unit price) and then rescaling by taking the logarithm. In doing so, we reduce the variability in the data (we are more equipped to deal with outliers), the model can be trained using more stable data through a smoother distribution of the data, and we maintain the relative ordering given by the purchases. It can also be seen that most of the purchases are made in the UK.

## Data Cleaning
The data contained null values in Description as well as CustomerID so rows with either of them missing were dropped. Country was label-encoded to allow for numerical operations to be performed on it such as correlation matrices. We converted Invoice Date into a datetime object in order to create the hidden states to our model based on the months in which items were purchased. We originally planned to use the fiscal quarters, but we then realized that in order to consider more hidden states in our model, it made more sense to switch our time periods to months. 

In addition to computing the log of the total purchases, we added a couple of additional columns to our main data frame: `PurchaseFrequency`, denotes the total number of purchases each customer has made, and `TotalItemsBought` has the total number of items bought by a particular customer. 

## Describe how your agent is set up and where it fits in probabilistic modeling
Our Jupyter Notebook can be thought of being split into three sections: a data processing section (where the data is cleaned and appropriately organized - see the "Data Cleaning" section above), the CustomerPurchaseAgent class (which creates our agent), and the integration of the HMM section (which creates and develops the model, and then incorporates it for our agent's prediction capabilities). 

### High level overview of our agent
The main goal of our CustomerPurchaseAgent is to use past retail transaction data for a particular customer and be able to provide relevant recommendations for that customer's future purchasing activity based on probabilistic reasoning. Our agent takes in raw retail transactions and transforms those purchases into meaningful features (such as the log-transformed total purchase, the purchase frequency, and the total items bought). Our agent then uses a trained HMM that is trying to identify underlying temporal patterns in customer behavior; the time periods on which the model gets trained varies, although the two most effective ones are months or fiscal quarters (more on this a little later). Then, based on the trained model, the agent tries to influence the purchasing behavior of a given customer. In practice, the agent 
* preprocesses and cleans customer data (i.e., handling null or missing values, normalizing key purhasing indicators, encoding categorical variables, transforming the target variables)
* uses a trained Gaussian HMM model to predict the most likely hidden state for each customer's transactions as a means of capturing latent purchasing patterns over time,
* uses the hidden states from the trained Gaussian HMM and Conditional Probability Tables (CPTs) to generate purchase recommendations and marketing actions tailored to individual customers.

Our agent uses three Conditional Probabilities Tables (CPTs) to help make purchasing recommendations for a customer: one based on country & quantity (how likely a customer from a given country buys a certain quantity), one based on product & unit price (how likely it is than item is purchased at a given price), and one based on total customer spending (how likely it is that a customer made certain purchases were made given how much those items cost). 


### Explain what your AI agent does in terms of PEAS. What is the "world" like? 
The PEAS for our CustomerPurchaseAgent, as described from our first milestone, are:
- Performance Measure: accuracy, which denotes the percentage of the predicted states that match the hidden states of a customer's purchase. 
- Environment: the current time based data using Invoice Dates and the months purchases were made as well as the purchasing information CustomerID and the log of the total purchase price.
- Actuators: the current model are the months in which customers purchased items. 
- Sensors: processed retrail transactional data that the Guassian HMM model is trained on; the data includes variables like the log of the total purchase, the customer's ID, the country in which the purchase was made, etc.


### What kind of agent is it? Goal based? Utility based? etc.
Our agent is goal-based: we want it to able to take in a CustomerID and return the kinds of purchases a customer may be interested in at a particular time. 



## Model of Choice

### Choice for a Gaussian HMMM
We wanted to capture patterns in sequential transaction data for specific customers, so it made sense to use a Hidden Markov Model, in which observations are directly dependent upon a hidden state at a particular time. Since the total purchase values are naturally modeled by a Gaussian distribution,  we felt like it was reasonable to have our model of choice be a Gaussian Hidden Markov Model, or a Guassian HMM. Each hidden state's data is normally distributed with its own normal distribution, and this model innately assumes that observations (features like `LogTotalPurchase`, `PurchaseFrequency`, etc.) are normally distributed given each hidden state. 

### Technical Overview of Gaussian HMM
A Gaussian HMM assumes that each hidden state generates observations according to a Gaussian (normal) distribution. The training of the model involves an expectation-maximization (EM) process. During the expectation step, the algorithm calculates forward and backward probabilities to estimate the likelihood of the model being in each hidden state at every time step. Then, in the maximization step, it updates the parameters: the state transition probabilities, the initial state distribution, and the Gaussian parameters (means/covariance matrices for each state) to maximize the likelihood of the observed data. The model continues updating until the log-likelihood falls below a convergence threshold or the model has performed its maximum number of updating iterations. The result is a model that captures the underlying patterns in our continuous purchase data.

In our implementation, we run the model by using the 

```model = hmm.GaussianHMM(n_components=n_hidden_states, covariance_type="full", n_iter=1000, random_state=42).```

The `n_hidden_states` were generated based on the time periods we chose to run our model on. We discretized our datetime object based on month or fiscal quarter, the two time period intervals we thought would be best for our model. In the end, we decided that using months would be more helpful in creating more hidden states for our HMM, and so we moved away from fiscal quarters. The choice for iterations (i.e., `n_iter = 1000`) gives the model a maximimum number of iterations to perform for convergence. There is a trade-off between the number of iterations of the model needed for convergence and the runtime of the training. If the number of iterations the model will perform is very large, then it is more likely that the log-likelhood from the EM algorithm will fall below the convergence threshold; but increasing the number of iterations also increases the runtime of the program. Conversely, if the number of iterations the model will perform is small, then it is less likely that the log-likelihood from the EM algorithm will fall below the convergence threshold; but decreasing the number of iterations also decreases the runtime of the program. The `random_state` is functions similarly to `random.set_seed(42)`, and this basically sets a random number generator instance. Lastly, the `covariance_type` dictates the type of covariance parameters to use for the model. Since the joint probability across all states is a multivariate normal distribution, we must have a covariance parameter that represents thsi distribution. The two types of covariance that we considered using for our model were "diag" and "full": "diag" assumes each state uses a diagonal covariance matrix, or that each of the features in our model are independent of each other. On the other hand, "full" assumes each state has a unresitricted, or full, covariance matrix. We settled on using the "full" covariance matrix because this would allow us to capture the correlation, or the relationship, between some of our features. It did not make sense to assume that `LogTotalPurchase` and `PurchaseFrequency`, let's say, were uncorrelated because, intuitively and empirically, a higher `PurchaseFrequency` would translate to a higher `LogTotalPurchase`. Thus, using the "full" covariance matrix would help us be sure that in our model's training pahse, it can start to learn some of the relationships between our features. Theoretically, this would help our CustomerPuchaseAgent make tailored recommendations for each customer. 



### Evaluate your model
Based on the features we have developed and trained our model on (i.e., log of total purchases, country of purchase, purchase frequency, and total items purchased), the model currently has roughly a 0.09 accuracy rate. This accuracy is particularly poor, and although it varies because our model is trained by looking at specific random samples of our data (instead of all of our data), it has consistenyl been between 0.07 and 0.10. 

The predictions made by the agent given the trained HMM model are also particularly generic in that they do not really focus on the specific items a customer may want to purchase and instead propose recommendations for what customers should do in regards to future types of purchases. (Here, we define "type of purchases" to be categorized based on purchases for some particular set of circumstances. Those circumstances include "pre-holiday deals" or "summer discounts".) 



## Conclusions
Our current CustomerPurchaseAgent works very poorly and does not accomplish the goals we want it to. In order to improve this model, we would like to optimize the code further as well as find additional datasets that would give us the information we need to take in a customerID and provide more specific item-based purchase recommendations (see some comments from the *Evaluate your model* section above). We are thinking to use either the descriptions of items or find a dataset that has better labels for item types for the recommendation system. We are also thinking about changing the type of HMM that we have been relying on since our Gaussian HMM assumes the observations are each normally distributed within each hidden state, which may not necessarily be the case (see comments in *Technical Overview of Gaussian HMM* section). Moreover, since we figured our covariance matrix would be "full", this model is quite computationally intensive. 
