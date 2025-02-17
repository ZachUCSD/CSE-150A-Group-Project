# CSE-150A-Group-Project

## Milestone 2

## Data Exploration
The dataset contained 541909 rows and 6 columns representing the purchases made by customers of a UK-based and registered non-store online retail. This dataset has the following features.

- Description: Contains a short description of the item
- Quantity: The number of the item purchased
- Invoice Date: The time of purchase
- Unit Price: Cost per item
- CustomerID: A unique identifier given to each customer
- Country: Country of the purchaser

From the summaries of the features it can be seen that Quantity and Unit Price have dramatic outliers; for this reason, when developing our Hidden Markov Model (HMM), we consider the logarithm of the total purchase. This is calculated by computing the total purchase price (quantity times unit price) and then rescaling by taking the logarithm. In doing so, we reduce the variability in the data (we are more equipped to deal with outliers), the model can be trained using more stable data through a smoother distribution of the data, and we maintain the relative ordering given by the purchases. It can also be seen that most of the purchases are made in the UK.

## Data Cleaning
The data contained null values in Description as well as CustomerID so rows with either of them missing were dropped. Country was label-encoded to allow for numerical operations to be performed on it such as correlation matrices. We converted Invoice Date into a datetime object in order to create the hidden states to our model based on the months in which items were purchased. We originally planned to use the fiscal quarters, but we then realized that in order to consider more hidden states in our model, it made more sense to switch our time periods to months. 

In addition to computing the log of the total purchases, we added a couple of additional columns to our main data frame: 'PurchaseFrequency', denotes the total number of purchases each customer has made, and 'TotalItemsBought' has the total number of items bought by a particular customer. 

### Explain what your AI agent does in terms of PEAS. What is the "world" like? 
- Performance Measure: Our current performance measure is accuracy, which denotes the percentage of the predicted states that match the hidden states of a customer's purchase. 
- Environment: The current time based data using Invoice Dates and the months purchases were made as well as the purchasing information CustomerID and the log of the total purchase price.
- Actuators: the current model are the months in which customers purchased items. 
- Sensors: Currently would be the input data we are feeding the HMM, which include the log of the total purchase, the customer's ID, the country in which the purchase was made, etc. 

### What kind of agent is it? Goal based? Utility based? etc.
Our agent is goal-based, and we want it to able to take in a CustomerID and return the kinds of purchases a customer may be interested in, including what items they may be interested in buying. 



### Describe how your agent is set up and where it fits in probabilistic modeling
In the Jupyter Notebook, we develop the CustomerPurchaseAgent, designed to predict and influence customer purchasing behavior using Hidden Markov Models (HMMs) and probabilistic reasoning. The agent
* predicts customer behavior by mapping hidden states to observed purchasing patterns,
* processes customer data by normalizing key purchase indicators and encoding categorical variables,
* uses Conditional Probability Tables (CPTs) to assess purchase likelihood based on three CPTs (country & quantity: how likely a customer from a given country buys a certain quantity; product & unit price: probability of an item being purchased at a given price; and total Customer Spending, which tracks overall spending patterns to refine recommendations), and uses the CPTs to
* recommend personalized marketing strategies based on predicted purchasing behavior and seasonal trends.

The agent does the following: 
* Data preprocessing: it computes log-transformed purchase values to normalize spending patterns, encodes categorical features, and standardizes numerical features.
* Predicts future behavior: it uses a trained HMM model to predict the most likely hidden state for each customer transaction.
* Generates personalized purchasing recommendations: it combines CPT-based probabilities with customer spending data to compute a personalized purchase probability, and suggests marketing actions based on the predicted state (month) and purchase likelihood.

Our Hidden Markov Model is set up in uch a way that it tries to capture hidden patterns in sequential transaction data. The HMM has hidden states based on each month, which represent a different phase in a customer's purchasing behavior and has observations (the log-transformed purchase amounts, the country of purchase, the purchase frequency, ad the total items bought by customers). The HMM is trained on monthly customer transactions to understand purchase trends and seasonal patterns, and then the model develops predictions by assigning each transaction to a most likely hidden state (month) as a qay of forecasting what future customer activity might look like. We are making use of a Gaussian HMM as our current model, which innately assumes that observations (features like LogTotalPurchase, PurchaseFrequency, etc.) are normally distributed within each hidden state (month). 


### Evaluate your model
Based on the features we have developed and trained our model on (i.e., log of total purchases, country of purchase, purchase frequency, and total items purchased), the model currently has roughly a 0.09 accuracy rate. This accuracy is particularly poor, and although it varies because our model is trained by looking at specific random samples of our data (instead of all of our data), it has consistenyl been between 0.07 and 0.10. 

The predictions made by the agent given the trained HMM model are also particularly generic in that they do not really focus on the specific itesms and instead propose recommendations for what customers should do in regards to future purchases. 

### Conclusion
Our current agent works very poorly and does not accomplish the goals we want it to. In order to improve this model, we would like to optimize the code further as well as find additional datasets that would give us the information we need to take in a customerID and provide more specific item-based purchase recommendations. We are thinking to use either the descriptions of items or find a dataset that has better labels for item types for the recommendation system. We are also thinking about changing the type of HMM that we're using as a Gaussian HMM assumes the observations are each normally distributed within each hidden state, which may not necessarily be the case, and is quite computationally intensive since we have to assume the covariance matrix between the various observations is 'full'. 
