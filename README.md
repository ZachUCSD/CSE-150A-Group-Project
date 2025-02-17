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

From the summaries of the features it can be seen that Quantity and Unit Price have dramatic outliers. It can also be seen that most of the purchases are made in the UK.

## Data Cleaning
The data contained null values in Description as well as CustomerID so rows with either of them missing were dropped. Country was labelencoded to allow for numerical operations to be performed on it such as correlation matricies. Converted Invoice Date into a datetime object.

### Explain what your AI agent does in terms of PEAS. What is the "world" like? 
- Performance Measure: Our current performance measure is Accuracy which is predicting 
- Environment: Is current the time based data using Invoice Dates and Quarters as well as the purchasing information CustomerID and TotalPurchase.
- Actuators: The Actuators for the current model are the Quarters.
- Sensors: Currently would be the input data we are feeding the HMM. TotalPurchase, CustomerID, etc

### What kind of agent is it? Goal based? Utility based? etc.
Our Agent is a goal based agent and we want it to able to take in a CustomerID and return items that the customer may be interested in.

### Describe how your agent is set up and where it fits in probabilistic modeling
Currently the HMM is setup in such a way that it is trying to find quarterly trends in the data by splitting the Invoice Date into 4 sections. It is using these quarters as a hidden state and using TotalPurchase as the observed state to attempt to predict how much a customer would purchase during a quarter. We are also using a Gaussian HMM as our current model.

### Evaluate your model
The model currently has around a 0.09 accuracy rate which is extremely poor.

### Conclusion
Our current agent works very poorly and does not accomplish the goals we want it to. In order to improve this model we would like to optimize the code further as well as find additional datasets that would give us the information we need to take in a customerID and provide recommendations. We are thinking to use either the descriptions of items or find a dataset that has labels for item types for the recommendation system. Changing the type of HMM would likely help as Gaussian HMM does not fit the data and the distributions we find within it.
