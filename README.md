# CSE-150A-Group-Project

## Our Agent
We are working on the same task from the previous two milestones, hoping to use a new model that produces better results. The main goal of our `CustomerPurchaseAgent` is to use past retail transaction data for a particular customer and be able to provide relevant recommendations for that customer's future purchasing activity based on probabilistic reasoning. Our agent takes in raw retail transactions and transforms those purchases into meaningful features (such as the log-transformed total purchase, the purchase frequency, and the total items bought). Our agent then uses a trained HMM that is trying to identify underlying temporal patterns in customer behavior; the time periods on which the model gets trained varies are months. Then, based on the trained model, the agent tries to predict the purchasing behavior of a given customer and to influence the purchasing behavior of a given customer by providing reasonable recommendations for future purchases. In practice, the agent
* uses a set of conditional probability tables (CPTs) to understand relationships between some of the key variables in the retail transactional data for a given customer,
* uses a trained Hidden Markov Model (HMM) to predict potential purchases for a given customer,
* uses the predicted hidden states from the model and the CPTs to propose purchasing recommendations for a given customer at a particular time.

The main CPTs our agent incorporates into its prediction and recommendation stage are: 
* `P(Product_Category \| CustomerID)`, denoting the probability of each product category given a customer,
* `P(Total_Spend \| CustomerID, Purchase_Month)`, denoting the probability of spending a particular amount by a customer in a particular month,
* `P(Avg_Price \| CustomerID, Product_Category)`, denoting the probability of paying an average price for a given customer for items in a particular category,
* `P(Days_Since_Last_Purchase \| CustomerID)`, denoting the probability of a certain number of days elapsing since a given customer last made a purchase, and 
* `P(Used_Coupon \| CustomerID, Product_Category)`, denoting the rrobability that a customer uses a coupon for a given category.

These CPTs help govern the prediction and recommendation stage for our agent, as well as being a part of our model (see ['Discussion of Model' section](#discussion_of_model) for more detail). Our agent is goal-based: we want it to able to take in a CustomerID and, perhaps given some more information, return some kinds of purchases a customer may be interested in at a particular time and return some purchasing recommendations for the customer. In code, our `CustomerPurchaseAgent` class has two key methods (aside from the constructor, which is somewhat self-explanatory): 
```
class CustomerPurchaseAgent:
    """
    CustomerPurchaseAgent class designed to:
    - Predict potential purchases for a given customer based on CPTs and trained HMM model.
    - Provide personalized recommendations tailored to the customer's behavior.
    - Evaluate performance by comparing predictions with actual purchase data.

    Methods:
    - predict_purchases(): Predicts categories, spending, and purchase behavior for a customer.
    - recommend_purchases(): Provides tailored recommendations based on known conditions.
    """
```
The constructor for our `CustomerPurchaseAgent` class allows us to initialize our agent by giving it a trained HMM model, our set of CPTs, and the dataset from which it can try and make predictions and recommendations. (The reason for including the dataset here is that it would allow for us to more easily compare the results of our agent's predictions against some of the actual purchases made by a given customer.) The PEAS for our agent remain consistent with our previous milestones, with some small changes: 
* Performance Measure: accuracy, which denotes the percentage of the predictions that match actual purchasing behvaior of a customer^*predicted states that match the hidden states of a customer's purchase.
* Environment: the current time-based data using transaction dates and other purchasing information like total amount spent, product categories, amongst other pieces of information.
* Actuators: the months in which customers purchased items, and the purchase categories those items would fall under.
* Sensors: processed retrail transactional data that the HMM model is trained on; the data includes variables like the total purchase amount spent, the customer's ID, the kind of purchase made, if a coupon was applied, etc.


## Data Processing and Model Setup
We found that our previous datasets were not as tailor-made for such a project as we would have liked, so we began working with a new dataset that seems to be more suited for this kind of project. The Kaggle dataset can be found at [this link.]([/guides/content/editing-an-existing-page#modifying-front-matter](https://www.kaggle.com/datasets/jacksondivakarr/online-shopping-dataset?resource=download.), and it includes a detailed record of customer purchase behavior across various transactions; its list of variables is: 

| **Variable**               | **Description**                                               | **Data Type** |
|----------------------------|---------------------------------------------------------------|----------------|
| `CustomerID`                | Unique identifier for each customer.                          | Numeric         |
| `Gender`                    | Gender of the customer (e.g., Male, Female).                   | Categorical     |
| `Location`                  | Location or address information of the customer.               | Text            |
| `Tenure_Months`             | Number of months the customer has been associated.              | Numeric         |
| `Transaction_ID`            | Unique identifier for each transaction.                        | Numeric         |
| `Transaction_Date`          | Date of the transaction.                                        | Date            |
| `Product_SKU`               | Stock Keeping Unit (SKU) identifier for the product.            | Text            |
| `Product_Description`       | Description of the product.                                      | Text            |
| `Product_Category`          | Category to which the product belongs.                          | Categorical     |
| `Quantity`                  | Quantity of the product purchased.                              | Numeric         |
| `Avg_Price`                 | Average price of the product.                                    | Numeric         |
| `Delivery_Charges`          | Charges associated with the delivery of the product.             | Numeric         |
| `Coupon_Status`             | Status of the coupon associated with the transaction.            | Categorical     |
| `GST`                       | Goods and Services Tax associated with the transaction.         | Numeric         |
| `Date`                      | Date of the transaction (potentially redundant with `Transaction_Date`).| Date        |
| `Offline_Spend`             | Amount spent offline by the customer.                            | Numeric         |
| `Online_Spend`              | Amount spent online by the customer.                             | Numeric         |
| `Month`                     | Month of the transaction.                                         | Categorical     |
| `Coupon_Code`               | Code associated with a coupon, if applicable.                     | Text            |
| `Discount_pct`              | Percentage of discount applied to the transaction.                 | Numeric         |

Not all of these variables are particularly relevant to our model, but it felt somewhat relevant to do a little bit of feature engineering. We add the following columns to our data: 
- `Days_Since_Last_Purchase` — Introduced to better track customer engagement cycles.
- `Purchase_Month` — Extracted from `Transaction_Date` to model seasonal trends.
- `Used_Coupon` — Binary feature representing whether a coupon was redeemed.
- `Discount_Applied` — Binary feature representing whether a discount was applied in the transaction.

These features will be useful for the Conditional Probability Tables (CPTs) that will contribute to our models and to our agent's ability to make good predictions and to provide reasonable recommendations for a particular customer. In addition to adding these columns, we rescale some of the columns so that the data are less susceptible to outliers and to increased variability. For some of the relevant features of the data, we include some plots that give us a visual interpretation for our data: 
