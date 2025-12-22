+++
authors = ["Rauzan Sumara"]
title = "Predict Churning Customers"
slug = "predict-churning-customers"
date = "2025-01-12"
tags = [
    "Random Forest",
    "Randomized SearchCV",
]
categories = [
    "Machine Learning",
]
image = "static/post/predict-churning-customers/predicting-customer-churn.png"
+++


# Final Project of Data Mining Course - Big Data Analysis

This is my final project of Data Mining Course, dataset and Python code can be downloaded in [*my Github*](https://github.com/rauzansumara/predict-churning-customers)

A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate if one could predict for them who is gonna get churned so they can proactively go to the customer to provide them better services and turn customers' decisions in the opposite direction

This dataset is from original website with the URL of https://leaps.analyttica.com/home or [Kaggle](https://www.kaggle.com/sakshigoyal7/credit-card-customers). Now, this dataset consists of 10,000 customers mentioning their age, salary, marital_status, credit card limit, credit card category, etc. There are nearly 18 features.

From this data set we can predict the customers who are going to stop using credit cards. Using this model/result, the company can make offer to employess to retain them.

## Attribute Information

* **CLIENTNUM**                : Client number. Unique identifier for the customer holding the account
* **Attrition_Flag**           : Internal event (customer activity) variable - if the account is closed then 1 else 0
* **Customer_Age**             : Customer's Age in Years
* **Gender**                   : M=Male, F=Female
* **Dependent_count**          : Number of dependents
* **Education_Leel**           : Educational Qualification of the account holder (example: high school, college graduate, etc.)
* **Marital_status**           : Married, Single, Divorced, Unknown
* **Income _Category**         : Annual Income Category of the account holder (< $40K, $40K - 60K, $60K - $80K, $80K-$120K, >
* **Card_Category**            : Product Variable - Type of Card (Blue, Silver, Gold, Platinum)
* **Months_On_Book**           : Period of relationship with bank
* **Total_Relationship_Count** : Total no. of products held by the customer
* **Months_Inactive_12_mon**   : No. of months inactive in the last 12 months
* **Contacts_Count_12_mon**    : No. of Contacts in the last 12 months
* **Credit_Limit**             : Credit Limit on the Credit Card                                                                                                           
* **Total_Revolving_Bal**      : Total Revolving Balance on the Credit Card
* **Avg_Open_To_Buy**          : Open to Buy Credit Line (Average of last 12 months)
* **Total_Amt_Chng_Q4_Q1**     : Change in Transaction Amount (Q4 over Q1)
* **Total_Trans_Amt**          : Total Transaction Amount (Last 12 months)
* **Total_Trans_Ct**           : Total Transaction Count (Last 12 months)
* **Total_Ct_Chng_Q4_Q1**      : Change in Transaction Count (Q4 over Q1)
* **Avg_Utilization_Ratio**    : Average Card Utilization Ratio

## Import Dataset
Import Dataset from local computer. Before doing it, we need to import packages as following :


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
plt.style.use('classic')
sns.set()
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
```


```python
pd.set_option('display.max_columns', 6)
df = pd.read_csv('D:/Material Lacture S2/3 Third Semester/Data Mining Methods/Final Project/BankChurners.csv')
df.drop(df.columns[[-1,-2]], axis=1, inplace=True) # Drop 2 last columns
print(df.shape)
df.head(5)
```

    (10127, 21)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CLIENTNUM</th>
      <th>Attrition_Flag</th>
      <th>Customer_Age</th>
      <th>...</th>
      <th>Total_Trans_Ct</th>
      <th>Total_Ct_Chng_Q4_Q1</th>
      <th>Avg_Utilization_Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>768805383</td>
      <td>Existing Customer</td>
      <td>45</td>
      <td>...</td>
      <td>42</td>
      <td>1.625</td>
      <td>0.061</td>
    </tr>
    <tr>
      <th>1</th>
      <td>818770008</td>
      <td>Existing Customer</td>
      <td>49</td>
      <td>...</td>
      <td>33</td>
      <td>3.714</td>
      <td>0.105</td>
    </tr>
    <tr>
      <th>2</th>
      <td>713982108</td>
      <td>Existing Customer</td>
      <td>51</td>
      <td>...</td>
      <td>20</td>
      <td>2.333</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>769911858</td>
      <td>Existing Customer</td>
      <td>40</td>
      <td>...</td>
      <td>20</td>
      <td>2.333</td>
      <td>0.760</td>
    </tr>
    <tr>
      <th>4</th>
      <td>709106358</td>
      <td>Existing Customer</td>
      <td>40</td>
      <td>...</td>
      <td>28</td>
      <td>2.500</td>
      <td>0.000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>



Here we can see that our data consist on **10,127 observation with 21 features**, the features names mantion down below,


```python
# Show all names of features 
df.columns
```




    Index(['CLIENTNUM', 'Attrition_Flag', 'Customer_Age', 'Gender',
           'Dependent_count', 'Education_Level', 'Marital_Status',
           'Income_Category', 'Card_Category', 'Months_on_book',
           'Total_Relationship_Count', 'Months_Inactive_12_mon',
           'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
           'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
           'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'],
          dtype='object')



## Exploratory Data Analysis
before modeling data, we would like to explore and vizualize them so that we can understand what kinds of data we have.


```python
print(df.dtypes)
```

    CLIENTNUM                     int64
    Attrition_Flag               object
    Customer_Age                  int64
    Gender                       object
    Dependent_count               int64
    Education_Level              object
    Marital_Status               object
    Income_Category              object
    Card_Category                object
    Months_on_book                int64
    Total_Relationship_Count      int64
    Months_Inactive_12_mon        int64
    Contacts_Count_12_mon         int64
    Credit_Limit                float64
    Total_Revolving_Bal           int64
    Avg_Open_To_Buy             float64
    Total_Amt_Chng_Q4_Q1        float64
    Total_Trans_Amt               int64
    Total_Trans_Ct                int64
    Total_Ct_Chng_Q4_Q1         float64
    Avg_Utilization_Ratio       float64
    dtype: object
    

as you can see that types of features are such as **Object, Integer and Float**. First of all, we would like to vizualize features with object types. Down there are code for making pie chart and bar chart in order to get  insight from **Attrition_Flag, Gender, Education_Level, Marital_Status, Income_Category and Card_Category.**  


```python
fig = plt.figure(constrained_layout=False, figsize=(17, 20))
spec = gridspec.GridSpec(ncols=2, nrows=3, figure = fig)
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[0, 1])
ax3 = fig.add_subplot(spec[1, 0])
ax4 = fig.add_subplot(spec[1, 1])
ax5 = fig.add_subplot(spec[2, 0])
ax6 = fig.add_subplot(spec[2, 1])

# Attrition_Flag
labels = df['Attrition_Flag'].value_counts().keys()
ax1.pie(df['Attrition_Flag'].value_counts(),labels = labels,  autopct='%.1f%%',
        shadow=True, wedgeprops={'edgecolor': 'black'})
ax1.set_title('Proportion of Attrition_Flag')

# Gender
labels = df['Gender'].value_counts().keys()
ax2.pie(df['Gender'].value_counts(),labels = labels,  autopct='%.1f%%',
        shadow=True, wedgeprops={'edgecolor': 'black'})
ax2.set_title('Proportion of Gender')

# Education_Level
sns.countplot(ax=ax3, x=df['Education_Level'])
ax3.set_title('Education_Level of Customers')

# Marital_Status 
sns.countplot(ax=ax4, x=df['Marital_Status'])
ax4.set_title('Marital_Status of Customers')

# Income_Category 
sns.countplot(ax=ax5, x=df['Income_Category'])
ax5.set_title('Income_Category of Customers')              

# Card_Category                 
labels = df['Card_Category'].value_counts().keys()
ax6.pie(df['Card_Category'].value_counts(),labels = labels,  autopct='%.1f%%',
        shadow=True, wedgeprops={'edgecolor': 'black'})
ax6.set_title('Proportion of Card_Category')
```




    Text(0.5, 1.0, 'Proportion of Card_Category')




    
![svg](/post/predict-churning-customers/output_11_1.svg)
    


based on charts above, we got proportion of Attrition_Flag 83.9% existing and 16.1% attrited customer. If we look from Gender, most of the customer are female 52.0%. They were also having Graduate and got merried in majority. Thus, according to income category, the customer has mostly less than 40,000 USD in a year. We can see also that more than 93% they hold blue card.


```python
df.select_dtypes(include='float64').describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Credit_Limit</th>
      <th>Avg_Open_To_Buy</th>
      <th>Total_Amt_Chng_Q4_Q1</th>
      <th>Total_Ct_Chng_Q4_Q1</th>
      <th>Avg_Utilization_Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>10127.000000</td>
      <td>10127.000000</td>
      <td>10127.000000</td>
      <td>10127.000000</td>
      <td>10127.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8631.953698</td>
      <td>7469.139637</td>
      <td>0.759941</td>
      <td>0.712222</td>
      <td>0.274894</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9088.776650</td>
      <td>9090.685324</td>
      <td>0.219207</td>
      <td>0.238086</td>
      <td>0.275691</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1438.300000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2555.000000</td>
      <td>1324.500000</td>
      <td>0.631000</td>
      <td>0.582000</td>
      <td>0.023000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4549.000000</td>
      <td>3474.000000</td>
      <td>0.736000</td>
      <td>0.702000</td>
      <td>0.176000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>11067.500000</td>
      <td>9859.000000</td>
      <td>0.859000</td>
      <td>0.818000</td>
      <td>0.503000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>34516.000000</td>
      <td>34516.000000</td>
      <td>3.397000</td>
      <td>3.714000</td>
      <td>0.999000</td>
    </tr>
  </tbody>
</table>
</div>



here we consider to calculate summary statistics towards numeric features such as **Credit_Limit, Avg_Open_To_Buy, Total_Amt_Chng_Q4_Q1, Total_Ct_Chng_Q4_Q1 and Avg_Utilization_Ratio**. We can interpret Credit_Limit as that feature has mean 8631.953698 less than standar deviation 9088.776650, which mean that the data has high volatility and it happens to Avg_Open_To_Buy as well.


```python
df.groupby('Attrition_Flag')[df.select_dtypes(include='float64').columns].describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Attrition_Flag</th>
      <th>Attrited Customer</th>
      <th>Existing Customer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="8" valign="top">Credit_Limit</th>
      <th>count</th>
      <td>1627.000000</td>
      <td>8500.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8136.039459</td>
      <td>8726.877518</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9095.334105</td>
      <td>9084.969807</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1438.300000</td>
      <td>1438.300000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2114.000000</td>
      <td>2602.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4178.000000</td>
      <td>4643.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9933.500000</td>
      <td>11252.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>34516.000000</td>
      <td>34516.000000</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Avg_Open_To_Buy</th>
      <th>count</th>
      <td>1627.000000</td>
      <td>8500.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>7463.216472</td>
      <td>7470.273400</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9109.208129</td>
      <td>9087.671862</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.000000</td>
      <td>15.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1587.000000</td>
      <td>1184.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3488.000000</td>
      <td>3469.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>9257.500000</td>
      <td>9978.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>34516.000000</td>
      <td>34516.000000</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Total_Amt_Chng_Q4_Q1</th>
      <th>count</th>
      <td>1627.000000</td>
      <td>8500.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.694277</td>
      <td>0.772510</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.214924</td>
      <td>0.217783</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.256000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.544500</td>
      <td>0.643000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.701000</td>
      <td>0.743000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.856000</td>
      <td>0.860000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.492000</td>
      <td>3.397000</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Total_Ct_Chng_Q4_Q1</th>
      <th>count</th>
      <td>1627.000000</td>
      <td>8500.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.554386</td>
      <td>0.742434</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.226854</td>
      <td>0.228054</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.028000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.400000</td>
      <td>0.617000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.531000</td>
      <td>0.721000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.692000</td>
      <td>0.833000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.500000</td>
      <td>3.714000</td>
    </tr>
    <tr>
      <th rowspan="8" valign="top">Avg_Utilization_Ratio</th>
      <th>count</th>
      <td>1627.000000</td>
      <td>8500.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.162475</td>
      <td>0.296412</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.264458</td>
      <td>0.272568</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.055000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.211000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.231000</td>
      <td>0.529250</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.999000</td>
      <td>0.994000</td>
    </tr>
  </tbody>
</table>
</div>



We may compare between Attrited Customer and Existing Customer based on Credit_Limit, Avg_Open_To_Buy, Total_Amt_Chng_Q4_Q1, Total_Ct_Chng_Q4_Q1 and Avg_Utilization_Ratio. As you can see above that those two categories have no much difference coresponding to mean and standar deviation. for example, mean of Attrited Customer is 7463.216472 (std: 9109.208129) and mean of Attrited Customer 7470.273400 (std: 9087.671862), and so are other features.


```python
fig = plt.figure(figsize=(17, 20))
spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[0, 1])
ax3 = fig.add_subplot(spec[1, 0])
ax4 = fig.add_subplot(spec[1, 1])
ax5 = fig.add_subplot(spec[2, 0])

df.boxplot(column=['Credit_Limit'], by=['Income_Category'], ax=ax1)
df.boxplot(column=['Avg_Open_To_Buy'], by=['Income_Category'], ax=ax2)
df.boxplot(column=['Total_Amt_Chng_Q4_Q1'], by=['Income_Category'], ax=ax3)
df.boxplot(column=['Total_Ct_Chng_Q4_Q1'], by=['Income_Category'], ax=ax4)
df.boxplot(column=['Avg_Utilization_Ratio'], by=['Income_Category'], ax=ax5)
```




    <AxesSubplot:title={'center':'Avg_Utilization_Ratio'}, xlabel='[Income_Category]'>




    
![svg](/post/predict-churning-customers/output_17_1.svg)
    


if we are grouping by Income_Category using boxplot, we can recognize that a lot of data points are above and below of the mean value there. let say we only concert to one feature which is Credit_Limit for example. according to **"less than 40K USD"** and **"40K - 60K"** categories, many customers have too high credit comparing to its average. The company should consider some rule for customers who want to propose the credit so that use of credit card by customers might be optimum.


```python
# Categorizing of Customer_Age into 4 categories
df['Customer_Age_Categorized'] = pd.cut(df['Customer_Age'], bins=4, precision=0) 
plt.figure(figsize=(15,8))
sns.countplot(y='Card_Category', hue='Customer_Age_Categorized', data = df)
plt.legend(loc = 'center right')
```




    <matplotlib.legend.Legend at 0x1b2a087a430>




    
![svg](/post/predict-churning-customers/output_19_1.svg)
    


Graph above says that most of customers are using blue card, and Age between 38 - 50 year old is dominant at every categories.


```python
df.select_dtypes(include='int64').columns
```




    Index(['CLIENTNUM', 'Customer_Age', 'Dependent_count', 'Months_on_book',
           'Total_Relationship_Count', 'Months_Inactive_12_mon',
           'Contacts_Count_12_mon', 'Total_Revolving_Bal', 'Total_Trans_Amt',
           'Total_Trans_Ct'],
          dtype='object')



Now, we also want to get some insight from count (Integer) features, such as **Customer_Age, Dependent_count,Months_on_book, Total_Relationship_Count, Months_Inactive_12_mon, Contacts_Count_12_mon, Total_Revolving_Bal, Total_Trans_Amt, and Total_Trans_Ct**


```python
df.groupby(['Attrition_Flag','Income_Category'])['Total_Relationship_Count','Total_Revolving_Bal',                            'Total_Trans_Amt','Total_Trans_Ct'].sum()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>Total_Relationship_Count</th>
      <th>Total_Revolving_Bal</th>
      <th>Total_Trans_Amt</th>
      <th>Total_Trans_Ct</th>
    </tr>
    <tr>
      <th>Attrition_Flag</th>
      <th>Income_Category</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="6" valign="top">Attrited Customer</th>
      <th>$120K +</th>
      <td>409</td>
      <td>85407</td>
      <td>427527</td>
      <td>5752</td>
    </tr>
    <tr>
      <th>$40K - $60K</th>
      <td>878</td>
      <td>172016</td>
      <td>811029</td>
      <td>12090</td>
    </tr>
    <tr>
      <th>$60K - $80K</th>
      <td>613</td>
      <td>107579</td>
      <td>613566</td>
      <td>8474</td>
    </tr>
    <tr>
      <th>$80K - $120K</th>
      <td>705</td>
      <td>183586</td>
      <td>905229</td>
      <td>11488</td>
    </tr>
    <tr>
      <th>Less than $40K</th>
      <td>2123</td>
      <td>419902</td>
      <td>1753529</td>
      <td>27048</td>
    </tr>
    <tr>
      <th>Unknown</th>
      <td>608</td>
      <td>126193</td>
      <td>524727</td>
      <td>8255</td>
    </tr>
    <tr>
      <th rowspan="6" valign="top">Existing Customer</th>
      <th>$120K +</th>
      <td>2338</td>
      <td>803130</td>
      <td>2865396</td>
      <td>40561</td>
    </tr>
    <tr>
      <th>$40K - $60K</th>
      <td>5894</td>
      <td>1925854</td>
      <td>7075029</td>
      <td>104261</td>
    </tr>
    <tr>
      <th>$60K - $80K</th>
      <td>4762</td>
      <td>1511722</td>
      <td>5626333</td>
      <td>80199</td>
    </tr>
    <tr>
      <th>$80K - $120K</th>
      <td>5154</td>
      <td>1668740</td>
      <td>5976115</td>
      <td>84751</td>
    </tr>
    <tr>
      <th>Less than $40K</th>
      <td>11492</td>
      <td>3657930</td>
      <td>13784610</td>
      <td>208529</td>
    </tr>
    <tr>
      <th>Unknown</th>
      <td>3634</td>
      <td>1113759</td>
      <td>4237092</td>
      <td>65416</td>
    </tr>
  </tbody>
</table>
</div>



Attrited Customer with **incoming less than 40k usd**, have big number of **Total_Relationship_Count, Total_Revolving_Bal, Total_Trans_Amt, and Total_Trans_Ct**. It does also happen in existing Customer.


```python
plt.figure(figsize=(15,8))
sns.countplot(y='Attrition_Flag', hue='Customer_Age_Categorized', data = df)
plt.legend(loc = 'center right')
```




    <matplotlib.legend.Legend at 0x1b2a0f0cbb0>




    
![svg](/post/predict-churning-customers/output_25_1.svg)
    



```python
# Bar chart Income_Category groupby Attrition_Flag 
plt.figure(figsize=(13,8))
sns.countplot(y='Attrition_Flag', hue='Income_Category', data = df)
plt.legend(loc = 'center right')
plt.xlabel('Number of Customers')
plt.ylabel('Attrition_Flag')
plt.title('Income_Category groupby Attrition_Flag')

# Table Income_Category groupby Attrition_Flag 
df.groupby('Attrition_Flag')['Income_Category'].value_counts()
```




    Attrition_Flag     Income_Category
    Attrited Customer  Less than $40K      612
                       $40K - $60K         271
                       $80K - $120K        242
                       $60K - $80K         189
                       Unknown             187
                       $120K +             126
    Existing Customer  Less than $40K     2949
                       $40K - $60K        1519
                       $80K - $120K       1293
                       $60K - $80K        1213
                       Unknown             925
                       $120K +             601
    Name: Income_Category, dtype: int64




    
![png](output_26_1.png)
    



```python
# Bar chart Income_Category groupby Attrition_Flag 
plt.figure(figsize=(13,8))
sns.countplot(y='Income_Category', hue='Customer_Age_Categorized', data = df)
plt.legend(loc = 'center right')
plt.xlabel('Number of Customers')
plt.ylabel('Attrition_Flag')
plt.title('Income_Category groupby Attrition_Flag')

# Table Income_Category groupby Attrition_Flag 
df.groupby('Customer_Age_Categorized')['Income_Category'].value_counts()
```




    Customer_Age_Categorized  Income_Category
    (26.0, 38.0]              Less than $40K      537
                              $40K - $60K         275
                              Unknown             179
                              $60K - $80K         178
                              $80K - $120K        161
                              $120K +              70
    (38.0, 50.0]              Less than $40K     1768
                              $40K - $60K         943
                              $80K - $120K        833
                              $60K - $80K         805
                              Unknown             542
                              $120K +             306
    (50.0, 61.0]              Less than $40K     1109
                              $80K - $120K        531
                              $40K - $60K         498
                              $60K - $80K         388
                              $120K +             350
                              Unknown             342
    (61.0, 73.0]              Less than $40K      147
                              $40K - $60K          74
                              Unknown              49
                              $60K - $80K          31
                              $80K - $120K         10
                              $120K +               1
    Name: Income_Category, dtype: int64




    
![svg](/post/predict-churning-customers/output_27_1.svg)
    


Based on some graphs above, we can conclude that most customers have age **between 38 - 50 year old**. It also imply that many of them are having **income less than 40k usd**. 

## Data Preprocessing



```python
df['Attrition_Flag'].value_counts().keys()
```




    Index(['Existing Customer', 'Attrited Customer'], dtype='object')



Before applying mechine learning model in order to clasify **Attrition_Flag (Existing Customer :0, Attrited Customer: 1)**, we are going to prepare the data (cleaning data) so that it will be easy to use into algorithm.


```python
# Remove 'Unknown' Observation and 'CLIENTNUM' column  
df.replace({'Unknown': np.nan}, inplace=True)
df.dropna(inplace=True) # Remove 'Unknown' observation
df.drop(['CLIENTNUM'], axis=1, inplace=True) # Delete 'CLIENTNUM' column
print(df.shape)
df.head(5)
```

    (7081, 21)
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Attrition_Flag</th>
      <th>Customer_Age</th>
      <th>Gender</th>
      <th>...</th>
      <th>Total_Ct_Chng_Q4_Q1</th>
      <th>Avg_Utilization_Ratio</th>
      <th>Customer_Age_Categorized</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Existing Customer</td>
      <td>45</td>
      <td>M</td>
      <td>...</td>
      <td>1.625</td>
      <td>0.061</td>
      <td>(38.0, 50.0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Existing Customer</td>
      <td>49</td>
      <td>F</td>
      <td>...</td>
      <td>3.714</td>
      <td>0.105</td>
      <td>(38.0, 50.0]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Existing Customer</td>
      <td>51</td>
      <td>M</td>
      <td>...</td>
      <td>2.333</td>
      <td>0.000</td>
      <td>(50.0, 61.0]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Existing Customer</td>
      <td>40</td>
      <td>M</td>
      <td>...</td>
      <td>2.500</td>
      <td>0.000</td>
      <td>(38.0, 50.0]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Existing Customer</td>
      <td>44</td>
      <td>M</td>
      <td>...</td>
      <td>0.846</td>
      <td>0.311</td>
      <td>(38.0, 50.0]</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
# Change categorical variables into dummy variables
df2 = pd.concat([df['Attrition_Flag'].replace({'Existing Customer': 0, 'Attrited Customer': 1}),
                df['Gender'].replace({'M': 0, 'F':1}),
                pd.get_dummies(df['Education_Level']), 
                pd.get_dummies(df['Marital_Status']), 
                pd.get_dummies(df['Income_Category']), 
                pd.get_dummies(df['Card_Category']),
                df.select_dtypes(include=['int64','float64'])], axis=1)

df2.drop(['Uneducated','Divorced','$120K +','Platinum'], axis=1, inplace=True) # Delete because of base categories
```

As we can see that some of our features are categorical variables, therefore we have to make dummy variables instead of categorical features.


```python
df2.columns
```




    Index(['Attrition_Flag', 'Gender', 'College', 'Doctorate', 'Graduate',
           'High School', 'Post-Graduate', 'Married', 'Single', '$40K - $60K',
           '$60K - $80K', '$80K - $120K', 'Less than $40K', 'Blue', 'Gold',
           'Silver', 'Customer_Age', 'Dependent_count', 'Months_on_book',
           'Total_Relationship_Count', 'Months_Inactive_12_mon',
           'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
           'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
           'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'],
          dtype='object')



Here is final dataset, which is **'Attrition_Flag'** as output, and the others are input variables

## Data Modeling 1: Random Forest (Default Hyperparameter)

In clasification problem, many machine learning method are used, one of them is Random forest. This algorithm is a supervised learning algorithm. The "forest" it builds, is an ensemble of decision trees, usually trained with the “bagging” method. The general idea of the bagging method is that a combination of learning models increases the overall result. **Why do we use this algorithm?** bacause Random forest is flexible, easy to use machine learning algorithm that produces a great result most of the time. It is also one of the most used algorithms, because of its simplicity and diversity (it can be used for both classification and regression tasks).


```python
# Spliting dataset into training and testing
x = df2.drop(['Attrition_Flag'], axis=1)
y = df2['Attrition_Flag']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("Proportion of training set :")
print(y_train.value_counts())
print("Proportion of testing set :")
print(y_test.value_counts())
```

    Proportion of training set :
    0    4763
    1     901
    Name: Attrition_Flag, dtype: int64
    Proportion of testing set :
    0    1205
    1     212
    Name: Attrition_Flag, dtype: int64
    


```python
model1 = RandomForestClassifier(random_state= 0)
model1.fit(x_train, y_train)
yhat = model1.predict(x_test)
plt.figure(figsize=(5, 4))
cf_mat = confusion_matrix(y_test, yhat)
sns.heatmap(cf_mat, annot=True, fmt='g')
plt.show()

print('Random Forest Classifier :\n\n\t',
     f'The Training model accuracy :{model1.score(x_train, y_train)}\n\t', 
     f'The Test model accuracy: {model1.score(x_test, y_test)}')
print(classification_report(y_test, yhat))
```


    
![svg](/post/predict-churning-customers/output_40_0.svg)
    


    Random Forest Classifier :
    
    	 The Training model accuracy :1.0
    	 The Test model accuracy: 0.958362738179252
                  precision    recall  f1-score   support
    
               0       0.96      0.99      0.98      1205
               1       0.94      0.77      0.85       212
    
        accuracy                           0.96      1417
       macro avg       0.95      0.88      0.91      1417
    weighted avg       0.96      0.96      0.96      1417
    
    

by using hyperparameter from python which are **criterion ='gini', n_estimators=100, max_depth=None, max_features = 'auto', min_samples_leaf = 1,** and **min_samples_split = 2 default from python**, we got accuracy **95.8%**, sensitivity/recall **77.0%**, and precision **94.0%**. it's quite good so far. But to be honest, we can improve accuracy by doing some treatments, here we try to increase accuracy in choosing hyperparameter by **Randomized SearchCV**.

## Data Modeling 2: Improving Random Forest (Randomized SearchCV)

Randomized search is the most widely used strategies for hyper-parameter optimization. Many papers showed empirically and theoretically that randomly chosen trials are more efficient for hyper-parameter optimization than trials on a grid. In Random Search, we create a grid of hyperparameters and train/test our model on just some random combination of these hyperparameters. In this example, I additionally decided to perform Cross-Validation on the training set.

We can now start implementing Random Search by first defying a grid of hyperparameters which will be randomly sampled when calling RandomizedSearchCV(). For this example, I decided to divide our training set into 5 Folds (cv = 5) and select 20 as the number of combinations to sample (n_iter = 20). Using the scikit-learn best_estimator_ attribute, we can then retrieve the set of hyperparameters which performed best during training to test our model.


```python
random_search = {'criterion': ['entropy', 'gini'],
               'max_depth': list(np.linspace(10, 1200, 10, dtype = int)) + [None],
               'max_features': ['auto', 'sqrt','log2', None],
               'min_samples_leaf': [4, 5, 6, 7, 8, 9, 10, 11, 12],
               'min_samples_split': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
               'n_estimators': list(np.linspace(100, 1200, 5, dtype = int))}

clf = RandomForestClassifier()
model2 = RandomizedSearchCV(estimator = clf, param_distributions = random_search, n_iter = 20, 
                            cv = 5, verbose= 5, n_jobs = -1)
model2.fit(x_train,y_train)
```

    Fitting 5 folds for each of 20 candidates, totalling 100 fits
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 16 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  40 tasks      | elapsed:   17.9s
    [Parallel(n_jobs=-1)]: Done  90 out of 100 | elapsed:   44.1s remaining:    4.8s
    [Parallel(n_jobs=-1)]: Done 100 out of 100 | elapsed:   52.5s finished
    




    RandomizedSearchCV(cv=5, estimator=RandomForestClassifier(), n_iter=20,
                       n_jobs=-1,
                       param_distributions={'criterion': ['entropy', 'gini'],
                                            'max_depth': [10, 142, 274, 406, 538,
                                                          671, 803, 935, 1067, 1200,
                                                          None],
                                            'max_features': ['auto', 'sqrt', 'log2',
                                                             None],
                                            'min_samples_leaf': [4, 5, 6, 7, 8, 9,
                                                                 10, 11, 12],
                                            'min_samples_split': [3, 4, 5, 6, 7, 8,
                                                                  9, 10, 11, 12, 13,
                                                                  14],
                                            'n_estimators': [100, 375, 650, 925,
                                                             1200]},
                       verbose=5)




```python
table = pd.pivot_table(pd.DataFrame(model2.cv_results_), values='mean_test_score', index='param_n_estimators', 
                       columns='param_criterion')
sns.heatmap(table)
print(model2.best_estimator_)
```

    RandomForestClassifier(criterion='entropy', max_depth=1067, max_features=None,
                           min_samples_leaf=4, min_samples_split=8,
                           n_estimators=650)
    


    
![svg](/post/predict-churning-customers/output_45_1.svg)
    


According to Randomized SearchCV above, we got the best hyperparameter are **criterion='entropy', n_estimators=650, max_depth=1067, max_features = None, min_samples_leaf = 4,** and **min_samples_split = 8**.


```python
yhat2 = model2.best_estimator_.predict(x_test)
plt.figure(figsize=(5, 4))
cf_mat2 = confusion_matrix(y_test, yhat2)
sns.heatmap(cf_mat2, annot=True, fmt='g')
plt.show()

print('Random Forest Classifier 2:\n\n\t',  
     f'The Test model accuracy: {model2.best_estimator_.score(x_test, y_test)}')
print(classification_report(y_test, yhat2))
```


    
![svg](/post/predict-churning-customers/output_47_0.svg)
    


    Random Forest Classifier 2:
    
    	 The Test model accuracy: 0.9611856033874383
                  precision    recall  f1-score   support
    
               0       0.97      0.98      0.98      1205
               1       0.89      0.84      0.87       212
    
        accuracy                           0.96      1417
       macro avg       0.93      0.91      0.92      1417
    weighted avg       0.96      0.96      0.96      1417
    
    

By using the best hyperparameter from Randomized SearchCV, we got accuracy **96.1%**, sensitivity/recall **84.0%**, and precision **89.0%**. This **(model 2)** is bit better than previous one **(model 1)**.

## Comparison of ROC Curve of Both Models 


```python
# predict probabilities
pred_prob1 = model1.predict_proba(x_test)
pred_prob2 = model2.predict_proba(x_test)

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,1], pos_label=1)

# auc scores
auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
auc_score2 = roc_auc_score(y_test, pred_prob2[:,1])

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Model 1 (area = %0.2f)' % auc_score1)
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='Model 2 (area = %0.2f)' % auc_score2)

# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.show()
```


    
![svg](/post/predict-churning-customers/output_50_0.svg)
    


As we can see that ROC curves of both models are not significantly different. We also got AUC for **model 1 = 0.98** and **model 2 = 0.99**. Even it's so, we still believe that hyperparameters have effected big enough to the accuracy of the model in circumstance situation.

## Feature Importance


```python
sorted_idx = model1.feature_importances_.argsort()
plt.barh(x_train.columns[sorted_idx], model1.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
```




    Text(0.5, 0, 'Random Forest Feature Importance')




    
![svg](/post/predict-churning-customers/output_53_1.svg)
    


According to the graph, we can get that 5 most dominant/important features are **Total_Trans_Amt, Total_Trans_Ct, Total_Revolving_Bal, Total_Ct_Chng_Q4_Q1** and **Avg_Utilization_Ratio**. Therefore the company can put effort more through these features.
