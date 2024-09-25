import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from prefect import flow, task


# load the data set to workflow
@task
def load_data():
    return pd.read_csv('Ecommerce Customers')

## Preprocessing Analysis
@task(log_prints=True)
def preprocess(df):
    df = df.drop(['Email','Address','Avatar'], axis = 1)
    #df = df.drop('Address')
    #df = df.drop('Avatar')
    missing_values = df.isna().sum()
    columns_with_missing = missing_values[missing_values > 0]
    print("Columns with missing values: ")
    print(columns_with_missing)

    # Replace with Median value
    df.fillna(df.median(), inplace=True)
    return df

## Checking correlation Analysis
@task
def corr_analysis(df):
    #plt.figure(figsize=(6,6))
    sns.lmplot(x='Length of Membership', 
           y='Yearly Amount Spent', 
           data=df,
           scatter_kws={'alpha':0.3})
    
    plt.title('correlation analysis')
    plt.show()

## Model Creation -- Linear Regression.
@task(log_prints=True)
def train_model(df):
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import math
    import pylab 
    import scipy.stats as stats
    X = df[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
    y = df['Yearly Amount Spent']
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)
    model = LinearRegression()
    model.fit(X_train,y_train)
    
    ## print the rsquared.
    print("Model_score:",model.score(X, y))
    #print(model.coef_)
    cdf = pd.DataFrame(model.coef_,X.columns,columns=['Coef'])
    print("Model Coefficients:")
    print(cdf)    


    ## Evaluate the model.
    predictions = model.predict(X_test)

    print('Mean Absolute Error:',mean_absolute_error(y_test, predictions))
    print('Mean Squared Error:',mean_squared_error(y_test, predictions))
    print('Root Mean Squared Error:',math.sqrt(mean_squared_error(y_test, predictions)))

    #### Residuals :
    residuals = y_test-predictions
    sns.distplot(residuals, bins=30)
    plt.title('Residuals')
    plt.show()

    stats.probplot(residuals, dist="norm", plot=pylab)
    pylab.show()


##Prefect workflow
@flow(log_prints=True)
def cust_flow():
    customers = load_data()
    print(customers.describe())
    preprocess_data = preprocess(customers)
    print(preprocess_data.head())
    corr_analysis(preprocess_data)
    train_model(preprocess_data)


if __name__ == "__main__":

    #commented below 4 lines to do deplyment for 120 seconds on prefect cloud

    # cust_flow.serve(name = "Customers-WorkFlow",
    #                 tags=["demo-flow"],
    #                 parameters={},
    #                 interval = 120)

    # below line to execute one time the flow in prefect cloud
    cust_flow()