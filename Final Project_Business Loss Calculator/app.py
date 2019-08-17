from flask import Flask, render_template, request
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression

from pip._internal import main as pipmain

pipmain(['install', 'pandas_redshift'])
import pandas_redshift as pr

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/exploratory_analysis')
def exploratory_analysis():
    images = [join("eda/", f) for f in listdir("./static/eda")]
   
    return render_template('exploratory_analysis.html', images=images)

@app.route('/business_loss', methods=["GET", "POST"])
def business_loss():
    images = [join("customer_lifetime_value/", f) for f in listdir("./static/customer_lifetime_value")]

    
    db_name = "info7374dbassignment2"#-------------------------------------Redshift: Database Name for gaming data
    
    master_username = "root"#----------------------------------------Redshift: Admin Username
    master_password = "Info7374gap"#---------------------------------Redshift: Admin Password
    
    
    hostname = "info7374clusterproject.cwtvmzfhaqaf.us-east-1.redshift.amazonaws.com" #----------------Redshift: Hostname for database
    port_number = 5439    #----------------Redshift: Port Number for databse
    
    pr.connect_to_redshift(dbname = db_name ,
                            host = hostname,
                            port = port_number,
                            user = master_username,
                            password =master_password)
    
    data = pr.redshift_to_pandas('select * from sales')
    
    data=data.drop_duplicates()
    
    data= data[pd.notnull(data['customerid'])]
    
    data = data[(data['quantity']>0)]
    
    #most bought product
    data['description'].value_counts()[:10]
    
    #which customer bought the most number of items?
    cust_data=pd.DataFrame()
    cust_data['customerid']=list(set(data['customerid']))
    cust_data=cust_data.set_index('customerid')
    for cust_id in cust_data.index:
        cust_data.at[cust_id,'Number_of_items']=(len(data[data['customerid']==cust_id]['description']))
    cust_data=cust_data.sort_values('Number_of_items',ascending=False)
    
    #	stockcode	description					states
    data=data[['customerid','invoicedate','invoiceno','quantity','unitprice']]
    #Calulate total purchase
    data['TotalPurchase'] = data['quantity'] * data['unitprice']
    
    data_group=data.groupby('customerid').agg({'invoicedate': lambda date: (date.max() - date.min()).days,
                                                     'invoiceno': lambda num: len(num),
                                                     'quantity': lambda quant: quant.sum(),
                                                     'TotalPurchase': lambda price: price.sum()})
    
    # Change the name of columns
    data_group.columns=['num_days','num_transactions','num_units','spent_money']
    data_group.head()
    
    # Average Order Value
    data_group['avg_order_value']=data_group['spent_money']/data_group['num_transactions']
    
    purchase_frequency=sum(data_group['num_transactions'])/data_group.shape[0]
    
    # Repeat Rate
    repeat_rate=data_group[data_group.num_transactions > 1].shape[0]/data_group.shape[0]
    #Churn Rate
    churn_rate=1-repeat_rate
    
    purchase_frequency,repeat_rate,churn_rate
    
    # Profit Margin
    data_group['profit_margin']=data_group['spent_money'].astype('float') *0.05
    
    # Customer Value
    data_group['CLV']=(data_group['avg_order_value'].astype('float')*purchase_frequency)/churn_rate
    #Customer Lifetime Value
    data_group['cust_lifetime_value']=data_group['CLV'].astype('float')*data_group['profit_margin'].astype('float')
    data_group.head()
    
    clv=data_group.loc[:,"cust_lifetime_value"].mean()/1000000
    
    # drop the row missing customerID
    data = data[data.customerid.notnull()]
    
    # extract year, month and day 
    data['invoiceday'] = data.invoicedate.apply(lambda x: dt.datetime(x.year, x.month, x.day))
    data.head()
    
    monthly_unique_customers_df = data.set_index('invoiceday')['customerid'].resample('M').nunique()
    
    pd.DataFrame(monthly_unique_customers_df)['invoicedate']=pd.DataFrame(monthly_unique_customers_df).index
    
    df = pd.DataFrame(monthly_unique_customers_df).reset_index()
    
    Customer_count=df.loc[:,"customerid"].mean()
    
    df["CustomerIDshift"] = [0]+list(df["customerid"][:-1])
    
    df["ChurnRate"] = (df["CustomerIDshift"]-df["customerid"])/df["CustomerIDshift"]
    
    df.rename(columns={'invoiceday': 'Month'}, inplace=True)
    
    df['ChurnRate'][0]=1
    
    data = df.drop(columns=['customerid','CustomerIDshift'])
    
    table1=data
    
    table1
    
    from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
    from statsmodels.tsa.arima_model import ARIMA
    
    data = data.set_index('Month')
    data.index
    
    model = ARIMA(data, order=(2,1,0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())
    # plot residual errors
    # residuals = pd.DataFrame(model_fit.resid)
    # residuals.plot()
    # plt.show()
    # residuals.plot(kind='kde')
    # plt.show()
    
    X=data.values
    history = [x for x in X]
    
    test=['2019-01-31','2019-02-28','2019-03-31','2019-04-30','2019-05-31','2019-06-31']
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=(2,1,0))
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        history.append(yhat)
        predictions.append(yhat)
        print('predicted=%f'%(yhat))
    
    print(predictions)
    
    i=0
    yes_array=[]
    for value in predictions:
      print(predictions[i])
      yes_array.append(predictions[i])
      i+=1
    
    df_toplot=pd.DataFrame({
        "ChurnRate" : yes_array,
        "Month" : test
        
    })
    
    df_toplot["Business_Loss"] = df["ChurnRate"]*clv*Customer_count
    
    x=df_toplot["Business_Loss"].astype(int)
    
    
    df_toplot['Business Loss']=x
    
    
    final_df = df_toplot
    
    del final_df['Business_Loss']
    
    table2 = final_df
    table1 = table1.to_html(classes="data")
    table2 = table2.to_html(classes="data")

    return render_template('business_loss.html', tables=[table1, table2], 
                           titles=["Blah", "Churn Rate", "Future Churn Rate"], images=images, clv=clv)

@app.route('/customer_analysis')
def customer_segmentation():
    data = pd.read_csv("./Lost_Customers.csv").head()
    table1 = data.to_html(classes='data')
    table2 = pd.read_csv("./Lostcheap_Customers.csv").head().to_html(classes='data')
    table3 = pd.read_csv("./Best_Customers.csv").head().to_html(classes='data')

    images = [join("customer_analysis/", f) for f in listdir("./static/customer_analysis")]
    titles = ["Blah", "Lost Customers","Bad Customers", "Best Customers"]   
 
    return render_template('customer_analysis.html', images=images, tables=[table1, table2, table3],
                           titles=titles)

@app.route('/customer_conversion', methods=["GET", "POST"])
def customer_conversion():
    if request.method == "POST":

        data_file = r'./FinalLead.csv'
        data = pd.read_csv(data_file)
        data.head()
        data.corr()
        data_data=data.ix[:,(2,4,5,6,7)].values
        y=data.ix[:,(8)].values
        x=scale(data_data)
        LogReg=LogisticRegression()
        LogReg.fit(x,y)
        
        csv_file = request.files["data_file"]
        new_data = pd.read_csv(csv_file)
        new_data_ch=new_data.ix[:,(2,4,5,6,7)].values
        
        c=scale(new_data_ch)
        e=LogReg.predict_proba(c)
        i=0
        yes_array=[]
        for value in e:
          print(e[i,0])
          yes_array.append(e[i,0])
          i+=1
        
        df_final=pd.DataFrame({
            "Pobability of Customer" : yes_array,
            "Customer" : new_data['Customername']
            
        })
       
        df = df_final

        #  df should be the final data frame that you want to display
        data =  df.to_html(classes="data")
        titles = ["Blah", "Lead Scoring"]
        return render_template('customer_conversion_calculated.html', tables=[data], titles=titles)
    else:
        return render_template('customer_conversion.html')

if __name__ == '__main__':
    app.run(debug=True)
