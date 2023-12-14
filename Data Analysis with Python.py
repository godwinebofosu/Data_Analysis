#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
df = pd.read_csv(url, header = None )


# In[2]:


import numpy as np


# In[3]:


df.head()


# In[4]:


headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n", headers)


# In[5]:


df.columns = headers
df.head(10)


# In[6]:


#replacing unknown values 
df1=df.replace('?', np.NaN)


# In[7]:


df=df1.dropna(subset=["price"], axis=0)
df.head(20)


# In[8]:


#saving the data
df.to_csv("automobile.csv", index=False)


# In[9]:


#getting the types of each column
df.dtypes


# In[10]:


#a statistical summary of each column
df.describe()


# In[11]:


df.describe(include = "all")


# In[12]:


#checking dataset
df.info()


# In[13]:


df.replace("?", np.nan, inplace = True)
df.head()


# In[14]:


#identifying missing number with output of a boolean
missing_data = df.isnull()
missing_data.head(5)


# In[15]:


for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")


# In[16]:


avg_norm_loss = df["normalized-losses"].astype("float").mean(axis=0)
print("Average of normalized-losses:", avg_norm_loss)


# In[17]:


df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace=True)


# In[18]:


avg_bore=df['bore'].astype('float').mean(axis=0)
print("Average of bore:", avg_bore)


# In[19]:


df["bore"].replace(np.nan, avg_bore, inplace=True)


# In[20]:


avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average of horsepower:", avg_horsepower)


# In[21]:


df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)


# In[22]:


avg_peakrpm = df['peak-rpm'].astype('float').mean(axis=0)
print("Average of peak rpm:", avg_peakrpm)


# In[23]:


df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)


# In[24]:


#to see values that are present in a column
df['num-of-doors'].value_counts()


# In[25]:


df['num-of-doors'].value_counts().idxmax()


# In[26]:


df["num-of-doors"].replace(np.nan, "four", inplace=True)


# In[27]:


# simply drop whole row with NaN in "price" column
df.dropna(subset=["price"], axis=0, inplace=True)

# reset index, because we droped two rows
df.reset_index(drop=True, inplace=True)


# In[28]:


df.head()


# In[29]:


df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")


# In[30]:


df.dtypes


# In[31]:


df.head()


# In[32]:


# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df['city-L/100km'] = 235/df["city-mpg"]

#check your transformed data
df.head()


# In[33]:


df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()


# In[34]:


df["horsepower"]=df["horsepower"].astype(int, copy=True)


# In[35]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# In[36]:


bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)
bins


# In[37]:


group_names = ['Low', 'Medium', 'High']


# In[38]:


df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels=group_names, include_lowest=True )
df[['horsepower','horsepower-binned']].head(20)


# In[39]:


df["horsepower-binned"].value_counts()


# In[40]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
pyplot.bar(group_names, df["horsepower-binned"].value_counts())

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# In[41]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot


# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

# set x/y labels and plot title
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")


# In[42]:


df.columns


# In[43]:


dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()


# In[44]:


dummy_variable_1.rename(columns={'gas':'fuel-type-gas', 'diesel':'fuel-type-diesel'}, inplace=True)
dummy_variable_1.head()


# In[45]:


# merge data frame "df" and "dummy_variable_1" 
df = pd.concat([df, dummy_variable_1], axis=1)

# drop original column "fuel-type" from "df"
df.drop("fuel-type", axis = 1, inplace=True)


# In[46]:


df.head()


# In[47]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[48]:


df.corr()


# In[49]:


# Engine size as potential predictor variable of price
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)


# In[50]:


df[["engine-size", "price"]].corr()


# In[51]:


sns.regplot(x="highway-mpg", y="price", data=df)


# In[52]:


df[['highway-mpg', 'price']].corr()


# In[53]:


sns.regplot(x="peak-rpm", y="price", data=df)


# In[54]:


df[['peak-rpm', 'price']].corr()


# In[55]:


sns.boxplot(x="body-style", y="price", data=df)


# In[56]:


sns.boxplot(x="engine-location", y="price", data=df)


# In[57]:


#drive-wheels
sns.boxplot(x="drive-wheels", y="price", data=df)


# In[58]:


df.describe()


# In[59]:


df.describe(include=['object'])


# In[60]:


df['drive-wheels'].value_counts()


# In[61]:


df['drive-wheels'].value_counts().to_frame()


# In[62]:


drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts


# In[63]:


drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts


# In[64]:


# engine-location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)


# In[65]:


df['drive-wheels'].unique()


# In[66]:


df_group_one = df[['drive-wheels', 'body-style', 'price']]


# In[67]:


# grouping results
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
df_group_one


# In[68]:


# grouping results
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_test1


# In[69]:


#Pivot table
grouped_pivot = grouped_test1.pivot(index='drive-wheels',columns='body-style')
grouped_pivot


# In[70]:


#fill missing values with 0
grouped_pivot = grouped_pivot.fillna(0)
grouped_pivot


# In[71]:


#use the grouped results
plt.pcolor(grouped_pivot, cmap='RdBu')
plt.colorbar()
plt.show()


# In[72]:


fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

#label names
row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

#rotate label if too long
plt.xticks(rotation=90)

fig.colorbar(im)
plt.show()


# In[73]:


df.corr()


# In[74]:


from scipy import stats


# In[75]:


pearson_coef, p_value = stats.pearsonr(df['wheel-base'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P =", p_value)  


# In[76]:


pearson_coef, p_value = stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  


# In[77]:


pearson_coef, p_value = stats.pearsonr(df['length'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  


# In[78]:


pearson_coef, p_value = stats.pearsonr(df['width'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  


# In[79]:


pearson_coef, p_value = stats.pearsonr(df['curb-weight'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  


# In[80]:


pearson_coef, p_value = stats.pearsonr(df['engine-size'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  


# In[81]:


pearson_coef, p_value = stats.pearsonr(df['bore'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  


# In[82]:


pearson_coef, p_value = stats.pearsonr(df['city-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  


# In[83]:


pearson_coef, p_value = stats.pearsonr(df['highway-mpg'], df['price'])
print("The Pearson Correlation Coefficient is", pearson_coef, " with a P-value of P = ", p_value)  


# In[84]:


grouped_test2=df_gptest[['drive-wheels', 'price']].groupby(['drive-wheels'])
grouped_test2.head(2)


# In[85]:


df_gptest


# In[86]:


grouped_test2.get_group('4wd')['price']


# In[87]:


# ANOVA
f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'], grouped_test2.get_group('4wd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val)  


# In[88]:


f_val, p_val = stats.f_oneway(grouped_test2.get_group('fwd')['price'], grouped_test2.get_group('rwd')['price'])  
 
print( "ANOVA results: F=", f_val, ", P =", p_val )


# In[89]:


f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('rwd')['price'])  
   
print( "ANOVA results: F=", f_val, ", P =", p_val)


# In[90]:


f_val, p_val = stats.f_oneway(grouped_test2.get_group('4wd')['price'], grouped_test2.get_group('fwd')['price'])  
 
print("ANOVA results: F=", f_val, ", P =", p_val) 


# In[91]:


from sklearn.linear_model import LinearRegression


# In[92]:


lm = LinearRegression()
lm 


# In[93]:


X = df[['highway-mpg']]
Y = df[['price']]


# In[94]:


lm.fit(X,Y)


# In[95]:


Yhat=lm.predict(X)
Yhat[0:5]


# In[96]:


lm.intercept_


# In[97]:


lm.coef_


# In[98]:


Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]


# In[99]:


lm.fit(Z, df['price'])


# In[100]:


lm.intercept_


# In[101]:


lm.coef_


# In[102]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[103]:


width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x="highway-mpg", y="price", data=df)
plt.ylim


# In[104]:


plt.figure(figsize=(width, height))
sns.regplot(x="peak-rpm", y="price", data=df)
plt.ylim(0,)


# In[105]:


width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(x=df['highway-mpg'], y=df['price'])
plt.show()


# In[106]:


Y_hat = lm.predict(Z)


# In[107]:


plt.figure(figsize=(width, height))


ax1 = sns.distplot(df['price'], hist=False, color="r", label="Actual Value")
sns.distplot(Y_hat, hist=False, color="b", label="Fitted Values" , ax=ax1)


plt.title('Actual vs Fitted Values for Price')
plt.xlabel('Price (in dollars)')
plt.ylabel('Proportion of Cars')

plt.show()
plt.close()


# In[108]:


def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()


# In[109]:


x = df['highway-mpg']
y = df['price']


# In[110]:


f = np.polyfit(x, y, 3)
p = np.poly1d(f)
print(p)


# In[111]:


PlotPolly(p, x, y, 'higheway-mpg')


# In[112]:


np.polyfit(x, y, 3)


# In[113]:


from sklearn.preprocessing import PolynomialFeatures


# In[114]:


pr=PolynomialFeatures(degree=2)
pr


# In[115]:


Z_pr=pr.fit_transform(Z)


# In[116]:


Z.shape


# In[117]:


Z_pr.shape


# In[118]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[119]:


Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]


# In[120]:


pipe=Pipeline(Input)
pipe


# In[121]:


Z = Z.astype(float)
pipe.fit(Z,y)


# In[122]:


ypipe=pipe.predict(Z)
ypipe[0:4]


# In[123]:


#highway_mpg_fit
lm.fit(X,Y)
#find the R
print('The R-square is: ', lm.score(X, Y))


# In[124]:


Yhat=lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])


# In[125]:


from sklearn.metrics import mean_squared_error


# In[126]:


mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)


# In[127]:


from sklearn.metrics import r2_score


# In[128]:


r_squared = r2_score(y, p(x))
print('The R-square value is: ', r_squared)


# In[129]:


mean_squared_error(df['price'], p(x))


# In[130]:


import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[131]:


new_input=np.arange(1, 100,1).reshape(-1, 1)


# In[132]:


lm.fit(X, Y)
lm


# In[133]:


yhat=lm.predict(new_input)
yhat[0:5]


# In[134]:


plt.plot(new_input, yhat)
plt.show()


# In[135]:


def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.kdeplot(RedFunction, color="r", label=RedName)
    ax2 = sns.kdeplot(BlueFunction, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.show()
    plt.close()


# In[136]:


def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()


# In[137]:


y_data = df['price']


# In[138]:


x_data = df.drop('price', axis=1)


# In[139]:


from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# In[140]:


from sklearn.linear_model import LinearRegression


# In[141]:


lre=LinearRegression()


# In[142]:


lre.fit(x_train[['horsepower']], y_train)


# In[143]:


lre.score(x_test[['horsepower']], y_test)


# In[144]:


lre.score(x_train[['horsepower']], y_train)


# In[145]:


from sklearn.model_selection import cross_val_score


# In[146]:


Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4)


# In[147]:


Rcross


# In[148]:


print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())


# In[149]:


-1 * cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')


# In[150]:


-1 * cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')


# In[151]:


from sklearn.model_selection import cross_val_predict


# In[152]:


yhat = cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)
yhat[0:5]


# In[153]:


lr = LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)


# In[154]:


yhat_train = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_train[0:5]


# In[155]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[156]:


yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
yhat_train[0:5]


# In[157]:


Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)


# In[158]:


Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)


# In[159]:


from sklearn.preprocessing import PolynomialFeatures


# In[160]:


x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.45, random_state=0)


# In[161]:


pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train[['horsepower']])
x_test_pr = pr.fit_transform(x_test[['horsepower']])
pr


# In[162]:


poly = LinearRegression()
poly.fit(x_train_pr, y_train)


# In[163]:


yhat = poly.predict(x_test_pr)
yhat[0:5]


# In[164]:


print("Predict values:", yhat[0:4])
print("True values:", y_test[0:4].values)


# In[165]:


PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly,pr)


# In[166]:


poly.score(x_train_pr, y_train)


# In[167]:


poly.score(x_test_pr, y_test)


# In[168]:


Rsqu_test = []

order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    
    x_test_pr = pr.fit_transform(x_test[['horsepower']])    
    
    lr.fit(x_train_pr, y_train)
    
    Rsqu_test.append(lr.score(x_test_pr, y_test))

plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')    


# In[169]:


def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr,y_train)
    PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train,y_test, poly, pr)


# In[171]:


pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr=pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])


# In[172]:


from sklearn.linear_model import Ridge


# In[173]:


RigeModel=Ridge(alpha=1)


# In[174]:


RigeModel.fit(x_train_pr, y_train)


# In[175]:


yhat = RigeModel.predict(x_test_pr)


# In[176]:


print('predicted:', yhat[0:4])
print('test set :', y_test[0:4].values)


# In[177]:


from tqdm import tqdm

Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0,1000))
pbar = tqdm(Alpha)

for alpha in pbar:
    RigeModel = Ridge(alpha=alpha) 
    RigeModel.fit(x_train_pr, y_train)
    test_score, train_score = RigeModel.score(x_test_pr, y_test), RigeModel.score(x_train_pr, y_train)
    
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)


# In[178]:


width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()


# In[179]:


from sklearn.model_selection import GridSearchCV


# In[180]:


parameters1= [{'alpha': [0.001,0.1,1, 10, 100, 1000, 10000, 100000, 100000]}]
parameters1


# In[181]:


RR=Ridge()
RR


# In[182]:


Grid1 = GridSearchCV(RR, parameters1,cv=4)


# In[183]:


Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)


# In[184]:


BestRR=Grid1.best_estimator_
BestRR


# In[185]:


BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)


# In[ ]:




