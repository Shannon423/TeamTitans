# TeamTitans
###Sprint 2
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("Setup Complete")
df = pd.read_excel (r'C:\Users\user\Downloads\MASA.xlsx')

print(df.info())

Agency = df.groupby('Agency').size()
Agency_Type = df.groupby('Agency Type').size()
D_Channel = df.groupby('Distribution Channel').size()
Name = df.groupby('Product Name').size()
Claim = df.groupby('Claim').size()
Duration = df.groupby('Duration').size()
Destination = df.groupby('Destination').size()
Sales = df.groupby('Net Sales').size()
Comm = df.groupby('Commision (in value)').size()
Gender = df.groupby('Gender').size()
Age = df.groupby('Age').size()

df.describe()
print(df.corr())

#Correlation of Columns using Heatmap
f,ax = plt.subplots(figsize=(10, 10))
plt.title('Correlation of Columns using Heatmap')
sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", linewidth=0.5, linecolor="black", fmt= '.2f', ax=ax)

#Correlation of Net Sales and Duration by using Scatterplot
f,ax = plt.subplots(figsize=(10, 10))
plt.title('Correlation of Net Sales and Duration by using Scatterplot')
sns.scatterplot(x="Duration", y="Net Sales", data= df, color='green')
plt.axhline(0, c='black', ls='--')


#Correlation of Duration and Age by using Scatterplot
f,ax = plt.subplots(figsize=(10, 10))
plt.title('Correlation of Duration and Age by using Scatterplot')
sns.scatterplot(x="Duration", y="Age", data= df, color = '#228B22')

#Pairplot by grouping as Agency Type
sns.set(style = "ticks")
sns.pairplot(df, hue = "Agency Type", palette= ['r','g'])


#Correlation of Net Sales and Commision (in value) by grouping of Agency Type
f,ax = plt.subplots(figsize=(10, 10))
plt.title('Correlation of Net Sales and Commision (in value) by grouping of Agency Type using Scatterplot')
sns.scatterplot(x="Commision (in value)", y="Net Sales", data= df, hue='Agency Type',palette = ['#e41a1c','#377eb8'])
plt.axhline(0, c='black', ls='--')
f,ax = plt.subplots(figsize=(10, 10))
sns.lmplot(x="Commision (in value)", y="Net Sales", data= df, hue='Agency Type',palette = ['#e41a1c','#377eb8'])
plt.title('Correlation of Net Sales and Commision (in value) by grouping of Agency Type using Line Plot')
plt.axhline(0, c='black', ls='--')

#Correlation of the Net Sales and Commision (in value)
df.groupby('Agency Type')[['Net Sales','Commision (in value)']].corr()

#The Distribution plot by Age

distPlot = sns.distplot(df['Age'], color ='#4daf4a')
plt.title("The Distribution plot of Age")
print(distPlot)

meanAgencyType = df.groupby(by = "Agency Type")["Net Sales"].mean()
print(meanAgencyType)
plt.title('The Violin Plot of Agency Type with Net Sales')
sns.violinplot(x = "Agency Type", y = "Net Sales", data = df, palette =['#e41a1c','#377eb8'])
plt.axhline(0, c='black', ls='--')


#Box plot of Commision according to age categories
df['age_bins'] = pd.cut(df['Age'], bins = [0, 20, 40, 60, 80, 100, 120])

plt.figure(figsize=(12,4))
sns.boxplot(x='age_bins', y='Commision (in value)', data=df) 
plt.title('Commision according to age categories', size='23')
plt.xticks(rotation='25')
plt.grid(True)
plt.ylabel('Commision (in value)',size=18)
plt.xlabel('Age',size=18)


#Lineplot of Age and Net Sales
f,ax = plt.subplots(figsize=(10, 10))
plt.title('Line Plot of Net Sales with Age (years)')
sns.lineplot(x="Age", y='Net Sales',data = df, color='#4daf4a' )
plt.show()


###Sprint 3
## Missing Value Treatment
### Determine which column has missing values
print(df.isnull().sum())

### Remove the column as approximately 70% are missing values
updated_df = df.dropna(axis=1)
print(updated_df.info())

## Outlier Analysis
from scipy import stats

### Find Z-score for Duration
z1 = np.abs(stats.zscore(df['Duration']))
print(z1)

threshold = 3

### Position of the outlier for Duration
print(np.where(abs(z1) > 3))

### Find Z-score for Net Sales
z2 = np.abs(stats.zscore(df['Net Sales']))
print(z2)

threshold = 3

### Position of the outlier for Net Sales
print(np.where(abs(z2) > 3))

### Find Z-score for Commision
z3 = np.abs(stats.zscore(df['Commision (in value)']))
print(z3)

threshold = 3

### Position of the outlier for Commision
print(np.where(abs(z3) > 3))

### Find Z-score for Age
z4 = np.abs(stats.zscore(df['Age']))
print(z4)

threshold = 3

### Position of the outlier for Age
print(np.where(abs(z4) > 3))


## Identify outlier (graphical method)
### boxplot method(for Duration)
f,a=plt.subplots(figsize=(10,6))
sns.boxplot(df['Duration'])
plt.title("Boxplot of Duration")

### boxplot method(for Net Sales)
f,a=plt.subplots(figsize=(10,6))
sns.boxplot(df['Net Sales'])
plt.title("Boxplot of Net Sales")

### boxplot method(for Commision)
f,a=plt.subplots(figsize=(10,6))
sns.boxplot(df['Commision (in value)'])
plt.title("Boxplot of Commision")

### boxplot method(for Age)
f,a=plt.subplots(figsize=(10,6))
sns.boxplot(df['Age'])
plt.title("Boxplot of Age")

## Count for outliers
### Duration
d_mean=df['Duration'].agg(['mean','std'])
mu = d_mean.loc['mean']
sigma = d_mean.loc['std']
def get_outliers(df, mu=mu, sigma=sigma, n_sigmas=3):
    x = df['Duration']
    mu = mu
    sigma = sigma
    
    if (x > mu+n_sigmas*sigma) | (x<mu-n_sigmas*sigma):
        return 1
    else:
        return 0

df['outliers']=df.apply(get_outliers,axis=1)

print(df.outliers.value_counts())

### Net Sales
d_mean=df['Net Sales'].agg(['mean','std'])
mu = d_mean.loc['mean']
sigma = d_mean.loc['std']
def get_outliers(df, mu=mu, sigma=sigma, n_sigmas=3):
    x = df['Net Sales']
    mu = mu
    sigma = sigma
    
    if (x > mu+n_sigmas*sigma) | (x<mu-n_sigmas*sigma):
        return 1
    else:
        return 0

df['outliers']=df.apply(get_outliers,axis=1)

print(df.outliers.value_counts())

### Commision

d_mean=df['Commision (in value)'].agg(['mean','std'])
mu = d_mean.loc['mean']
sigma = d_mean.loc['std']
def get_outliers(df, mu=mu, sigma=sigma, n_sigmas=3):
    x = df['Commision (in value)']
    mu = mu
    sigma = sigma
    
    if (x > mu+n_sigmas*sigma) | (x<mu-n_sigmas*sigma):
        return 1
    else:
        return 0

df['outliers']=df.apply(get_outliers,axis=1)

print(df.outliers.value_counts())

### Age
d_mean=df['Age'].agg(['mean','std'])
mu = d_mean.loc['mean']
sigma = d_mean.loc['std']
def get_outliers(df, mu=mu, sigma=sigma, n_sigmas=3):
    x = df['Age']
    mu = mu
    sigma = sigma
    
    if (x > mu+n_sigmas*sigma) | (x<mu-n_sigmas*sigma):
        return 1
    else:
        return 0

df['outliers']=df.apply(get_outliers,axis=1)

print(df.outliers.value_counts())


## Data transformation(log transformation)
df=df.dropna()
log_v=['Duration','Net Sales','Commision (in value)','Age']
u=plt.figure(figsize=(25,10))
for i in range(len(log_v)):
    v=log_v[i]
    trf= "log_"+v
    df[trf]=np.log10(df[v]+1)
    
    s=u.add_subplot(2,5,i+1)
    s.set_xlabel(v)
    df[trf].plot(kind='hist')


## Data transformation(arrange columns)
p=pd.DataFrame(updated_df,columns=['Product Name', 'Destination', 'Distribution Channel','Agency','Agency Type','Age','Claim','Duration','Net Sales','Commision (in value)'])
modified_df = p.sort_values(by = ['Product Name','Age'])
print(modified_df)

plt.show()

###Sprint 4
##Train and Test Sprint (based on Commission to other variables)
#input dataframe y
y_data = df['Commision (in value)']

#drop commision data in x data
x_data=df.drop('Commision (in value)', axis=1)

#import module
from sklearn.model_selection import train_test_split

#randomly split our data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.15, random_state=1)
print("number of test samples:", x_test.shape[0])
print("number of training samples:", x_train.shape[0])

#create a Linear Regression object (for Net Sales)
x = df['Net Sales']
y = df['Commision (in value)']
import numpy as np
from sklearn.linear_model import LinearRegression
lre = LinearRegression()
lre.fit(x_train[['Net Sales']], y_train)
a = lre.score(x_test[['Net Sales']], y_test)
print(a)

lre.fit(x_test[['Net Sales']], y_test)
b = lre.score(x_train[['Net Sales']], y_train)
print(b)

#create a Linear Regression object (for Duration)
z = df['Duration']
y = df['Commision (in value)']
import numpy as np
from sklearn.linear_model import LinearRegression
lre = LinearRegression()
lre.fit(x_train[['Duration']], y_train)
c = lre.score(x_test[['Duration']], y_test)
print(c)

lre.fit(x_test[['Duration']], y_test)
d = lre.score(x_train[['Duration']], y_train)
print(d)

#create a Linear Regression object (for Age)
w = df['Age']
y = df['Commision (in value)']
import numpy as np
from sklearn.linear_model import LinearRegression
lre = LinearRegression()
lre.fit(x_train[['Age']], y_train)
e = lre.score(x_test[['Age']], y_test)
print(e)

lre.fit(x_test[['Age']], y_test)
f = lre.score(x_train[['Age']], y_train)
print(f)

###Module Development
Z = df[['Duration', 'Net Sales', 'Age']]
lre.fit(Z, df['Commision (in value)'])
print(lre.intercept_)
print(lre.coef_)

#Commision = -6.449592910116424 + 0.01586273 x Duration + 0.24843086 x Net Sales + 0.13424185 x Age

##Multiple Linear Regression
# fit the model 
lre.fit(Z, df['Commision (in value)'])

# Find the R^2
print('The R-square is: ', lre.score(Z, df['Commision (in value)']))
#We can say that ~ 43.84 % of the variation of commision is explained by this multiple linear regression

#Calculate MSE
from sklearn.metrics import mean_squared_error
Y_predict_multifit = lre.predict(Z)
print('The mean square error of commision and predicted value using multifit is: ', \
      mean_squared_error(df['Commision (in value)'], Y_predict_multifit))
      
###Sprint 5
# Pivot Tables
## Pivot Table 1 (Claim frequency of each agency)
![](Images/PivotTable1.png)

## Pivot Table 2 (Claim frequency of specific destinations with duration = 60)
![](Images/PivotTable2.png)


# One Way Analysis
## Airlines
![](Images/OneWay-Airlines.png)

## Travel Agency
![](Images/OneWay-TravelAgency.png)

## Destination
![](Images/OneWay-Destination.png)


# Two Way Analysis
## Airlines
![](Images/TwoWay-Airlines.png)

## Travel Agency
![](Images/TwoWay-TravelAgency.png)

## Destination
![](Images/TwoWay-Destination.png)
