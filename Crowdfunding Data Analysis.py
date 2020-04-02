#!/usr/bin/env python
# coding: utf-8

# # Crowd Funding Data Analysis for Kickstarter

# ## 1. Project Overview : Crowd Funding Data Analysis for Kickstarter

# ### CrowdFunding
# 
# Crowdfunding is a concept of collecting funds from people , on a public platform for a particular cause. The cause can be initated as a campaign or a project which is launched globally, for instance on a website. This website can be made accessible to the contributers who wish to pledge any amount for a cause from anywhere across the world. The pledged amount could be contributions made for social causes or funds for startups.
# 
# ### Kickstarter
# 
# Kickstarter is a one of the crowdfunding websites that is focused on encouraging creativity. It hosts projects across a wide range of categories thereby providing a public platform for the campaign or project owners to intiate their cause.
# 
# Access website here : https://www.kickstarter.com/
# 
# ### Goal
# 
# In this project we would like to analyze the success rate of a campaign hosted on kicstarter. Some of the questions that we would like to address are :
# 
# 1. What are the factors contributing towards the success or failure of a project?
# 2. For a campaign owner to launch a project, what parameters should be considered in order for it to be a success ?
# 
# ### Audience
# 
# We are aiming for this analysis will be useful to people who are interested in raising funds for their cause, so that they can decide the amount that should be set as their goal and the timeline within which this goal can be achieved, in order to have a successful campaign.
# 
# ### Source of data
# 
# This analysis is based on datasets that is obtained by webscrapping services provided by WebRobots.
# 
# 1. Using the WebRobots tools, information is extracted from the Kickstarter website.
# 2. There were 56 files from years 2009 to 2019.
# 3. All of the 56 files were merged into a single file and some unwanted columns dropped before cleaning the data further.

# ## 2. Scope of Analysis :

# ### Understanding Data :
# 
# Below is a preview of Kickstarter. To give a brief on the features we are focusing on :
# 
# 1. The 'State' column which defines the status of the Project hosted on kickstarter for which funds are being raised. The     'Project Status' can be one of the following : Successful, Failed, Suspended and Cancelled. 
# 
# 2. Every project that is hosted on the Kickstarter website has the a target 'Goal' Amount (USD scaled to thousands) and a 'Timeline' (number of days to achieve the target goal). 
# 
# 3. The contributors are the 'backers' which who pledge amounts that they wish to , in order to support the campaign. In other words, the 'Pledged' Amount is the amount of donations made. 
# 
# 4. Contributors can pledge amounts until the deadline date. 
# 
# ### Objective :
# 
# Our analysis aims at building a classification model to predict the success or failure of a new project with a given goal and timeline. Further, we attempt to predict ranges that goal and timelines can be modified to prevent a project from failing.

# ## 3. Data Cleaning and Merging

# #### Importing important python libraries to carry out this analysis

# In[1]:


# installations
#tree isualization 
#pip install graphviz
#pip install pydotplus

#gradient boosting
#pip install xgboost


# In[2]:


#import all important libraries

#Data cleaning 
import glob
import pandas as pd
import pandas_profiling
import numpy as np
import datetime as dt
import ast 
import re
import math 
import os
import jupyter_contrib_nbextensions

#ignore warnings
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

#Exploratory Data Analysis 
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt

#Machine Learning

#Hypothesis testing
from scipy.stats import ttest_ind

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Cross Validation
from sklearn import model_selection

#Linear Regression
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

#Decision Trees
from sklearn.tree import DecisionTreeRegressor
#from sklearn.cross_validation import cross_val_score, crossvalidation
from sklearn.model_selection import cross_val_score, ShuffleSplit
#from sklearn.model_selection import crossvalidation

#KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

#Random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# #### 3.1 Reading multiple files into one

# In[7]:


#Reading multiple files
path = r'C:\Users\Radhika\Anaconda3\Data Science Project\Dataset\Files' # use your path
all_files = glob.glob(path + "/*.csv")

files = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    files.append(df)

frame = pd.concat(files, axis=0, ignore_index=True)

#Preview Data
frame.head()


# #### Selecting features from Webscrapped data

# In[8]:


#webscraped
df_kickstarter = frame[["state","id","created_at","launched_at","deadline","category","country","currency","current_currency","fx_rate","static_usd_rate","goal","usd_pledged","location","backers_count","urls"]]
df_kickstarter.head()


# #### 3.2 Data types and conversions

# In[9]:


print(df_kickstarter.dtypes)

#factor conversions
df_kickstarter['id'] = df_kickstarter['id'].astype(object)


# ##### 3.2.1 Date conversions

# In[10]:


#date conversions
df_kickstarter["created_at"] = pd.to_datetime(df_kickstarter["created_at"],unit='s')
df_kickstarter["created_at"]= df_kickstarter["created_at"].dt.date

df_kickstarter["launched_at"] = pd.to_datetime(df_kickstarter["launched_at"],unit='s')
df_kickstarter["launched_at"]= df_kickstarter["launched_at"].dt.date

df_kickstarter["deadline"] = pd.to_datetime(df_kickstarter["deadline"],unit='s')
df_kickstarter["deadline"]= df_kickstarter["deadline"].dt.date

#Splitting Created_at date into year and month 
df_kickstarter[['lauchedat_year','lauchedat_month']] = df_kickstarter.launched_at.apply(lambda x : pd.Series(x.strftime("%Y,%m").split(",")))
df_kickstarter[['deadline_year','deadline_month']] = df_kickstarter.deadline.apply(lambda x : pd.Series(x.strftime("%Y,%m").split(",")))

df_kickstarter["launched_at"] = pd.to_datetime(df_kickstarter["launched_at"])
df_kickstarter["deadline"] = pd.to_datetime(df_kickstarter["deadline"])
df_kickstarter["created_at"] = pd.to_datetime(df_kickstarter["created_at"])

df_kickstarter['month_deadline'] = pd.to_datetime(df_kickstarter['deadline'], format='%m').dt.month_name().str.slice(stop=3)
df_kickstarter['month_launched'] = pd.to_datetime(df_kickstarter['launched_at'], format='%m').dt.month_name().str.slice(stop=3)


# ##### 3.2.2 Extract category

# In[11]:


# looping all records and extracting category
array_cat = df_kickstarter["category"].values
rows = len(array_cat)
i=0
#array_cat_names=[]
category=[]
sub_category=[]
print(rows)
print(i)


while i <=(rows-1):
    str_temp = array_cat[i]
    tuple_temp = ast.literal_eval(str_temp)
    dict_temp = dict(tuple_temp)
 #   print(dict_temp["name"])
 #   array_cat_names.append(dict_temp["name"])
    if('/' in dict_temp['slug']):
        cat = dict_temp["slug"].split('/',2)
        category.append(cat[0])
        sub_category.append(cat[1])
    else :
        category.append(dict_temp['slug'])
        sub_category.append(dict_temp['slug'])
    i=i+1
#    print(i)

#drop old category
df_kickstarter = df_kickstarter.drop(columns="category")

#insert new category
df_kickstarter.insert(4,"category",category)
df_kickstarter.insert(5,"sub_category",sub_category)


# ##### 3.2.3 Extract Location

# In[12]:


# looping all records and extracting location
array_loc = df_kickstarter["location"].values
rows = len(array_loc)
i=0
array_loc_names=[]
print(rows)
print(i)


while i <=(rows-1):
    if( pd.isnull(array_loc[i]) ):
        array_loc_names.append('NA')
        i=i+1
    else:
        loc=re.search('"name":"(.*)","slug"',array_loc[i])
        array_loc_names.append(loc.group(1))
        i=i+1
        
len(array_loc_names)  

#drop old location
df_kickstarter = df_kickstarter.drop(columns="location")

#insert new location
df_kickstarter.insert(6,"city",array_loc_names) 


# ##### 3.2.4 Calculating Timeline

# In[13]:


#calculating number of days
no_days=df_kickstarter['deadline']-df_kickstarter['launched_at']

#inserting timeline
df_kickstarter.insert(4,"timeline",no_days.dt.days) 


# ##### 3.2.5 Currency Conversion

# In[14]:


#currency conversion
usd_goal = df_kickstarter['goal'] * df_kickstarter['static_usd_rate']
df_kickstarter.insert(13,"usd_goal",usd_goal) 

# Scale the pledged amount and goal for interpretable plots and stats

df_kickstarter['usd_pledged'] = df_kickstarter['usd_pledged']/1000
df_kickstarter['usd_goal'] =  df_kickstarter['usd_goal'] / 1000


# ##### 3.2.6 Extracting URL for case study

# In[15]:


#extract URL
print(df_kickstarter["urls"].head())

# looping all records and extracting urls
array_url = df_kickstarter["urls"].values
rows = len(array_url)
i=0
#array_cat_names=[]
web_url=[]
rewards_url=[]
print(rows)
print(i)


while i <=(rows-1):
    str_temp = array_url[i]
    tuple_temp = ast.literal_eval(str_temp)
    dict_temp = dict(tuple_temp)
 #   print(dict_temp["name"])
 #   array_cat_names.append(dict_temp["name"])
    web_url.append(dict_temp['web']['project'])
    rewards_url.append(dict_temp['web']['rewards'])
    i=i+1
#    print(i)

print(len(web_url))
print(len(rewards_url))

#insert the URLS
df_kickstarter.insert(14,"web_url",web_url) 
df_kickstarter.insert(15,"rewards_url",rewards_url) 

df_kickstarter = df_kickstarter.drop(columns="urls")


# ### 3.3 Preview of Cleaned Data and Saving Merged file

# In[16]:


# Preview cleaned data and save into a file
df_kickstarter.head()


# In[ ]:


#Saving the file for reference
#df_kickstarter.to_csv('C:\\Users\\Radhika\\Anaconda3\\Data Science Project\\Dataset\\cleaned\\Files'+'CleanedData.csv')


# ## 4. Exploratory Data Analysis
# 
# After cleaning the data i.e, extracting the features from webscrapped data, converting date and numeric columns to appropriate formats, the file is merged to excel and is now ready for exploratory data analysis.
# 
# 1. Trends of Project Status in the time frame of 9 years
# 2. Proportions of Successful and Failed projects per category and geographical location
# 3. Scatterplots to understand relationships between features

# In[7]:


# Reading the merged file

# set path
path= r"C:\Users\akhur\Desktop\IDS Project\Cleaned"
cwd=os.chdir(path)

# read the file
df_merge = pd.read_csv("MergedDataNov.csv",index_col=0)

# check rows and colums
print("Number of observations and feautres in dataset : ")
print(df_merge.shape)


# In[11]:


#Removing Duplicates
# Sort by Project ID
df_merge.sort_values('id',inplace=True)

# Drop the duplicateds
df_merge.drop_duplicates(subset='id',keep='first', inplace=True)

#Check the final count
print("Dimensions after removing duplicates")
print(df_merge.shape)
df_merge[['state','category','country','launched_at','timeline','usd_goal','usd_pledged','backers_count']].head()


# Note : After removing duplicates the dataset contains total of 182634 rows and 26 columns. Duplciates are removed after data cleaning so that raw data can be carefully reviewed in its interpretable format before unknowingly deleting records.

# ## 4.1. Project status trendlines

# There are 5 states of projects on Kickstarter. They are,
# 1. Successful
# 2. Failed
# 3. Suspended
# 4. Live
# 5. Cancelled
#  
# The trendlines of these projects are observed below.

# In[13]:


df_merge['state'].unique()


# #### 4.1.1 Trendlines : Creating Pivot table

# In[10]:


#pivot
trend_state = pd.DataFrame({'count' : df_merge.groupby( ['lauchedat_year','lauchedat_month','launched_at','state'] )['id'].size()}).reset_index()
trend_state=trend_state.pivot_table(trend_state, index=('lauchedat_year','lauchedat_month','launched_at'), columns='state')
trend_state.columns=trend_state.columns.droplevel(0)
trend_state.columns.name=None
trend_state=trend_state.reset_index()
trend_state.head()

#Replacing NaN with 0
trend_state=trend_state.fillna(0)
#trend_state.head()

#Converting to monthly time series
trends = trend_state[['launched_at','canceled','failed','live','successful','suspended']]
trends['launched_at']=pd.to_datetime(trends['launched_at'])
trends=trends.resample('M',on='launched_at').sum()
trends.head()


# In[24]:


trends.tail()


# #### 4.1.2 Trendlines : Project Status

# In[25]:


#trendlines
ax=trends[['canceled','failed','live','successful','suspended']].plot(linewidth=1,figsize=(16,9),title='Monthly Trendlines for Project outcomes : April 2009 to August 2019 ')
ax.set_ylabel('Number of Projects')
ax.set_xlabel('Years')


# #### Trend Analysis : Project Status : Change in patterns
# 
# The above trend lines represent the monthly volume of projects hosted on kickstarter from April 2009 to August 2019.
# 
# 1. From years 2010 to 2014 the projects hosted on Kicstarter were increaseing owing to its popularity has a crowdfunding platform. The number of failed an successful projects were both on the rise, although the successfull projects were significantly higher than the failed ones.
# 
# 2. In the year 2014, the failure of projects was noticibly higher. 
# 
# 3. The failure trends of projects continued to dominate over the successful trends uptil 2016.
# 
# 4. From 2016 to end of 2017 , two years, rate of failure and success of projects were similar. 
# 
# 5. Since 2018, the past year, there is an improvement in the chances of projects turning to be successful being higher and the failure trends dropping. 
# 
# 6. We also observe a seasonal dip in the success of projects every year towards the end. We will analyze the seasonality patterns in further detail during timeseries analysis. 
# 

# #### 4.1.3  Proportions of Projects based on Status
# Percentage of projects that are successful, failed,live and suspended are calculated to understand the data better.

# In[26]:


# % calculations
counts = df_merge[['id','state']].groupby(['state']).count()
percentage_count =( counts / df_merge['id'].count())*100
percentage_count


# Note :
# 1. 52% of the Projects are successful.
# 2. 40% of the Projects have failed

# ## 4.2. Project Proportions : State and Category wise

# In[46]:


#creating stacked barplot
df_pct = (df_merge.groupby(['category','state'])['id'].count()/df_merge.groupby(['category'])['id'].count())
# plot the stacked plot
pal = sns.color_palette("Set2")
df_pct.unstack().plot.bar(stacked=True,figsize=(10,6),colors=pal)

plt.legend(loc='upper right', bbox_to_anchor=(1.2,1.))


# ### Inference of Categorical Classification
# 
# 1. The proportions of successful projects are higher for categories like 'Dance', 'Comics', 'Music' and 'Publishing'
# 2. The proportions of failed projects are higher for categories like 'food' , 'journalism' and 'techonology'
# 
# As a result , we might want to infer that Kickstarter is a good platform for Dancers, Comic writers and Musicians to raise funds for their talent. Unfortunately the projects for Food or Technology are not very popluar among the crowd.

# ### 4.2.1 Project Proportions : State and Location wise

# In[47]:


#creating stacked barplot
df_pct = (df_merge.groupby(['country','state'])['id'].count()/df_merge.groupby(['country'])['id'].count())
# plot the stacked plot
pal = sns.color_palette("Set2")
df_pct.unstack().plot.bar(stacked=True,figsize=(10,6),colors=pal)

plt.legend(loc='upper right', bbox_to_anchor=(1.2,1.))


# ### Inference of Country-wise Classification
# 
# 1. The proportions of successful projects are higher for categories like 'Hong Kong', 'Great Britain', 'United States
# 2. The proportions of failed projects are higher for categories like 'Italy', 'Austria','Belgium', 'China'
# 
# As a result , we might want to infer that Kickstarter has good amount of contributors for Projects initiated from Asian countries , US and Great Britian.

# ## 4.3  Distribution of numeric data

# In[34]:


#Feature distribution
ax=sns.boxplot(data=df_merge[['backers_count','usd_pledged','usd_goal','timeline']])
#ax.set_yscale('log')


# In[28]:


#stats
df_merge[['backers_count','usd_pledged','usd_goal','timeline']].describe()


# Note: 
# 1. Each numeric feature have different variance and are on differnt scales. For instance, timeline has smaller values compared to pledged and goal amounts
# 
# 2. Scaled Pledged and Goal amoutns in USD are extremely high indicating special cases of unrealistically high amounts.
# 3. Minimum values of each of backers count, pledged and goal amounts is 0.
# 4. Minimum timeline is 1 day.
# 

# ## 4.4 Removing Unusual cases

# ### Univariate Distributions

# In[35]:


#subplot settings
fig, axes = plt.subplots(nrows=4,ncols=2,figsize=(10,9))

fig.tight_layout()
#fig.subplots_adjust(hspace=1)

sns.boxplot(df_merge['backers_count'],ax=axes[0][0])
sns.distplot(df_merge['backers_count'], ax=axes[0][1])

sns.boxplot(df_merge['timeline'],ax=axes[1][0])
sns.distplot(df_merge['timeline'], ax=axes[1][1])

sns.boxplot(df_merge['usd_pledged'],ax=axes[2][0])
sns.distplot(df_merge['usd_pledged'], ax=axes[2][1])

sns.boxplot(df_merge['usd_goal'],ax=axes[3][0])
sns.distplot(df_merge['usd_goal'], ax=axes[3][1])

df_merge_summary = df_merge[['backers_count', 'timeline','usd_pledged','usd_goal']].describe()
print(df_merge_summary)


# ### Statistics : Data patterns in 9 years 
# 
# Averages :
# 1. Average goal amount set is 44000 USD
# 2. Average project timeline is 40 days
# 3. Average contributions made for a project is 135 in backers count.
# 4. Average pledged amount is 11000 USD
# 
# Minimum values : Unusually, the minimum goal amount is 10 cents, timeline is 1 day
# 
# Maximum values : Some extreme cases have very high goal amounts like 152,350,076 USD ; pledged amount as 12,143,435.670  USD and backers count 105857 contributions.
# 
# Note : Due to such unusual cases, the distirubtions of data are difficult to work with. 

# ### Comparison  : Successful and Failed Projects

# In[14]:


#separate data
df_success = df_merge.query('state == "successful" ' )
df_fail = df_merge.query(' state == "failed" ')


# In[6]:


#Correlations and Subplots for Goal Amount and Number of Contributors for log transformed data 

#subplot settings
fig, axes = plt.subplots(nrows=5,ncols=2,figsize=(16,12))

#fig.tight_layout()
fig.subplots_adjust(hspace=1)

#control factors : Goal and Timeline

#Goal : Backers
#Goal : pledged
#Goal : Timeline (trend)
#Timeline : Backers
#Timeline : Pledged

#scatterplot

sns.scatterplot(x="usd_goal", y="backers_count", data= df_success, ax=axes[0][0])
axes[0][0].set_title('Successful Projects : USD Goal Amount (thousand) vs Number of Contributors')
axes[0][0].set_xlabel(' Goal $K ')
axes[0][0].set_ylabel('Contributors')

sns.scatterplot(x="usd_goal", y="backers_count", data= df_fail, ax=axes[0][1])
axes[0][1].set_title('Failed Projects : USD Goal Amount (thousand) vs Number of Contributors')
axes[0][1].set_xlabel('Goal $K ')
axes[0][1].set_ylabel(' Contributors')

sns.scatterplot(x="usd_goal", y="usd_pledged", data= df_success, ax=axes[1][0])
axes[1][0].set_title('Successful Projects :USD Goal Amount (thousand) vs Number of Contributors')
axes[1][0].set_xlabel('Goal $K ')
axes[1][0].set_ylabel('Pledged $K ')


sns.scatterplot(x="usd_goal", y="usd_pledged", data= df_fail, ax=axes[1][1])
axes[1][1].set_title('Failed Projects : USD Goal Amount (thousand) vs Number of Contributors')
axes[1][1].set_xlabel('Goal $K ')
axes[1][1].set_ylabel('Pledged $K ')


sns.scatterplot(x="usd_goal", y="timeline", data= df_success, ax=axes[2][0])
axes[2][0].set_title('Successful Projects : USD Goal Amount (thousand) vs Number of Contributors')
axes[2][0].set_xlabel('Goal $K ')
axes[2][0].set_ylabel('Timeline (Days)')

sns.scatterplot(x="usd_goal", y="timeline", data= df_fail, ax=axes[2][1])
axes[2][1].set_title('Failed Projects : USD Goal Amount (thousand) vs Number of Contributors')
axes[2][1].set_xlabel('Goal $K')
axes[2][1].set_ylabel('Timeline (Days)')

sns.scatterplot(x="timeline", y="backers_count", data= df_success, ax=axes[3][0])
axes[3][0].set_title('Successful Projects : Timeline vs Number of Contributors')
axes[3][0].set_xlabel('Timeline (Days)')
axes[3][0].set_ylabel('Contributors')

sns.scatterplot(x="timeline", y="backers_count", data= df_fail, ax=axes[3][1])
axes[3][1].set_title('Failed Projects : Timeline vs Number of Contributors')
axes[3][1].set_xlabel('Timeline (Days)')
axes[3][1].set_ylabel('Contributors')

sns.scatterplot(x="timeline", y="usd_pledged", data= df_success, ax=axes[4][0])
axes[4][0].set_title('Successful Projects : Timeline vs Number of Contributors')
axes[4][0].set_xlabel('Timeline (Days)')
axes[4][0].set_ylabel('Pledged $K ')

sns.scatterplot(x="timeline", y="usd_pledged", data= df_fail, ax=axes[4][1])
axes[4][1].set_title('Failed Projects : Timeline vs Number of Contributors')
axes[4][1].set_xlabel('Timeline (Days)')
axes[4][1].set_ylabel('Pledged $K )')


#print(corr_matrix)


# In[15]:


#stats : Successful projects
df_success[['usd_goal','usd_pledged','timeline','backers_count']].describe()


# In[17]:


#stats : Failed projects
df_fail[['usd_goal','usd_pledged','timeline','backers_count']].describe()


# In[20]:


#cap data
cond_cap = df_merge['usd_goal'] <= max(df_success['usd_goal'])
df_merge_org = df_merge
df_merge = df_merge[cond_cap]
print(df_merge.shape)
df_merge[['usd_goal','usd_pledged','timeline','backers_count']].describe()


# #### Inference
# 1. Average Goal amount of Failed projets is higher than Succesful projects.
# 2. The maximum goal amount for successful projects is 2000,000 USD. 
# 3. We can cap the data at goal amount of 2000,000 USD

# #### Removing all 1s

# In[21]:


# Scaling Goal and Pledged amount back to USD
df_merge['usd_goal']= df_merge['usd_goal']*1000
df_merge['usd_pledged']= df_merge['usd_pledged']*1000


# In[23]:


# Removing all data <= 1

#check for records where USD goal = 1 and remove

cond_goal = df_merge['usd_goal'] <= 1
cond_state = df_merge['state'] == 'successful'
print(df_merge[cond_goal & cond_state].shape)
p = round(  ((len(df_merge[cond_goal & cond_state])/len(df_merge))*100 ) , 2)
print("Proportion :  %s %%\n" % p)
#df_merge[cond_goal & cond_state][['state','category','country','launched_at','timeline','usd_goal','usd_pledged','backers_count']].head()


# In[24]:


#df_merge[cond_goal].groupby(['lauchedat_year','state'])['id'].count()


# In[25]:


# remove goal <= 1 day
cond = df_merge['usd_goal'] > 1
df_merge = df_merge[cond]
print("Count after removing Goal <= 1")
print(df_merge.shape)

# remove timeline <= 1 day
cond_days = df_merge['timeline'] > 1
df_merge = df_merge[cond_days]
print("Count after removing Timeline <= 1")
print(df_merge.shape)

# remove backers <= 1
cond_backers = df_merge['backers_count']> 1
df_merge = df_merge[cond_backers]
print("Count after removing Backers <= 1")
print(df_merge.shape)

# remove pledged <= 1
cond_pledged = df_merge['usd_pledged'] > 1
df_merge=df_merge[cond_pledged]
print("Count after removing Pledged <= 1")
print(df_merge.shape)


# ## 4.4 Outliers Treatment
# 
# Removing outliers based on z-score to retain 95% of the data

# #### 4.4.1 Outliers Treatment : Goal Amount in  (USD) 

# In[26]:


#outliers USD_goal
outliers = df_merge[(np.abs(stats.zscore(df_merge['usd_goal'])) >= 3 )]
outliers.sort_values('usd_goal',inplace=True)
print("Total Outlier observations :")
print(outliers.shape)
p_out_usd = round(  ((len(outliers)/len(df_merge))*100 ) , 2)
print("Proportion of Outliers : %s %%\n" % p_out_usd)
#print(outliers[['usd_goal','timeline','backers_count','usd_pledged']].describe())
#print(outliers[['id','launched_at','category','sub_category','country','usd_goal','timeline','backers_count','usd_pledged','state','web_url']].head())

#remove the outliers
outliers_sub = df_merge[(np.abs(stats.zscore(df_merge['usd_goal'])) >=3 )]
df_merge = df_merge[ ~df_merge.index.isin(outliers_sub.index) ]
print("Observations after removing outliers (USD Goal) : ")
print(df_merge.shape)


# In[27]:


#outliers[['id','launched_at','category','sub_category','country','usd_goal','timeline','backers_count','usd_pledged','state','web_url']].tail()


# #### 4.4.2 Outliers Treatment : Pledged Amount  (USD) 

# In[28]:


#outliers Pledged
outliers = df_merge[(np.abs(stats.zscore(df_merge['usd_pledged'])) >= 3 )]
outliers.sort_values('usd_pledged',inplace=True)
print("Total Outlier observations :")
print(outliers.shape)
p_out_usd = round(  ((len(outliers)/len(df_merge))*100 ) , 2)
print("Proportion of Outliers : %s %%\n" % p_out_usd)
#print(outliers[['usd_goal','timeline','backers_count','usd_pledged']].describe())

#remove the outliers
outliers_sub = df_merge[(np.abs(stats.zscore(df_merge['usd_pledged'])) >=3 )]
df_merge = df_merge[ ~df_merge.index.isin(outliers_sub.index) ]
print("Observations after removing outliers (USD Pledged) : ")
print(df_merge.shape)

#outliers[['id','launched_at','category','sub_category','country','usd_goal','timeline','backers_count','usd_pledged','state','web_url']].head()


# #### 4.4.3 Outliers Treatment : Backers Count 

# In[29]:


#outliers backers count
outliers = df_merge[(np.abs(stats.zscore(df_merge['backers_count'])) >= 3 )]
outliers.sort_values('backers_count',inplace=True)
print("Total Outlier observations :")
print(outliers.shape)
p_out_usd = round(  ((len(outliers)/len(df_merge))*100 ) , 2)
print("Proportion of Outliers : %s %%\n" % p_out_usd)
#print(outliers[['usd_goal','timeline','backers_count','usd_pledged']].describe())

#remove the outliers
outliers_sub = df_merge[(np.abs(stats.zscore(df_merge['backers_count'])) >=3 )]
df_merge = df_merge[ ~df_merge.index.isin(outliers_sub.index) ]
print("Observations after removing outliers (backers_count) : ")
print(df_merge.shape)

#outliers[['id','launched_at','category','sub_category','country','usd_goal','timeline','backers_count','usd_pledged','state','web_url']].head()


# #### 4.4.4 Outliers Treatment : Timeline

# In[30]:


#outliers timeline
outliers = df_merge[(np.abs(stats.zscore(df_merge['timeline'])) >= 3 )]
outliers.sort_values('timeline',inplace=True)
print("Total Outlier observations :")
print(outliers.shape)
p_out_usd = round(  ((len(outliers)/len(df_merge))*100 ) , 2)
print("Proportion of Outliers : %s %%\n" % p_out_usd)
#print(outliers[['usd_goal','timeline','backers_count','usd_pledged']].describe())

#remove the outliers
outliers_sub = df_merge[(np.abs(stats.zscore(df_merge['timeline'])) >=3 )]
df_merge = df_merge[ ~df_merge.index.isin(outliers_sub.index) ]
print("Observations after removing outliers (timeline) : ")
print(df_merge.shape)

#outliers[['id','launched_at','category','sub_category','country','usd_goal','timeline','backers_count','usd_pledged','state','web_url']].head()


# ## 4.5 Log Transformation of Numeric Features
# 
# Log transformations additionally scale the data on the log scale.

# In[90]:


#normalizing data using logs
to_log = ['backers_count', 'timeline','usd_pledged','usd_goal']
df_log = df_merge[to_log].applymap(lambda x: np.log(x))
#df_log.insert(0,"state",df_merge['state'])

df_log.columns = 'log_' + df_log.columns

#subplot settings
fig, axes = plt.subplots(nrows=4,ncols=2,figsize=(10,9))

fig.tight_layout()
#fig.subplots_adjust(hspace=1)

sns.boxplot(df_log['log_backers_count'],ax=axes[0][0])
sns.distplot(df_log['log_backers_count'], ax=axes[0][1])

sns.boxplot(df_log['log_timeline'],ax=axes[1][0])
sns.distplot(df_log['log_timeline'], ax=axes[1][1])

sns.boxplot(df_log['log_usd_pledged'],ax=axes[2][0])
sns.distplot(df_log['log_usd_pledged'], ax=axes[2][1])

sns.boxplot(df_log['log_usd_goal'],ax=axes[3][0])
sns.distplot(df_log['log_usd_goal'], ax=axes[3][1])

df_log_summary = df_log[['log_backers_count', 'log_timeline','log_usd_pledged','log_usd_goal']].describe()
print(df_log_summary)


# In[42]:


#boxplots - revisited
#sns.boxplot(data=df_log[['log_backers_count','log_usd_pledged','log_usd_goal','log_timeline']])


# ### Remove outliers based on IQR on orginial data

# In[43]:


#df_merge[['backers_count', 'timeline','usd_pledged','usd_goal']].describe()


# In[47]:


# remove outliers based on IQR of Goal amount
#df_q =df_merge[['backers_count', 'timeline','usd_pledged','usd_goal']]

#Q1 = df_q.quantile(0.25)
#Q3 = df_q.quantile(0.75)
#IQR = Q3 - Q1

#df_IQR_ = df_q[~((df_q < (Q1 - 1.5 * IQR)) |(df_q > (Q3 + 1.5 * IQR))).any(axis=1)]
#df_IQR_.head()


# In[38]:


#subsetting data based on IQR
df_IQR = df_merge[ df_merge.index.isin(df_IQR_.index) ]
df_IQR.shape


# In[50]:


#IQR Describe
#df_IQR_.describe()


# In[51]:


#max goal
#cond_max = df_merge['usd_goal']== max(df_merge['usd_goal'])
#df_merge[['state','category','sub_category','country','launched_at','usd_goal', 'timeline','usd_pledged','backers_count']][cond_max]


# In[53]:


#max pledged
#cond_max = df_merge['usd_pledged']== max(df_merge['usd_pledged'])
#df_merge[['state','category','sub_category','country','launched_at','usd_goal', 'timeline','usd_pledged','backers_count']][cond_max]

#### Note : 
#For max Pledged amount, Goal amount is 88K USD which is very high. These are spcial cases where the factors involved depend on the popularity of Project host or the nature of the project for which they are raising funds.


# In[54]:


#max backers
#cond_max = df_merge['backers_count']== max(df_merge['backers_count'])
#df_merge[['state','category','sub_category','country','launched_at','usd_goal', 'timeline','usd_pledged','backers_count']][cond_max]


# In[55]:


#max timeline
#cond_max = df_merge['timeline']== max(df_merge['timeline'])
#df_merge[['state','category','country','sub_category','launched_at','usd_goal', 'timeline','usd_pledged','backers_count']][cond_max]


# ### Ascending and Descending order of Goal and Timeline

# In[45]:


#sorted data
features = ['state','category','sub_category','country','launched_at','usd_goal', 'timeline','usd_pledged','backers_count']
df_sort = df_merge[features]
#df_sort[['usd_goal', 'timeline','usd_pledged','backers_count']].describe()


# In[56]:


#Decreasing order of Goal amount : High to Low
df_sort.sort_values('usd_goal', ascending=False).head()


# In[57]:


#Increasing order of Goal amount : High to Low
df_sort.sort_values('usd_goal', ascending=True).head()


# #### Note :
# Goal amounts are very low less that 2 USD, which have high pledged amounts. However in these scenarios the aunthenticity of the cause is questionable. Such unusual cases are excluded in this analysis due to the limitated avaiable data for analysis. 

# In[46]:


#Decreasing order of Timeline in days : High to Low
df_sort.sort_values('timeline', ascending=False).head()


# In[58]:


#Increasing order of Timeline in days : High to Low
df_sort.sort_values('timeline', ascending=True).head()


# #### Inference
#  Our analysis of extremely high and low values of individual features, show that the outlier cases are unusual and for the purpose of building classification and regression models with limited data, we can retain 50% of the data falling in the IQR. 
# 

# ### Removing outliers based on IQR after log transformation

# In[86]:


# remove outliers based on IQR of Goal amount
df_q =df_log[['log_backers_count', 'log_timeline','log_usd_pledged','log_usd_goal']]

Q1 = df_q.quantile(0.25)
Q3 = df_q.quantile(0.75)
IQR = Q3 - Q1

df_log_IQR_ = df_q[~((df_log < (Q1 - 1.5 * IQR)) |(df_log > (Q3 + 1.5 * IQR))).any(axis=1)]
#df_log_IQR_.head()
df_log_IQR_[['log_backers_count', 'log_timeline','log_usd_pledged','log_usd_goal']].describe()


# In[92]:


df_IQR = df_log[ df_log.index.isin(df_log_IQR_.index)]
print(df_IQR.shape)
#df_IQR.head()
df_IQR.insert(0,'state',df_merge['state'][ df_merge.index.isin(df_log_IQR_.index)])
df_IQR.insert(5,'category',df_merge['category'][ df_merge.index.isin(df_log_IQR_.index)])
df_IQR.insert(6,'sub_category',df_merge['sub_category'][ df_merge.index.isin(df_log_IQR_.index)])
df_IQR.insert(7,'country',df_merge['country'][ df_merge.index.isin(df_log_IQR_.index)])
#df_IQR.drop('log_state', axis=1)
df_IQR.head()


# In[93]:


sns.boxplot(data=df_log_IQR_[['log_backers_count','log_usd_pledged','log_usd_goal','log_timeline']])


# ## 4.6. Bivariate Analysis  : Feature Relationship post Log transformation

# In[94]:


#separating failed and successful projects
#seperate Successful and Failed Projects
df_success = df_IQR.query('state == "successful" ' )
df_fail = df_IQR.query(' state == "failed" ')
print(df_success.shape)
print(df_fail.shape)


# ### 4.6.1 Scatter plot : Correlation

# In[95]:


df_data = pd.concat([df_success,df_fail])
print(df_data.shape)
#df_data.head()


# ### 1. Correlations and Subplots for Pledge Amount and Goal Amount - log transformed

# In[98]:


#Correlations and Subplots for Pledge Amount and Goal Amount - log transfomred

#subplot settings
fig, axes = plt.subplots(ncols=2,figsize=(15,6))

#fig.tight_layout()
fig.subplots_adjust(hspace=2)

#correlation matrix 

#df1=df_merge[["usd_pledged","usd_goal"]]
df1=df_data[["log_usd_pledged","log_usd_goal"]]
corr_matrix = df1.corr()


#scatterplot

sns.scatterplot(y="log_usd_pledged", x="log_usd_goal",hue="state", data= df_data, ax=axes[0])
axes[0].set_title('USD Pledged Amount vs Goal Amount')
axes[0].set_xlabel('USD Goal $')
axes[0].set_ylabel('USD Pledged $')

#correlleogram

sns.heatmap(corr_matrix, vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True, ax=axes[1])
axes[1].set_title('Correlation Headmap of Log transfromed data')

print(corr_matrix)


# ### 2. Correlations and Subplots for Pledged Amount and Timeline - log transformed

# In[99]:


#Correlations and Subplots for Pledge Amount and Timeline - log transformed

#subplot settings
fig, axes = plt.subplots(ncols=2,figsize=(15,6))

#fig.tight_layout()
fig.subplots_adjust(hspace=2)

#correlation matrix 
df2=df_data[["log_usd_pledged","log_timeline"]]
corr_matrix = df2.corr()

#scatterplot

sns.scatterplot(x="log_timeline", y="log_usd_pledged",hue="state", data= df_data, ax=axes[0])
axes[0].set_title('Log transformed : USD Pledged Amount (USD) vs Project Campaign Timeline')
axes[0].set_xlabel('Project Campaign Timeline')
axes[0].set_ylabel('USD Pledged $')

#correlleogram

sns.heatmap(corr_matrix, vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True, ax=axes[1])
axes[1].set_title('Correlation Headmap')

print(corr_matrix)


# ### 3. Correlations and Subplots for Backers and Timeline - log transformed

# In[101]:


#Correlations and Subplots for Number of Contributors and Timeline - log transformed

#subplot settings
fig, axes = plt.subplots(ncols=2,figsize=(15,6))

#fig.tight_layout()
fig.subplots_adjust(hspace=2)

#correlation matrix 
df3=df_data[["log_backers_count","log_timeline"]] 
corr_matrix = df3.corr()

#scatterplot

sns.scatterplot(x="log_timeline", y="log_backers_count",hue="state", data= df_data, ax=axes[0])
axes[0].set_title('Number of Contributors vs Project Campaign Timeline')
axes[0].set_xlabel('Project Campaign Timeline')
axes[0].set_ylabel('Number of Contributors')

#correlleogram

sns.heatmap(corr_matrix, vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True, ax=axes[1])
axes[1].set_title('Correlation Headmap')

print(corr_matrix)


# ### 4. Correlations and Subplots for Backers and Goal amount - log transformed

# In[102]:


#Correlations and Subplots for Goal Amount and Number of Contributors - log transformed

#subplot settings
fig, axes = plt.subplots(ncols=2,figsize=(15,6))

#fig.tight_layout()
fig.subplots_adjust(hspace=2)

#correlation matrix 
df4=df_data[["log_backers_count","log_usd_goal"]] 
corr_matrix = df4.corr()

#scatterplot

sns.scatterplot(x="log_usd_goal", y="log_backers_count",hue="state", data= df_data, ax=axes[0])
axes[0].set_title('USD Goal Amount (thousand) vs Number of Contributors')
axes[0].set_xlabel('USD Goal Amount in thousands')
axes[0].set_ylabel('Number of Contributors')

#correlleogram

sns.heatmap(corr_matrix, vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True, ax=axes[1])
axes[1].set_title('Correlation Headmap')

print(corr_matrix)


# ## 5. T-test for Successful and Failed Projects

# At this point , the volume of projects sucessful and failure is similar at 52% and 40% respectively . With hypothesis testing we want to understand if the features of the success and failure states are statistically different. 
# 
# In this test we are testing our hypothesis that Successful Projects have different Goal amounts, Pledged amounts, backers counts and timelines as compared to the Failed Projects. 

# In[103]:


#test

ttest, pval = ttest_ind(df_success['log_usd_goal'], df_fail['log_usd_goal'])
print("Ttest for goal amounts of Successful and Failed Projects")
print("p-value : ",pval)
print("ttest : ",ttest)


ttest, pval = ttest_ind(df_success['log_usd_pledged'], df_fail['log_usd_pledged'])
print("\nTtest for pledged amounts of Successful and Failed Projects")
print("p-value : ",pval)
print("ttest : ",ttest)


ttest, pval = ttest_ind(df_success['log_backers_count'], df_fail['log_backers_count'])
print("\nTtest for Backers count of Successful and Failed Projects")
print("p-value : ",pval)
print("ttest : ",ttest)

ttest, pval = ttest_ind(df_success['log_timeline'], df_fail['log_timeline'])
print("\nTtest for timeline of Successful and Failed Projects")
print("p-value : ",pval)
print("ttest : ",ttest)


# ### Inference :
# 
# The above t-tests show that Successful and Failed projects are statistically different. 
# 
# 1. The p-values of each t-statistic is less that the level of significance of 5%.  
# 2. The t-statistic is far from 0 for each of the tests indicating the means are different for each data slice. 

# ## 6. Machine Learning Techniques

# ### 6.1 Logistic Regression 
# 
# As per the scatter plots we can see the data doesnot have any linear pattern. Therefore we shall perform Logistic Regression as an intial step for determining the relationship between features and the project status outcome.

# In[104]:


# Logistic regression with cross validation
from sklearn import model_selection

train, test = train_test_split(df_data, test_size=0.2, random_state=0)

kfold = model_selection.KFold(n_splits = 10, random_state=0, shuffle=True)
model_kfold = LogisticRegression()
results_kfold = model_selection.cross_val_score(model_kfold, train[['log_usd_goal','log_timeline']], train['state'], cv=kfold)
print("Training Accuracy : %.2f%%" % (results_kfold.mean()*100.0))
#print(results_kfold)
#print(results_kfold.cv)

results_kfold_pred = model_selection.cross_val_predict(model_kfold, test[['log_usd_goal','log_timeline']], test['state'], cv=kfold)
#print("Accuracy : %.2f%%" % (results_kfold_pred.mean()*100.0))
#print(results_kfold_pred)
confusion_matrix_cv = pd.crosstab( results_kfold_pred, test['state'], rownames=['Predicted'], colnames=['Actuals'])
print(confusion_matrix_cv)
print('Testing Accuracy: ',metrics.accuracy_score(test['state'], results_kfold_pred))
#print("Misclassification on Test data : ", metrics.classification_report )

sns.heatmap(confusion_matrix_cv, annot=True)


# #### Inference & Summary of Logistic Regression
# 
# As per the results of logistic regression, accuracy of prediction of successful projects is 85.8% which is good.
# 
# However the misclassification of failed projects is 60% which is high. 
# 
# Logistic regression based on the goal and timeline set by the project owners is not a model with high predictive power. We will need to make some transformations and apply more effective machine learning techniques to improve the accuracy of prediction and reduce the misclassification rate. 

# ### 6.2 Linear Regression
# 
# Predicting Backers Count based on Goal and Timeline

# In[110]:


# linear regression to predict backers count

train, test = train_test_split(df_data, test_size=0.2, random_state=0)

backers_model = sm.OLS(train['log_usd_pledged'],train[['log_usd_goal','log_timeline']]).fit()
#backers_model = sm.OLS(df_data['backers_count'],X_poly).fit()

print(backers_model.params)

print(backers_model.summary())
#backers_lm.

pred_backers = backers_model.predict(test[['log_usd_goal','log_timeline']])

# RMSE
print(np.sqrt(metrics.mean_squared_error(test['log_usd_pledged'], pred_backers)))

#accuracy
#print(r2_score(test['log_backers_count'],pred_backers))


# In[112]:


metrics.r2_score(test['log_usd_pledged'],pred_backers)


# We cannot use Linear regression to predict backers count because the relationship between usd_goal and backers count is not linear

# ### 6.3 Decision Tree

# In[113]:


#Building Tree
from sklearn.tree import DecisionTreeClassifier

train, test = train_test_split(df_data, test_size=0.2, random_state=0)

X= train[['log_usd_goal','log_timeline']]
y= train['state']

dtreeclass = DecisionTreeClassifier()

dtreeclass = dtreeclass.fit(X, y)

X_test = test[['log_usd_goal','log_timeline']]
y_test = test['state']

y_pred = dtreeclass.predict(X_test)

print("Accuracy : ",metrics.accuracy_score(y_test,y_pred))


# In[115]:


#import IPython
#from sklearn.tree import export_graphviz
#from sklearn.externals.six import StringIO
#from IPython.display import Image
#import pydotplus

#dot_data = StringIO()
#export_graphviz(dtreeclass, out_file = dot_data, filled=True, rounded=True, special_characters=True, feature_names=['log_usd_goal','log_timeline'], class_names=['Successful','Failed'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#graph.write_png('tree.png')
#Image(graph.create_png())


# In[98]:


#Decision Tree for predicting backers count



train, test = train_test_split(df_data, test_size=0.2, random_state=0)
#kfold = model_selection.KFold(n_splits = 10, random_state=0, shuffle=True)

X= train[['usd_goal','timeline']]
y= train['backers_count']

reg_tree = DecisionTreeRegressor(random_state=0)
reg_tree.fit(X,y)

#score= np.mean(cross_val_score(reg_tree,test[['usd_goal','timeline']],test['backers_count'],scoring='mean_squared_error',cv=kfold, n_jobs=1 ))

score= cross_val_score(reg_tree,test[['usd_goal','timeline']],test['backers_count'],cv=kfold, n_jobs=1, scoring )

print(score)


# ### 6.4 KNearest Neighbors Model

# In[116]:


# setting X and Y
feature =["state"]
y = df_data[feature]
feature =['log_usd_goal','log_timeline']
X = df_data[feature]

#spliting data into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# KNN for 10 neighhbours
classifier = KNeighborsClassifier(n_neighbors=25)
classifier.fit(X_train, y_train)

# predict for test data
y_pred = classifier.predict(X_test)

# accuracy of prediction
print(metrics.accuracy_score(y_test, y_pred))

#Classification matrix
print(confusion_matrix( y_pred,y_test, labels=['successful','failed'] ))
#confusion_matrix_knn = pd.crosstab( y_pred, y_test, rownames=['Predicted'], colnames=['Actuals'])
#print(confusion_matrix_knn)
print(classification_report( y_pred,y_test))



# In[136]:


#optimal k

# try K=1 through K=25 and record testing accuracy
k_range = range(1, 25)

# We can create Python dictionary using [] or dict()
scores = []

# We use a loop through the range 1 to 26
# We append the scores in the dictionary
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

#print(scores)
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')


# ### Prediction based on recent data >2015 

# In[119]:


#filter records
df_data.insert(1,'lauchedat_year',df_merge['lauchedat_year'][df_merge.index.isin(df_data.index)])
df_data.head()

# Extracting records from 2015 for prediciont
cond_past = df_data['lauchedat_year'] < 2015
df_history = df_data[cond_past]
print("Past data Year <= 2014")
print(df_history.shape)
df_recent = df_data[~cond_past]
print("Recent data Year > 2014")
print(df_recent.shape)


# ### 6.5 Random Forest

# In[124]:


# Creating dummy variables
dummy_category=pd.get_dummies(df_recent['category'])
dummy_category.head()

#creating a dataset for Random Forest
df_forest=df_recent[['log_usd_goal','log_timeline','state']]
df_forest=pd.concat([df_forest, dummy_category.reindex(df_forest.index)], axis=1)
#print(df_forest.shape)
#df_forest.head()

df_forest_nostate=df_forest[['log_usd_goal','log_timeline']]
df_forest_nostate=pd.concat([df_forest_nostate, dummy_category.reindex(df_forest_nostate.index)], axis=1)
print(df_forest_nostate.shape)
df_forest_nostate.head()


# In[127]:


#Model Evaluation

X = df_forest_nostate
y = df_forest['state']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 100)

#Fit the model
rf=RandomForestClassifier(n_estimators=1000,max_depth=None,max_features='auto')

#Model Evaluation
rf.fit(X_train, y_train)
yhat=rf.predict(X_test)
accuracy_score(y_test,yhat)


# ### 6.6 Gradient Boosting

# In[133]:


#model Building
from numpy import loadtxt
from xgboost import XGBClassifier

y = df_forest['state']
X = df_forest_nostate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
model = XGBClassifier(n_estimators=1000)
model.fit(X_train, y_train)


# In[134]:


# Model evaluation
y_pred = model.predict(X_test)
#predictions = [round(value) for value in y_pred]
# evaluate predictions
#accuracy = accuracy_score(y_test, predictions)
print("Accuracy:",accuracy_score(y_test, y_pred))


# ## 7. Conclusion : 
# 
# ### Recomendations for Audience
# 1. Project Goal Amount below 2,000,000 USD is highly recommended.
# 2. Practically, lower goal amounts are more successfull compared to extremely higher amounts
# 3. Projects could be launched from Asian countries like Hong Kong, Singapore, European countries like Great Britain and United States have more visibility. 
# 4. An average timeline of 30 days would help raise more funds.
# 5. Projects launched at the beginning of the year are more likely to be successful.
# 6. Projects from creative categories like Dance, Comics , Music are more likely to succeed.
# 
# ### Challenges of Analysis
# 1. The kickstarter data over the past 9 years is highly skewed due to the presence of outliers.
# 2. Unusual data such has projects trying to raise funds for 1 USD fetching 1000 times higher pledged amounts leads to questions on the authenticity of the objective of the project.
# 3. Projects with higher goal amounts achiveing  overwhelming funds can be attributed to the influence of social media popularity.
# 
# ### Model Evaluation
# 1. The features of the data being analyized is non linear and in the form of clusters.
# 2. Due to the absence of linear relationship, regresssion using simple algorithms is a challenge.
# 3. There is a lot of overlap between succesful and failed proejcts data.
# 4. Classification using numeric parameters is not the strongest with an accuracy of 70% by KNN. 
# 5. Predicting using recent data reduces the accuracy due to the votatile nature of the trends of projects on kicstarter.
# 

# #### References

# 
# 1. Group and count : https://stackoverflow.com/questions/47320572/pandas-groupby-and-count
# 2. Group and count in dataframe : https://stackoverflow.com/questions/10373660/converting-a-pandas-groupby-output-from-series-to-dataframe
# 3. Pivot table : https://stackoverflow.com/questions/34642180/convert-categorical-data-into-various-columns-for-plotting-in-pandas
# 4. Drop multilevel index : https://stackoverflow.com/questions/43756052/transform-pandas-pivot-table-to-regular-dataframe
# 5. Fill na with 0 : https://stackoverflow.com/questions/13295735/how-can-i-replace-all-the-nan-values-with-zeros-in-a-column-of-a-pandas-datafra
# 6. Stacked plot : https://stackoverflow.com/questions/56251848/how-do-i-make-pandas-catagorical-stacked-bar-chart-scale-to-100
# 7. Hypothesis Testing : https://towardsdatascience.com/hypothesis-testing-in-machine-learning-using-python-a0dc89e169ce
# 8. Split data into test and train : https://stackoverflow.com/questions/24147278/how-do-i-create-test-and-train-samples-from-one-dataframe-with-pandas
# 9. Logistic Regression : https://datatofish.com/logistic-regression-python/
# 10. Yearly Time series : https://stackoverflow.com/questions/50997339/convert-daily-data-in-pandas-dataframe-to-monthly-data
# 11. Time Series https://stackoverflow.com/questions/33191857/how-can-i-convert-from-pandas-dataframe-to-timeseries
# 12. Time Series https://stackoverflow.com/questions/23859840/python-aggregate-by-month-and-calculate-average
# 13. Resampling https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.resample.html
