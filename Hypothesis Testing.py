#!/usr/bin/env python
# coding: utf-8

# T Test is of two types.
# 1.1-Sample T test.
# 2.2-Sample T test.

# In[2]:


ages=[12,10,0,45,76,24,42,12,23,43,23,56,98,78,56,44,18,22,26,37,9,29,27,75,99,42,15,17,21,55,67,34,32,47]


# In[3]:


len(ages)


# In[4]:


import numpy as np
age_mean=np.mean(ages)
print(age_mean)


# In[5]:


sample_size=10
age_sample=np.random.choice(ages,sample_size)


# In[6]:


age_sample


# In[8]:


from scipy.stats import ttest_1samp


# In[9]:


ttest,p_value=ttest_1samp(age_sample,30)


# In[10]:


print(p_value)


# In[11]:


if p_value>0.05:
    print("Accept Null Hypothesis")
else:
    print("Reject Null Hypothesis")


# # Some More Example:
# consider the age of students in college and in a class room.

# In[12]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import math
np.random.seed(6)
school_ages=stats.poisson.rvs(loc=18,mu=35,size=1500)
classA_ages=stats.poisson.rvs(loc=18,mu=30,size=60)


# In[13]:



classA_ages.mean()


# In[14]:


school_ages.mean()


# In[15]:



ttest,p_value=stats.ttest_1samp(a=classA_ages,popmean=school_ages.mean())


# In[16]:


ttest


# In[17]:


p_value


# In[18]:


if p_value < 0.05:    # alpha value is 0.05 or 5%
    print(" we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")


# # Two-sample T-test With Python¶
# The Independent Samples t Test or 2-sample t-test compares the means of two independent groups in order to determine whether there is statistical evidence that the associated population means are significantly different. The Independent Samples t Test is a parametric test. This test is also known as: Independent t Test

# In[19]:



np.random.seed(12)
ClassB_ages=stats.poisson.rvs(loc=18,mu=33,size=60)
ClassB_ages.mean()


# In[20]:


ClassB_ages


# In[22]:


_,p_value=stats.ttest_ind(a=classA_ages,b=ClassB_ages,equal_var=False)


# In[24]:


p_value


# In[26]:


if p_value < 0.05:    # alpha value is 0.05 or 5%
    print(" we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")


# # Paired T-test With Python
# When you want to check how different samples from the same group are, you can go for a paired T-test

# In[27]:


weight1=[25,30,28,35,28,34,26,29,30,26,28,32,31,30,45]
weight2=weight1+stats.norm.rvs(scale=5,loc=-1.25,size=15)


# In[28]:


print(weight1)
print(weight2)


# In[29]:



weight_df=pd.DataFrame({"weight_10":np.array(weight1),
                         "weight_20":np.array(weight2),
                       "weight_change":np.array(weight2)-np.array(weight1)})


# In[30]:


weight_df


# In[31]:


_,p_value=stats.ttest_rel(a=weight1,b=weight2)


# In[32]:


p_value


# In[33]:


if p_value < 0.05:    # alpha value is 0.05 or 5%
    print(" we are rejecting null hypothesis")
else:
    print("we are accepting null hypothesis")


# # Correlation

# In[34]:


df=sns.load_dataset('iris')


# In[35]:


df


# In[36]:


df.corr()


# In[37]:


df.shape


# In[38]:


sns.pairplot(df)


# # Chi-Square Test

# In[1]:


import scipy.stats as stats


# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[3]:


data=sns.load_dataset('tips')


# In[4]:


data.head()


# In[5]:


table=pd.crosstab(data['sex'],data['smoker'])


# In[6]:


table


# In[7]:


observed_value=table.values


# In[10]:


print("observed values:\n",observed_value  ) 


# In[12]:


val=stats.chi2_contingency(table)


# In[13]:


val


# In[14]:


expected_value=val[3]


# In[15]:


expected_value


# In[19]:


no_of_rows=len(table.iloc[0:2,0])
no_of_cols=len(table.iloc[0,0:2])
dof=(no_of_rows-1)*(no_of_cols-1)
print("degree of freedom :",dof)
alpha=0.05


# In[20]:


from scipy.stats import chi2
chi_squares=sum([(o-e)**2/e for o,e in zip(observed_value,expected_value)])
chi_square_statistics=chi_squares[0]+chi_squares[1]


# In[21]:


chi_square_statistics


# In[24]:


critical_value=chi2.ppf(q=1-alpha,df=dof)
critical_value


# In[25]:


#Another way-By calculating the p-value
p_value=1-chi2.cdf(x=chi_square_statistics,df=dof)
print('P_value:',p_value)
print('significance level:',alpha)
print('Degree of freedom:',dof)


# In[26]:


if  chi_square_statistics>=critical_value:
    print('Reject null hypothesis- There is a relationship')
else:
    print('Reject alternative hypothesis- There is no relationship')


# In[27]:


if p_value<alpha:
    print('Accept alternative hypothesis')
else:
    print('reject alternative hypothesis')


# In[ ]:




