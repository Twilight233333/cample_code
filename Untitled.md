```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# import data
data = pd.read_csv('/Users/jiaxuan/Desktop/oil_curse.csv')

data
data.head()

# create coefficient
data['const'] = 1
data.head()

# drop NA value
data = data.dropna()

reg1 = sm.OLS(endog=data['Arty1'], exog=data[['const', 'Aras1', 'Aodf1', 'Bcop1', 'Bcor1','Bopd1','Bcoe1']])
results1 = reg1.fit()
print(results1.summary())

# convert oil varables (production, reserves etc) into log version
data['Bcop1'] = np.log(data['Bcop1'])
data['Bcor1'] = np.log(data['Bcor1'])
data['Bopd1'] = np.log(data['Bopd1'])
data['Bcoe1'] = np.log(data['Bcoe1'])
data.head()

reg2 = sm.OLS(endog=data['Arty1'], exog=data[['const', 'Aras1', 'Aodf1', 'Bcop1', 'Bcor1','Bopd1','Bcoe1']])
results2 = reg2.fit()
print(results2.summary())

# generate odf1*ras1
data['odfXras'] = data['Aodf1']*data['Aras1']
data.head()

reg3 = sm.OLS(endog=data['Arty1'], exog=data[['const', 'Aras1', 'Aodf1', 'odfXras',  'Bcop1', 'Bcor1','Bopd1','Bcoe1']])
results3 = reg3.fit()
print(results3.summary())

# plot scatter of Arty1 and Aodf1
plt.scatter(data['Aodf1'], data['Arty1'], label='Data Points', color='blue')
fit = np.polyfit(data['Aodf1'], data['Arty1'], 1)
fit_fn = np.poly1d(fit)
plt.plot(data['Aodf1'], fit_fn(data['Aodf1']), color='red', label='Fit Line')
plt.legend()

plt.xlabel('Oil extraction difficulty index')
plt.ylabel('Democratization index')
plt.title('Large oil rents and a degree of democratization')
plt.show()
```
