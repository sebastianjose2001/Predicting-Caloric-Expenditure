# Importing some important libraries and get data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

calories = pd.read_csv('/content/calories.csv')
exercise=pd.read_csv('/content/exercise.csv')

df=exercise.merge(calories,on='User_ID')


# Visualizing Duration and Calories
df.plot(kind='scatter',x ="Duration" , y = 'Calories' ,alpha=0.1)

# Convert the 'Gender' column in the dataframe to a categorical type 
df["Gender"] = df["Gender"].astype("category").cat.codes


# Calculate the correlation matrix
cor_df = df.corr()
cor_df['Calories'].sort_values(ascending=False)
corr=df.corr()

# Heatmap
sns.heatmap(corr,cbar=True,square=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap='Blues')


x = df.drop(columns=['User_ID','Calories'])
y = df['Calories']

# Splitting the data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

# Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train['Gender'] = le.fit_transform(X_train['Gender'])
X_test['Gender'] = le.fit_transform(X_test['Gender'])
X_test

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.model_selection import cross_val_score


# Linear Regression
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(X_train,y_train)
model1.score(X_test,y_test)

y_pred=model1.predict(X_test)
y_pred

model1.score(X_train,y_train)

print('MSE is:',mean_squared_error(y_test,y_pred))
print('MAE is:',mean_absolute_error(y_test,y_pred))
print('RMSE is:',np.sqrt(mean_squared_error(y_test,y_pred)))
print('R2 score is:',r2_score(y_test,y_pred))

cv_scores = cross_val_score(model1, x, y, cv=5)
print('CV scores:', cv_scores)
print('Mean CV score:', cv_scores.mean())


# Decision Tree
from sklearn.tree import DecisionTreeRegressor
model2=DecisionTreeRegressor()
model2.fit(X_train,y_train)
model2.score(X_train,y_train)

y_pred2=model2.predict(X_test)
model2.score(X_train,y_train)

print('MSE is:',mean_squared_error(y_test,y_pred2))
print('MAE is:',mean_absolute_error(y_test,y_pred2))
print('RMSE is:',np.sqrt(mean_squared_error(y_test,y_pred2)))
print('R2 score is:',r2_score(y_test,y_pred2))

cv_scores = cross_val_score(model2, x, y, cv=5)
print('CV scores:', cv_scores)
print('Mean CV score:', cv_scores.mean())


# Random Forest
from sklearn.ensemble import RandomForestRegressor
model3= RandomForestRegressor()
model3.fit(X_train,y_train)
model3.score(X_test,y_test)

y_pred3=model3.predict(X_test)
model3.score(X_train,y_train)

print('MSE is:',mean_squared_error(y_test,y_pred3))
print('MAE is:',mean_absolute_error(y_test,y_pred3))
print('RMSE is:',np.sqrt(mean_squared_error(y_test,y_pred3)))
print('R2 score is:',r2_score(y_test,y_pred3))

cv_scores = cross_val_score(model3, x, y, cv=5)
print('CV scores:', cv_scores)
print('Mean CV score:', cv_scores.mean())


# Support Vector Machine
from sklearn import svm
model4 = svm.SVR(kernel='linear')
model4.fit(X_train, y_train)
model4.score(X_test, y_test)

y_pred4 = model4.predict(X_test)
model4.score(X_train, y_train)

print('MSE is:', mean_squared_error(y_test, y_pred4))
print('MAE is:', mean_absolute_error(y_test, y_pred4))
print('RMSE is:', np.sqrt(mean_squared_error(y_test, y_pred4)))
print('R2 score is:', r2_score(y_test, y_pred4))

cv_scores = cross_val_score(model4,x,y,cv=5)
print('CV scores:',cv_scores)
print('Mean CV score:',cv_scores.mean())


# Ridge Regression
from sklearn.linear_model import Ridge
model5 = Ridge(alpha=0.1)
model5.fit(X_train, y_train)
model5.score(X_test, y_test)

y_pred5 = model5.predict(X_test)
model5.score(X_train, y_train)

print('MSE is:', mean_squared_error(y_test, y_pred5))
print('MAE is:', mean_absolute_error(y_test, y_pred5))
print('RMSE is:', np.sqrt(mean_squared_error(y_test, y_pred5)))
print('R2 score is:', r2_score(y_test, y_pred5))

cv_scores = cross_val_score(model5, x, y, cv=5)
print('CV scores:', cv_scores)
print('Mean CV score:', cv_scores.mean())


# Lasso Regression
from sklearn.linear_model import Lasso
model6 = Lasso(alpha=0.1)
model6.fit(X_train, y_train)
model6.score(X_test, y_test)

y_pred6 = model6.predict(X_test)
model6.score(X_train, y_train)

print('MSE is:', mean_squared_error(y_test, y_pred6))
print('MAE is:', mean_absolute_error(y_test, y_pred6))
print('RMSE is:', np.sqrt(mean_squared_error(y_test, y_pred6)))
print('R2 score is:', r2_score(y_test, y_pred6))

cv_scores = cross_val_score(model6, x, y, cv=5)
print('CV scores:', cv_scores)
print('Mean CV score:', cv_scores.mean())


# Using subplots to compare the metric values of each algorithm
model_names=["Linear Regression","Decision Tree","Random Forest","SVR","Ridge Regression","Lasso Regression"]
mse_scores=[138.12,30.04,9.35,153.73,138.12,138.47]
mae_scores=[8.48,3.48,1.81,8.37,8.48,8.49]
rmse_scores=[11.75,5.48,3.06,12.40,11.75,11.77]
r_sq_values=[0.97,0.99,0.99,0.96,0.96,0.97]
cv_scores=[0.97,0.99,0.99,0.96,0.96,0.97]

x=np.arange(len(model_names))
width=0.2
fig, ax = plt.subplots(figsize=(10, 6))
rects1=ax.bar(x - width, mse_scores,width,label='MSE')
rects2=ax.bar(x, mae_scores,width,label='MAE')
rects3=ax.bar(x + width, rmse_scores,width,label='RMSE')
rects4=ax.bar(x + 2*width, r_sq_values,width,label='R2 Score')
rects5=ax.bar(x + 2*width, cv_scores,width,label='CV Score')
ax.set_ylabel('Scores')
ax.set_title('Model Comparison')
ax.set_xticks(x + width/2)
ax.set_xticklabels(model_names)
ax.legend()

ax.bar_label(rects1, padding=6)
ax.bar_label(rects2, padding=6)
ax.bar_label(rects3, padding=6)
ax.bar_label(rects4, padding=6)
ax.bar_label(rects5, padding=6)

fig.tight_layout()
plt.show()