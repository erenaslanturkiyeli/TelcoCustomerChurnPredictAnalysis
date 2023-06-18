import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

telecom_cust = pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')

telecom_cust.head(10)

print(telecom_cust.shape)

print(telecom_cust.columns.values)

print(telecom_cust.describe())

# Checking the data types of all columns
print(telecom_cust.dtypes)


# Converting Total Charges to a numerical data type.
telecom_cust.TotalCharges = pd.to_numeric(telecom_cust.TotalCharges, errors='coerce')
print(telecom_cust.isnull().sum())


#Removing missing values 
telecom_cust.dropna(inplace = True)
#Remove customer IDs from the data set
df2 = telecom_cust.drop(["customerID"], axis=1)
#Converting the predictor variable in a binary numeric variable
df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df2['Churn'].replace(to_replace='No',  value=0, inplace=True)

#Convert all the categorical variables into dummy variables
df_dummies = pd.get_dummies(df2)
print(df_dummies.head())


#Get Correlation of "Churn" with other variables:
plt.figure(figsize=(15,8))
df_dummies.corr()['Churn'].sort_values(ascending = False).plot(kind='bar')

## Aylık sözleşmeler(Month to month contracts), çevrimiçi güvenlik(online security) ve teknik desteğin(tech support) olmaması, müşteri kaybıyla pozitif olarak ilişkili görünüyor. 
## Bununla birlikte, kullanma süresi(tenure), iki yıllık sözleşmeler(two year contracts), müşteri kaybı ile negatif ilişkili görünmektedir.

sns.heatmap(df_dummies.corr())

print(df_dummies.head(5))

print(df_dummies.dtypes)

## Data Exploration


## Gender Distribution
colors = ['#0A64F8','#F80A39']
gnd = (telecom_cust['gender'].value_counts()*100.0 /len(telecom_cust)).plot(kind='bar',
                                                                           stacked = True,
                                                                          rot = 0,
                                                                          color = colors)
gnd.set_ylabel('% Customers')
gnd.set_xlabel('Gender')
gnd.set_title('Gender Distribution')
gender_Female_count = df_dummies.gender_Female.sum()
gender_Male_count = df_dummies.gender_Male.sum()
gender_Female_perc = (gender_Female_count/ (gender_Male_count + gender_Female_count))*100
gender_Male_perc = (gender_Male_count/ (gender_Male_count + gender_Female_count))*100
print('Gender Female: %s' % gender_Female_count)
print('Gender Male: %s' % gender_Male_count)
print('Gender Female Percentage: '+ str(round(gender_Female_perc,2)))
print('Gender Male Percentage: '+ str(round(gender_Male_perc,2)))
plt.show()

## Veri setimizdeki müşterilerin yaklaşık yarısı erkek, diğer yarısı kadındır.


## % Senior Citizens
colorss = ['#B08E23','#23B098']
snr = (telecom_cust['SeniorCitizen'].value_counts()*100.0 /len(telecom_cust))\
.plot.pie(autopct='%.1f%%', labels = ['No', 'Yes'],figsize =(5,5), colors=colorss)                                                                           
snr.set_title('% of Senior Citizens')
plt.show()

## Müşterilerin sadece %16'sı yaşlıdır. Dolayısıyla verilerdeki müşterilerimizin çoğu genç insanlardır.


## Dependents Distribution
colors = ['#7423B0','#CD6BCD']
dpdnt = (telecom_cust['Dependents'].value_counts()*100.0 /len(telecom_cust)).plot(kind='bar',
                                                                           stacked = True,
                                                                          rot = 0,
                                                                          color = colors)
dpdnt.set_ylabel('% Customers')
dpdnt.set_xlabel('Dependents')
dpdnt.set_title('Dependents Distribution')
dependents_no_count = df_dummies.Dependents_No.sum()
dependents_yes_count = df_dummies.Dependents_Yes.sum()
dependents_no_perc = (dependents_no_count/ (dependents_no_count + dependents_yes_count))*100
dependents_yes_perc = (dependents_yes_count/ (dependents_no_count + dependents_yes_count))*100
print('No: %s' % dependents_no_perc)
print('Yes: %s' % dependents_yes_perc)
print('No Percentage: '+ str(round(dependents_no_perc,2)))
print('Yes Percentage: '+ str(round(dependents_yes_perc,2)))

plt.show()

## Toplam müşterilerin yalnızca %30'unun bakmakla yükümlü olduğu kişiler vardır.


## Partner Distribution
colors = ['#F06262','#D68019']
part = (telecom_cust['Partner'].value_counts()*100.0 /len(telecom_cust)).plot(kind='bar',
                                                                           stacked = True,
                                                                          rot = 0,
                                                                          color = colors)
part.set_ylabel('% Customers')
part.set_xlabel('Partner')
part.set_title('Partner Distribution')
partner_no_count = df_dummies.Partner_No.sum()
partner_yes_count = df_dummies.Partner_Yes.sum()
partner_no_perc = (partner_no_count/ (partner_no_count + partner_yes_count))*100
partner_yes_perc = (partner_yes_count/ (partner_no_count + partner_yes_count))*100
print('No: %s' % partner_no_perc)
print('Yes: %s' % partner_yes_perc)
print('No Percentage: '+ str(round(partner_no_perc,2)))
print('Yes Percentage: '+ str(round(partner_yes_perc,2)))
plt.show()

## Müşterilerin yaklaşık %50'sinin bir eşi vardır.

##Tenure
plt.hist(telecom_cust['tenure'], width=1.5, bins=36, color="#30AFAF", edgecolor="black")
plt.ylabel('# of Customers')
plt.xlabel('Tenure (months)')
plt.title('# of Customers by their tenure')
plt.show()

## Yukarıdaki histograma bakılırsa, pek çok müşterinin telekom şirketinde yalnızca bir ay kaldığı, pek çoğunun ise yaklaşık 72 aydır orada olduğunu görebiliriz.

## Contracts
color=["#3064AF", "#7947A5", "#C23636"]
cont = telecom_cust['Contract'].value_counts().plot(kind = 'bar',rot = 0, width = 0.3, color=color)
cont.set_ylabel('# of Customers')
cont.set_title('# of Customers by Contract Type')
plt.show()

## Bu grafikte de görebileceğimiz gibi, müşterilerin çoğu month-to-month (1 yıldan az)'dır.

## Churn Rate
colors = ['#36C25A','#E4512B']
chu_rate = (telecom_cust['Churn'].value_counts()*100.0 /len(telecom_cust)).plot(kind='bar',
                                                                          rot = 0,
                                                                          color = colors,
                                                                         figsize = (8,6))
chu_rate.set_ylabel('% Customers')
chu_rate.set_xlabel('Churn')
chu_rate.set_title('Churn Rate')
plt.show()

## Veri incelendiğinde, müşterilerin %74'ünün aboneliğini bırakmadığı görülmektedir.

## Churn by Contract Type
colors = ['#3689C2','#9E36C2']
contract_churn = telecom_cust.groupby(['Contract','Churn']).size().unstack()

cont_chu = (contract_churn.T*100.0 / contract_churn.T.sum()).T.plot(kind='bar',
                                                                width = 0.3,
                                                                stacked = True,
                                                                rot = 0, 
                                                                figsize = (10,6),
                                                                color = colors)
cont_chu.legend(loc='best',prop={'size':14},title = 'Churn')
cont_chu.set_ylabel('% Customers',size = 14)
cont_chu.set_title('Churn by Contract Type',size = 14)

# Code to add the data labels on the stacked bar chart
for p in cont_chu.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    cont_chu.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                color = 'white',
               weight = 'bold',
               size = 14)

## Korelasyon grafiğinde gördüğümüze benzer şekilde, month-to-month olan müşterilerin müşteri kaybı oranı çok daha yüksektir.

## Churn by Seniority
colors = ['#C23660','#C26936']
seniority_churn = telecom_cust.groupby(['SeniorCitizen','Churn']).size().unstack()

sen_chu = (seniority_churn.T*100.0 / seniority_churn.T.sum()).T.plot(kind='bar',
                                                                width = 0.2,
                                                                stacked = True,
                                                                rot = 0, 
                                                                figsize = (8,6),
                                                                color = colors)
sen_chu.legend(loc='center',prop={'size':14},title = 'Churn')
sen_chu.set_ylabel('% Customers')
sen_chu.set_title('Churn by Seniority Level',size = 14)

# Code to add the data labels on the stacked bar chart
for p in sen_chu.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    sen_chu.annotate('{:.0f}%'.format(height), (p.get_x()+.25*width, p.get_y()+.4*height),
                color = 'white',
               weight = 'bold',size =14)

## Yaşlı vatandaşların, genç vatandaşlara göre kayıp oranı daha fazla olduğu görülmektedir.

##Linear Regression Model

features = df_dummies.drop("Churn", axis = 1)
target = df_dummies["Churn"]

# Create Train & Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=101)

from sklearn.linear_model import LinearRegression

lin_reg_mod = LinearRegression().fit(X_train, y_train)
y_predict_lin_mod = lin_reg_mod.predict(X_test)

from sklearn.metrics import mean_squared_error
rmse_lin_reg_mod = np.sqrt(mean_squared_error(y_test, y_predict_lin_mod, squared = False))
rmse_lin_reg_mod

## Decision Tree

features = df_dummies.drop("Churn", axis = 1)
target = df_dummies["Churn"]

# Create Train & Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=101)

from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state = 4)
dt_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(random_state = 4)
decision_tree = dt_model.fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)

from sklearn.metrics import mean_squared_error
rmse_dt_mod = mean_squared_error(y_test, y_pred_dt, squared = False)
rmse_dt_mod

## Logistic Regression Model

features = df_dummies.drop("Churn", axis = 1)
target = df_dummies["Churn"]

# Create Train & Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=101)

# Running logistic regression model
from sklearn.linear_model import LogisticRegression
log_reg_mod = LogisticRegression().fit(X_train, y_train)
y_predict_log_mod = log_reg_mod.predict(X_test)

from sklearn.metrics import mean_squared_error
rmse_log_mod = mean_squared_error(y_test, y_predict_log_mod, squared = False)
rmse_log_mod

### Bildiğimiz gibi mean squared error değeri küçüldükçe modelin doğruluğu artar. Bu bağlamda yukarıda incelediğimiz Linear Regression Model, Decision Tree ve Logistic Regression Model'leri arasında doğruluk değeri en düşük olan model Logistic Regression Model'dir.
###    - Linear Regression Model accuracy : 0.61
###    - Decision Tree accuracy : 0.53
###    - Logistic Regression Model accuracy : 0.44
