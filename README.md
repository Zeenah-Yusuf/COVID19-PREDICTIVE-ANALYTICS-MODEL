# COVID19-PREDICTIVE-ANALYTICS-MODEL
The CPAM is a predictive model that analyze and predict covid 19 trend to assist public health in data analysis for future covid 19 trend.
COVID-19 Predictive Modeling Technical Report

**Introduction**
This report presents the findings of a predictive modeling project aimed at forecasting COVID-19 trends using historical data.

**Methodology**
The project utilized the COVID-19 dataset, which includes COVID-19 case counts, demographic data, and various health metrics.

**Results**
The project developed three predictive models: Random Forest Classifier, Hist Gradient Boosting Classifier, and Support Vector Machine (SVM).

**Model Evaluation**
The models were evaluated using accuracy, precision, recall, F1-score, and RMSE.

**Conclusion**
The project demonstrated the effectiveness of machine learning models in predicting COVID-19 trends.


**Recommendations**
The findings of this project can inform public health policies, anticipate future outbreaks, and improve health resource allocation.

**Codes and scripts:**

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,  HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("covid_19_clean_complete.csv.csv", parse_dates=["Date"])
df.head(6)

# Remove duplicates
df.drop_duplicates(inplace=True)
df

df.isnull().sum()

df.shape

try:
    df.drop("Province/State", axis=1, inplace=True)
except KeyError:
    print("Column 'Province/State'not found.")
df
df['Date'] = pd.to_datetime(df['Date'])
df

# check to see if there are duplicate values
df_duplicate = df.duplicated().sum()
df_duplicate

# Calculate the Mortality and Recovery rate.
df["Mortality_rate"] = df["Deaths"]/df["Confirmed"]
df["Recovery_rate"] = df["Recovered"]/df["Confirmed"]

df.head(10), df.describe()

df = df.fillna(0)

df.head()

df.describe()

# checking for negative values in the active column because we can tell from the description above that there aren't any in the "Confirmed"
# "Deaths" and "Recovered" columns
# Check for negative values in the Active column
negative_active_cases = df[df['Active'] < 0]

# Display the rows with negative Active cases, if any
print(f"Number of rows with negative 'Active' cases: {negative_active_cases.shape[0]}")
negative_active_cases.head(18)
df
df['Case_Fatality_Rate'] = df['Deaths'] / df['Confirmed']
df['Recovery_Rate'] = df['Recovered'] / df['Confirmed']
df

#perform EDA

plt.figure(figsize=(10,6))
sns.lineplot(x='Date', y='Confirmed', data=df)
plt.title('COVID-19 Confirmed Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.show()

#visualize data

plt.figure(figsize=(10,6))
sns.barplot(x='Country/Region', y='Confirmed', data=df)
plt.title('COVID-19 Confirmed Cases by Country/Region')
plt.xlabel('Country/Region')
plt.ylabel('Confirmed Cases')
plt.show()

#Analyze WHO region factors

plt.figure(figsize=(10,6))
sns.barplot(x='WHO Region', y='Confirmed', data=df)
plt.title('COVID-19 Confirmed Cases by WHO Region')
plt.xlabel('WHO Region')
plt.ylabel('Confirmed Cases')
plt.show()

# Calculate Mortality_rate and Recovery_rate for each entry

df['Mortality_rate'] = (df['Deaths'] / df['Confirmed']) * 100
df['Recovery_rate'] = (df['Recovered'] / df['Confirmed']) * 100
df

MODEL OUTPUTS

	Province/State	Country/Region	Lat	Long	Date	Confirmed	Deaths	Recovered	Active	WHO Region
0	NaN	Afghanistan	33.93911	67.709953	2020-01-22	0	0	0	0	Eastern Mediterranean
1	NaN	Albania	41.15330	20.168300	2020-01-22	0	0	0	0	Europe
2	NaN	Algeria	28.03390	1.659600	2020-01-22	0	0	0	0	Africa
3	NaN	Andorra	42.50630	1.521800	2020-01-22	0	0	0	0	Europe
4	NaN	Angola	-11.20270	17.873900	2020-01-22	0	0	0	0	Africa
5	NaN	Antigua and Barbuda	17.06080	-61.796400	2020-01-22	0	0	0	0	Americas





	Province/State	Country/Region	Lat	Long	Date	Confirmed	Deaths	Recovered	Active	WHO Region
0	NaN	Afghanistan	33.939110	67.709953	2020-01-22	0	0	0	0	Eastern Mediterranean
1	NaN	Albania	41.153300	20.168300	2020-01-22	0	0	0	0	Europe
2	NaN	Algeria	28.033900	1.659600	2020-01-22	0	0	0	0	Africa
3	NaN	Andorra	42.506300	1.521800	2020-01-22	0	0	0	0	Europe
4	NaN	Angola	-11.202700	17.873900	2020-01-22	0	0	0	0	Africa
...	...	...	...	...	...	...	...	...	...	...
49063	NaN	Sao Tome and Principe	0.186400	6.613100	2020-07-27	865	14	734	117	Africa
49064	NaN	Yemen	15.552727	48.516388	2020-07-27	1691	483	833	375	Eastern Mediterranean
49065	NaN	Comoros	-11.645500	43.333300	2020-07-27	354	7	328	19	Africa
49066	NaN	Tajikistan	38.861000	71.276100	2020-07-27	7235	60	6028	1147	Europe
49067	NaN	Lesotho	-29.610000	28.233600	2020-07-27	505	12	128	365	Africa
49068 rows × 10 columns






df.isnull().sum()
Province/State    34404
Country/Region        0
Lat                   0
Long                  0
Date                  0
Confirmed             0
Deaths                0
Recovered             0
Active                0
WHO Region            0
dtype: int64




	Country/Region	Lat	Long	Date	Confirmed	Deaths	Recovered	Active	WHO Region
0	Afghanistan	33.939110	67.709953	2020-01-22	0	0	0	0	Eastern Mediterranean
1	Albania	41.153300	20.168300	2020-01-22	0	0	0	0	Europe
2	Algeria	28.033900	1.659600	2020-01-22	0	0	0	0	Africa
3	Andorra	42.506300	1.521800	2020-01-22	0	0	0	0	Europe
4	Angola	-11.202700	17.873900	2020-01-22	0	0	0	0	Africa
...	...	...	...	...	...	...	...	...	...
49063	Sao Tome and Principe	0.186400	6.613100	2020-07-27	865	14	734	117	Africa
49064	Yemen	15.552727	48.516388	2020-07-27	1691	483	833	375	Eastern Mediterranean
49065	Comoros	-11.645500	43.333300	2020-07-27	354	7	328	19	Africa
49066	Tajikistan	38.861000	71.276100	2020-07-27	7235	60	6028	1147	Europe
49067	Lesotho	-29.610000	28.233600	2020-07-27	505	12	128	365	Africa
49068 rows × 9 columns






	Country/Region	Lat	Long	Date	Confirmed	Deaths	Recovered	Active	WHO Region
0	Afghanistan	33.939110	67.709953	2020-01-22	0	0	0	0	Eastern Mediterranean
1	Albania	41.153300	20.168300	2020-01-22	0	0	0	0	Europe
2	Algeria	28.033900	1.659600	2020-01-22	0	0	0	0	Africa
3	Andorra	42.506300	1.521800	2020-01-22	0	0	0	0	Europe
4	Angola	-11.202700	17.873900	2020-01-22	0	0	0	0	Africa
...	...	...	...	...	...	...	...	...	...
49063	Sao Tome and Principe	0.186400	6.613100	2020-07-27	865	14	734	117	Africa
49064	Yemen	15.552727	48.516388	2020-07-27	1691	483	833	375	Eastern Mediterranean
49065	Comoros	-11.645500	43.333300	2020-07-27	354	7	328	19	Africa
49066	Tajikistan	38.861000	71.276100	2020-07-27	7235	60	6028	1147	Europe
49067	Lesotho	-29.610000	28.233600	2020-07-27	505	12	128	365	Africa
49068 rows × 9 columns













 (        Country/Region       Lat        Long       Date  Confirmed  Deaths  \
 0          Afghanistan  33.93911   67.709953 2020-01-22          0       0   
 1              Albania  41.15330   20.168300 2020-01-22          0       0   
 2              Algeria  28.03390    1.659600 2020-01-22          0       0   
 3              Andorra  42.50630    1.521800 2020-01-22          0       0   
 4               Angola -11.20270   17.873900 2020-01-22          0       0   
 5  Antigua and Barbuda  17.06080  -61.796400 2020-01-22          0       0   
 6            Argentina -38.41610  -63.616700 2020-01-22          0       0   
 7              Armenia  40.06910   45.038200 2020-01-22          0       0   
 8            Australia -35.47350  149.012400 2020-01-22          0       0   
 9            Australia -33.86880  151.209300 2020-01-22          0       0   
 
    Recovered  Active             WHO Region  Mortality_rate  Recovery_rate  
 0          0       0  Eastern Mediterranean             NaN            NaN  
 1          0       0                 Europe             NaN            NaN  
 2          0       0                 Africa             NaN            NaN  
 3          0       0                 Europe             NaN            NaN  
 4          0       0                 Africa             NaN            NaN  
 5          0       0               Americas             NaN            NaN  
 6          0       0               Americas             NaN            NaN  
 7          0       0                 Europe             NaN            NaN  
 8          0       0        Western Pacific             NaN            NaN  
 9          0       0        Western Pacific             NaN            NaN  ,
                 Lat          Long                 Date     Confirmed  \
 count  49068.000000  49068.000000                49068  4.906800e+04   
 mean      21.433730     23.528236  2020-04-24 12:00:00  1.688490e+04   
 min      -51.796300   -135.000000  2020-01-22 00:00:00  0.000000e+00   
 25%        7.873054    -15.310100  2020-03-08 18:00:00  4.000000e+00   
 50%       23.634500     21.745300  2020-04-24 12:00:00  1.680000e+02   
 75%       41.204380     80.771797  2020-06-10 06:00:00  1.518250e+03   
 max       71.706900    178.065000  2020-07-27 00:00:00  4.290259e+06   
 std       24.950320     70.442740                  NaN  1.273002e+05   
 
               Deaths     Recovered        Active  Mortality_rate  \
 count   49068.000000  4.906800e+04  4.906800e+04    39009.000000   
 mean      884.179160  7.915713e+03  8.085012e+03        0.027994   
 min         0.000000  0.000000e+00 -1.400000e+01        0.000000   
 25%         0.000000  0.000000e+00  0.000000e+00        0.000000   
 50%         2.000000  2.900000e+01  2.600000e+01        0.013699   
 75%        30.000000  6.660000e+02  6.060000e+02        0.038560   
 max    148011.000000  1.846641e+06  2.816444e+06        1.000000   
 std      6313.584411  5.480092e+04  7.625890e+04        0.043864   
 
        Recovery_rate  
 count   39009.000000  
 mean        0.475307  
 min         0.000000  
 25%         0.083333  
 50%         0.461002  
 75%         0.869388  
 max         1.001871  
 std         0.375111  )

















	Country/Region	Lat	Long	Date	Confirmed	Deaths	Recovered	Active	WHO Region	Mortality_rate	Recovery_rate
0	Afghanistan	33.93911	67.709953	2020-01-22	0	0	0	0	Eastern Mediterranean	0.0	0.0
1	Albania	41.15330	20.168300	2020-01-22	0	0	0	0	Europe	0.0	0.0
2	Algeria	28.03390	1.659600	2020-01-22	0	0	0	0	Africa	0.0	0.0
3	Andorra	42.50630	1.521800	2020-01-22	0	0	0	0	Europe	0.0	0.0
4	Angola	-11.20270	17.873900	2020-01-22	0	0	0	0	Africa	0.0	0.0






	Lat	Long	Date	Confirmed	Deaths	Recovered	Active	Mortality_rate	Recovery_rate
count	49068.000000	49068.000000	49068	4.906800e+04	49068.000000	4.906800e+04	4.906800e+04	49068.000000	49068.000000
mean	21.433730	23.528236	2020-04-24 12:00:00	1.688490e+04	884.179160	7.915713e+03	8.085012e+03	0.022255	0.377868
min	-51.796300	-135.000000	2020-01-22 00:00:00	0.000000e+00	0.000000	0.000000e+00	-1.400000e+01	0.000000	0.000000
25%	7.873054	-15.310100	2020-03-08 18:00:00	4.000000e+00	0.000000	0.000000e+00	0.000000e+00	0.000000	0.000000
50%	23.634500	21.745300	2020-04-24 12:00:00	1.680000e+02	2.000000	2.900000e+01	2.600000e+01	0.007086	0.250000
75%	41.204380	80.771797	2020-06-10 06:00:00	1.518250e+03	30.000000	6.660000e+02	6.060000e+02	0.028957	0.779221
max	71.706900	178.065000	2020-07-27 00:00:00	4.290259e+06	148011.000000	1.846641e+06	2.816444e+06	1.000000	1.001871
std	24.950320	70.442740	NaN	1.273002e+05	6313.584411	5.480092e+04	7.625890e+04	0.040710	0.385593


















	Country/Region	Lat	Long	Date	Confirmed	Deaths	Recovered	Active	WHO Region	Mortality_rate	Recovery_rate
0	Afghanistan	33.939110	67.709953	2020-01-22	0	0	0	0	Eastern Mediterranean	0.000000	0.000000
1	Albania	41.153300	20.168300	2020-01-22	0	0	0	0	Europe	0.000000	0.000000
2	Algeria	28.033900	1.659600	2020-01-22	0	0	0	0	Africa	0.000000	0.000000
3	Andorra	42.506300	1.521800	2020-01-22	0	0	0	0	Europe	0.000000	0.000000
4	Angola	-11.202700	17.873900	2020-01-22	0	0	0	0	Africa	0.000000	0.000000
...	...	...	...	...	...	...	...	...	...	...	...
49063	Sao Tome and Principe	0.186400	6.613100	2020-07-27	865	14	734	117	Africa	0.016185	0.848555
49064	Yemen	15.552727	48.516388	2020-07-27	1691	483	833	375	Eastern Mediterranean	0.285630	0.492608
49065	Comoros	-11.645500	43.333300	2020-07-27	354	7	328	19	Africa	0.019774	0.926554
49066	Tajikistan	38.861000	71.276100	2020-07-27	7235	60	6028	1147	Europe	0.008293	0.833172
49067	Lesotho	-29.610000	28.233600	2020-07-27	505	12	128	365	Africa	0.023762	0.253465
49068 rows × 11 columns






	Country/Region	Lat	Long	Date	Confirmed	Deaths	Recovered	Active	WHO Region	Mortality_rate	Recovery_rate	Case_Fatality_Rate	Recovery_Rate
0	Afghanistan	33.939110	67.709953	2020-01-22	0	0	0	0	Eastern Mediterranean	0.000000	0.000000	NaN	NaN
1	Albania	41.153300	20.168300	2020-01-22	0	0	0	0	Europe	0.000000	0.000000	NaN	NaN
2	Algeria	28.033900	1.659600	2020-01-22	0	0	0	0	Africa	0.000000	0.000000	NaN	NaN
3	Andorra	42.506300	1.521800	2020-01-22	0	0	0	0	Europe	0.000000	0.000000	NaN	NaN
4	Angola	-11.202700	17.873900	2020-01-22	0	0	0	0	Africa	0.000000	0.000000	NaN	NaN
...	...	...	...	...	...	...	...	...	...	...	...	...	...
49063	Sao Tome and Principe	0.186400	6.613100	2020-07-27	865	14	734	117	Africa	0.016185	0.848555	0.016185	0.848555
49064	Yemen	15.552727	48.516388	2020-07-27	1691	483	833	375	Eastern Mediterranean	0.285630	0.492608	0.285630	0.492608
49065	Comoros	-11.645500	43.333300	2020-07-27	354	7	328	19	Africa	0.019774	0.926554	0.019774	0.926554
49066	Tajikistan	38.861000	71.276100	2020-07-27	7235	60	6028	1147	Europe	0.008293	0.833172	0.008293	0.833172
49067	Lesotho	-29.610000	28.233600	2020-07-27	505	12	128	365	Africa	0.023762	0.253465	0.023762	0.253465
49068 rows × 13 columns

