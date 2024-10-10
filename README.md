# Breast_Cancer
In this project, we worked with a structured data to predict existing of brast cancer or not based on some features, like:radius_mean,	texture_mean	,perimeter_mean,	area_mean,	smoothness_mean,	compactness_mean,	concavity_mean,	concave points_mean,	symmetry_mean	,fractal_dimension_mean,	radius_se and more. 
we implemented different classification ML models ( KNN, logestic Regression, Random Forest. And we done feature selection using P-value measurment.


our Results were:

1. Breast Cancer Diagnosis using KNN
*With random state= 0
----------------------
The confusion Matrix Plot:

 ![image](https://github.com/ishraq-dagamseh/Breast_Cancer/assets/16488773/0417e16c-0130-4e35-8e8f-1a44ccff6e5e)


*Confusion matrix Report:
--------------------------
                       precision    recall  f1-score   support

           M             0.94      0.99      0.96        90
           B              0.98      0.89      0.93        53

    accuracy                                    0.95       143
   macro avg       0.96      0.94      0.95       143
weighted avg       0.95      0.95      0.95       143

*With random state= 42
------------------------

 ![image](https://github.com/ishraq-dagamseh/Breast_Cancer/assets/16488773/24dd2dc8-973f-43e2-a15f-884f55a45d7b)


precision    recall  f1-score   support

           M       0.96      0.99      0.97        89
           B       0.98      0.93      0.95        54

    accuracy                           0.97       143
   macro avg       0.97      0.96      0.96       143
weighted avg       0.97      0.97      0.96       143


with increasing number of random state its very benificial to increases the accuracy results
###########################

2. Breast Cancer regression using Logistic regression


Results:
*With random_state =1
----------------------
We split dataset into 25% for testing and 85% for training and we found these results after fitting on Logistic regression algorithm:
![image](https://github.com/ishraq-dagamseh/Breast_Cancer/assets/16488773/e4a002b7-0c5d-4a04-8771-15edd0d3bc96)

 
And the metrics results:
           precision    recall   f1-score   support
 M       0.97            0.98       0.97        88
 B        0.96            0.95       0.95        55

 accuracy                           0.97       143
 macro avg       0.96       0.96       0.96        143
weighted avg       0.97      0.97      0.96       143

*With random_state =42
-----------------------
![image](https://github.com/ishraq-dagamseh/Breast_Cancer/assets/16488773/570b8071-a037-4a0c-b7fd-ea7ae25a115f)

 
*And confusion report Results:
-------------------------------

               precision    recall  f1-score   support

           M       0.98      1.00      0.99        89
           B       1.00      0.96      0.98        54

    accuracy                           0.99       143
   macro avg       0.99      0.98      0.99       143
weighted avg       0.99      0.99      0.99       143


# we concluded that number of random state its very affected 
###########################

3.Breast Cancer regression using MLR


The decided features were:
a) radius (mean of distances from center to points on the perimeter)
b) texture (standard deviation of gray-scale values)
c) perimeter
d) area
e) smoothness (local variation in radius lengths)
f) compactness (perimeter^2 / area - 1.0)
g) concavity (severity of concave portions of the contour)
h) concave points (number of concave portions of the contour)
i) symmetry
j) fractal dimension ("coastline approximation" - 1)
أ) دائرة نصف قطرها (يعني المسافات من مركز إلى نقاط على المحيط)
ب) الملمس (الانحراف المعياري للقيم النطاق الرمادي)
ج) محيط
د) المنطقة
ه) نعومة (تباين محلي في أطوال دائرة نصف قطرها)
و) الارتياج (محيط ^ 2 / المنطقة - 1.0)
ز) concavity (شدة أجزاء مقعرة من المحيط)
ح) النقاط المقصرة (عدد الأجزاء المقررة من المحيط)
ط) التماثل
ي) البعد كسور ("التقريب الساحلي" - 1)

Results:
==============================================================================
Omnibus:                       30.729   Durbin-Watson:                   1.783
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               34.215
Skew:                           0.582   Prob(JB):                     3.72e-08
Kurtosis:                       3.296   Cond. No.                     1.33e+06


On the single linear regression, the results were:
r2 socre is: 0.7234520971661356
mean_sqrd_error is== 0.06545512493109215
root_mean_squared error of is== 0.2558419921183623.
so we will try to optimized the results using MLR.
We applied MLR on csv file contains some features decided the type of breast mass, and we noticed this results after firstly OLS:


# Which are indicated for the highest value and its impact less than the number of the number x9 that indicates to symmetry_mean feature based on P>|t| values, so we will re-compute the values of OLS regressor.
#  After delete x9 or symmetry_mean feature, the next table show the result
==============================================================================
Omnibus:                       30.636   Durbin-Watson:                   1.782
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               34.097
Skew:                           0.582   Prob(JB):                     3.94e-08
Kurtosis:                       3.292   Cond. No.                     1.33e+06

# We noticed from previous table that the x5 or smoothness_mean feature was the highest results, so we will delete it.


==============================================================================
Omnibus:                       30.552   Durbin-Watson:                   1.783
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               33.989
Skew:                           0.581   Prob(JB):                     4.16e-08
Kurtosis:                       3.291   Cond. No.                     1.33e+06
# Previous results showed that we must ignore the x10 0r the texture_se in column 12 in the original x
So we must delete it and re- compute the p value.

  ==============================================================================
     

==============================================================================
Omnibus:                       30.471   Durbin-Watson:                   1.784
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               33.886
Skew:                           0.579   Prob(JB):                     4.38e-08
Kurtosis:                       3.294   Cond. No.                     1.33e+06

# From previous results we noticed that x16 or column 19 or symmetry_se  feature had the highest p-value, so we must delete it.
                

==============================================================================
Omnibus:                       29.865   Durbin-Watson:                   1.781
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               33.118
Skew:                           0.575   Prob(JB):                     6.43e-08
Kurtosis:                       3.277   Cond. No.                     1.33e+06

# We conclude that the highest p-value was the x10 or col 13 or area_se feature, so we will delete it.

          

==============================================================================
Omnibus:                       30.032   Durbin-Watson:                   1.781
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               33.330
Skew:                           0.576   Prob(JB):                     5.79e-08
Kurtosis:                       3.281   Cond. No.                     1.29e+06
# The results were that x12 or col 17 or feature of concave points_se


==============================================================================
Omnibus:                       27.773   Durbin-Watson:                   1.787
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               30.631
Skew:                           0.563   Prob(JB):                     2.23e-07
Kurtosis:                       3.155   Cond. No.                     1.10e+06

# The highest p-value from x12 or col 18 or symmetry_se feature.
==============================================================================
Omnibus:                       27.760   Durbin-Watson:                   1.788
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               30.614
Skew:                           0.563   Prob(JB):                     2.25e-07
Kurtosis:                       3.155   Cond. No.                     1.05e+06

# The highest p_value from x 4 or col 4 or area_mean feature

==============================================================================
Omnibus:                       27.501   Durbin-Watson:                   1.787
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               30.313
Skew:                           0.561   Prob(JB):                     2.62e-07
Kurtosis:                       3.144   Cond. No.                     8.51e+05
# Highest value from x5 or col7 or smoothness_mean feature

                
==============================================================================
Omnibus:                       27.873   Durbin-Watson:                   1.785
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               30.747
Skew:                           0.564   Prob(JB):                     2.11e-07
Kurtosis:                       3.159   Cond. No.                     8.38e+05
===============================================================
# x8 or col 14 #the highest one  was area_se feature 


==============================================================================
Omnibus:                       28.925   Durbin-Watson:                   1.778
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               31.985
Skew:                           0.571   Prob(JB):                     1.13e-07
Kurtosis:                       3.213   Cond. No.                     8.37e+05
==============================================================================
# X14 or col 25 was the highest p- value

==============================================================================
Omnibus:                       29.212   Durbin-Watson:                   1.778
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               32.325
Skew:                           0.572   Prob(JB):                     9.57e-08
Kurtosis:                       3.231   Cond. No.                     7.03e+05

# X2 or col2 was the highest


==============================================================================
Omnibus:                       29.105   Durbin-Watson:                   1.778
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               32.192
Skew:                           0.571   Prob(JB):                     1.02e-07
Kurtosis:                       3.229   Cond. No.                     7.00e+05
# X14 or col 27 was the highest 

                 
==============================================================================
Omnibus:                       29.970   Durbin-Watson:                   1.792
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               33.266
Skew:                           0.579   Prob(JB):                     5.98e-08
Kurtosis:                       3.253   Cond. No.                     6.84e+05
# X8 or col20 was the highest one 

                  
==============================================================================
Omnibus:                       31.150   Durbin-Watson:                   1.797
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               34.754
Skew:                           0.588   Prob(JB):                     2.84e-08
Kurtosis:                       3.289   Cond. No.                     4.64e+05
# X10 or col 23
              
==============================================================================
Omnibus:                       29.269   Durbin-Watson:                   1.789
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               32.365
Skew:                           0.567   Prob(JB):                     9.38e-08
Kurtosis:                       3.279   Cond. No.                     4.61e+05
# X2 0r col3 was the highest 
                
==============================================================================
Omnibus:                       27.603   Durbin-Watson:                   1.774
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               30.285
Skew:                           0.551   Prob(JB):                     2.65e-07
Kurtosis:                       3.254   Cond. No.                     4.57e+05
#X4 or col 10
               
==============================================================================
Omnibus:                       26.548   Durbin-Watson:                   1.787
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               28.978
Skew:                           0.538   Prob(JB):                     5.10e-07
Kurtosis:                       3.254   Cond. No.                     4.51e+05
# X10 was the highest

                 
==============================================================================
Omnibus:                       29.452   Durbin-Watson:                   1.803
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               32.637
Skew:                           0.560   Prob(JB):                     8.19e-08
Kurtosis:                       3.349   Cond. No.                     4.51e+05


# X4 is the highest 
      
==============================================================================
Omnibus:                       34.039   Durbin-Watson:                   1.784
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               39.069
Skew:                           0.587   Prob(JB):                     3.28e-09
Kurtosis:                       3.517   Cond. No.                     4.35e+05

# Now, all x’s less than 0.025 , so we re-split data and then evaluate it.

The evaluation results were:

r2 socre is: 0.768052920531675
mean_sqrd_error is:  0.056202012407337305
root_mean_squared error of is: 0.23706963619860158.


our Conclusion:
----------------

We concluded that the most feature that effected on the prediction of the breast masses were 10 features from 30 features were the:
•	radius_mean
•	texture_mean
•	concavity_mean
•	symmetry_mean
•	concavity_se
•	perimeter_worst
•	area_worst
•	concavity_worst
•	symmetry_worst
•	fractal_dimension_worst.

Best results from classifiers were 99% from Logistic regression 















