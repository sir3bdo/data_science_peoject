# Importing libraries
import customtkinter, tkinter
from cProfile import label
import tkinter
from tkinter import *
import string
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,mean_squared_error,r2_score,accuracy_score,f1_score
from sklearn import metrics
from sklearn import tree
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB

############################################################
# Declaring some variables
dsn=None
dic={}
flag=[]
ac1=[]
ac=[]
ac2=[]
ac3=[]
xKNN=[]
xLINREG=[]
Xlogistic=[]
xNAIVE=[]
lisreg=[]
dic1={}
ax=None
ay=None
az=None
customtkinter.set_appearance_mode("dark")  
customtkinter.set_default_color_theme("blue")
app = customtkinter.CTk()

####################################################################################################
# Validation GUI 
def validation():
    # Global variable
    global app

    # Create label for validation
    label8 = Label(app, text="Validation", font=('Times New Roman', '20'))
    label8.grid(column=0, row=7)  # Place the label in the grid layout of the app

    # Create label for Confusion Matrix
    label9 = Label(app, text="Confusion Matrix", font=('Times New Roman', '10'))
    label9.grid(column=0, row=8)  # Place the label in the grid layout of the app
    # Create button for Confusion Matrix
    button8 = customtkinter.CTkButton(app, text="click", width=7, height=1, command=confusmtrx)
    button8.grid(column=1, row=8)  # Place the button in the grid layout of the app

    # Create label for Accuracy
    label10 = Label(app, text="Accuracy", font=('Times New Roman', '10'))
    label10.grid(column=2, row=8)  # Place the label in the grid layout of the app
    # Create button for Accuracy
    button9 = customtkinter.CTkButton(app, text="click", width=7, height=1, command=acure)
    button9.grid(column=3, row=8)  # Place the button in the grid layout of the app

###################################################
# Read the CSV file "satgpa.csv" into a DataFrame
df = pd.read_csv("satgpa.csv", encoding='latin-1')
df.head()               # Display the first few rows of the DataFrame
df.isna().sum()         # Check the number of missing values in each column of the DataFrame
df.duplicated().sum()   # Check the number of duplicated rows in the DataFrame
df.describe()           # Generate descriptive statistics of the DataFrame
df.columns              # Retrieve the column names of the DataFrame

##################################################
# Data Visualization
# Plotting Data individually
Gender = df['sex'].value_counts()

# Pie chart showing the percentage of females and males
plt.pie(Gender, labels=['Females', 'Males'], autopct='%.1f%%')
plt.title('Gender Pie Chart')
plt.show()

# Histograms of sat_scores
Sat_V = df['sat_v']
plt.hist(Sat_V, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Sat_v Grades')
plt.ylabel('Frequency')
plt.title('Histogram of Sat_v')
plt.show()

Sat_m = df['sat_m']
plt.hist(Sat_m, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Sat_m Grades')
plt.ylabel('Frequency')
plt.title('Histogram of Sat_m')
plt.show()

Sat_sum = df['sat_sum']
plt.hist(Sat_sum, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Sat_sum Grades')
plt.ylabel('Frequency')
plt.title('Histogram of Sat_sum')
plt.show()

# Box-Plot of hs_gba & fy_gba 
High_School_GPA=df['hs_gpa']
First_Year_GPA=df['fy_gpa']
data = [High_School_GPA,First_Year_GPA]
sns.boxplot(data)
plt.show()

# Generate the correlation matrix
correlation_matrix = df[['sex', 'sat_v', 'sat_m', 'sat_sum', 'hs_gpa', 'fy_gpa']].corr()

# Print out the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)

#Generate a heatmap for the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

sns.pairplot(df, hue='sex')
####################################################

y = df['fy_gpa']                            # Extract the 'fy_gba' column as the target variable
X = df.drop(['fy_gpa'], axis=1)             # Drop the 'fy_gba' column from the DataFrame and assign the remaining columns to X

# Split the data into training and testing sets, with 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
model1 = LinearRegression()                 # Create an instance of the LinearRegression model
model1.fit(X_train, y_train)                # Fit the model using the training data
y_hat = model1.predict(X_test)              # Predict the target variable using the test data

# Calculate the mean squared error (MSE) between the actual and predicted values
print('MSE :' , mean_squared_error(y_test, y_hat))
plt.scatter(y_test, y_hat)

# Calculate the coefficient of determination (R-squared score) between the actual and predicted values
print('r2_score = ',r2_score(y_test, y_hat))

######################################################
# Ridge Regression
def RIDGEREG():
    # Global variables
    global flag  # Indicator flag
    global ac3  # List to store evaluation metrics
    global lisreg  # List to store regression model information
    global ax  # List to store mean absolute error
    global az  # List to store root mean squared error
    global ay  # List to store mean squared error

    flag.append(4)  # Append value 4 to the flag list

    y = df['fy_gpa']  # Target variable
    X = df.drop(['fy_gpa'], axis=1)  # Features
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Split the data into training and test sets

    reg = linear_model.Ridge(alpha=0.01, fit_intercept=True)  # Create a Ridge regression object
    reg.fit(X_train, y_train)  # Fit the regression model using training data
    y_hat = reg.predict(X_test)  # Predict the target variable for test data

    #print('MSE :' , mean_squared_error(y_test, y_hat))
    plt.scatter(y_test, y_hat)  # Scatter plot of predicted vs. actual values
    print('R2 :' , r2_score(y_test, y_hat))  # Compute R2 score

    ax = [metrics.mean_absolute_error(y_test, y_hat)]  # Compute and store mean absolute error
    ay = [metrics.mean_squared_error(y_test, y_hat)]  # Compute and store mean squared error
    az = [np.sqrt(metrics.mean_squared_error(y_test, y_hat))]  # Compute and store root mean squared error

    validation()  # Call the validation function

    return reg  # Return the regression model

######################################################################################
# Read the CSV file "student_exam_data.csv" into a DataFrame
df_class = pd.read_csv("student_exam_data.csv")
df_class.head()                     # Display the first few rows of the DataFrame
df_class.columns                    # Retrieve the column names of the DataFrame
df_class.isna().sum()               # Check the number of missing values in each column of the DataFrame
df_class.duplicated().sum()         # Check the number of duplicated rows in the DataFrame
y_class = df_class['Pass/Fail']     # Extract the 'Pass/Fail' column as the target variable
X_class =  df_class.drop(['Pass/Fail'], axis=1)          # Drop the 'Pass/Fail' column from the DataFrame and assign the remaining columns to X_class

# Split the data into training and testing sets, with 80% for training and 20% for testing
X_train_cl, X_test_cl, y_train_cl, y_test_cl = train_test_split(X_class, y_class, test_size=0.2,random_state=0)
y_class.value_counts()              # Count the occurrences of each value in the 'Pass/Fail' column

##############################################################################################
# Logistic Regression
def LOGISTIC():
    # Global variables
    global flag     # Indicator flag
    global ac       # List to store confusion matrix and accuracy

    flag.append(2)  # Append value 2 to the flag list

    # Logistic Regression
    LR = LogisticRegression()       # Create a LogisticRegression object
    LR.fit(X_train_cl,y_train_cl)   # Fit the model using training data
    y_pred = LR.predict(X_test_cl)  # Predict the labels for test data

    # Evaluation Metrics
    accuracy_score(y_test_cl,y_pred)    # Compute accuracy score, but not assigned to a variable
    f1_score(y_test_cl,y_pred)           # Compute F1 score, but not assigned to a variable
    print(classification_report(y_test_cl,y_pred))      # Print classification report
    print(confusion_matrix(y_test_cl,y_pred))            # Print confusion matrix
    ac = [confusion_matrix(y_test_cl,y_pred), metrics.accuracy_score(y_pred,y_test_cl)]     # Store confusion matrix and accuracy in the ac list
    validation()        # Call the validation function

##################################################################################
# K-nearest neighboorhood
def KNN():
    # Global variables
    global flag  # Indicator flag
    global ac1  # List to store confusion matrix and accuracy

    flag.append(1)  # Append value 1 to the flag list

    KNN = KNeighborsClassifier(n_neighbors=3)  # Create a KNeighborsClassifier object with 3 neighbors
    KNN.fit(X_train_cl, y_train_cl)  # Fit the model using training data
    y_knn_pred = KNN.predict(X_test_cl)  # Predict the labels for test data

    f1_score(y_test_cl, y_knn_pred)  # Compute F1 score, but not assigned to a variable

    validation()  # Call the validation function

    ac1 = [confusion_matrix(y_test_cl, y_knn_pred), metrics.accuracy_score(y_knn_pred, y_test_cl)]
    # Store confusion matrix and accuracy in the ac1 list

###########################################################################################
# Naive Bayes
def navyby():
    # Global variables
    global dsn  # Dataset variable
    global flag  # Indicator flag
    global ac2  # List to store confusion matrix and accuracy

    flag.append(3)  # Append value 3 to the flag list

    Naive_Bayes = GaussianNB()  # Create a GaussianNB object
    Naive_Bayes.fit(X_train_cl, y_train_cl)  # Fit the model using training data
    y_NB_pred = Naive_Bayes.predict(X_test_cl)  # Predict the labels for test data

    f1_score(y_test_cl, y_NB_pred)  # Compute F1 score, but not assigned to a variable

    validation()  # Call the validation function

    ac2 = [metrics.confusion_matrix(y_test_cl, y_NB_pred), metrics.accuracy_score(y_NB_pred, y_test_cl)]
    # Store confusion matrix and accuracy in the ac2 list

######################################################################################################
# Regression GUI
def Regression():
    # Global variable
    global app

    #  Create label for Linear Regression
    label7 = Label(app,text="Linear Regression",font=('Times New Roman','10'))
    label7.grid(column=0,row=3)  # Place the label in the grid layout of the app
    #  Create button for Linear Regression
    button7 = customtkinter.CTkButton(app, text="click",width=7,height=1,command=RIDGEREG)
    button7.grid(column=1,row=3)  # Place the button in the grid layout of the app

#######################################################################################
# Classification GUI
def Classification():
    # Global variable
    global app

    # Create label for KNN
    label4 = Label(app, text="KNN", font=('Times New Roman', '10'))
    label4.grid(column=3, row=3)  # Place the label in the grid layout of the app
    # Create button for KNN
    button4 = customtkinter.CTkButton(app, text="click", width=7, height=1, command=KNN)
    button4.grid(column=4, row=3)  # Place the button in the grid layout of the app

    # Create label for logistic regression
    label5 = Label(app, text="Logistic Regression", font=('Times New Roman', '10'))
    label5.grid(column=3, row=4)  # Place the label in the grid layout of the app
    # Create button for logistic regression
    button5 = customtkinter.CTkButton(app, text="click", width=7, height=1, command=LOGISTIC)
    button5.grid(column=4, row=4)  # Place the button in the grid layout of the app

    # Create label for Naive Bayes
    label6 = Label(app, text="Naïve Bayes", font=('Times New Roman', '10'))
    label6.grid(column=3, row=5)  # Place the label in the grid layout of the app
    # Create button for Naive Bayes
    button6 = customtkinter.CTkButton(app, text="click", width=7, height=1, command=navyby)
    button6.grid(column=4, row=5)  # Place the button in the grid layout of the app

########################################################################################################
# Accuracy calculation
def acure():
    # Global variables
    global dic1  # Dictionary to store regression error metrics
    global dic  # Dictionary to store classification accuracy
    global flag  # Indicator flag
    global ax  # List to store mean absolute error
    global ay  # List to store mean squared error
    global az  # List to store root mean squared error

    if 1 in flag:
        dic["KNN"] = [ac1[1]]  # Store KNN accuracy in the dic dictionary

    if 2 in flag:
        dic["LOGISTIC"] = [ac[1]]  # Store Logistic accuracy in the dic dictionary

    if 3 in flag:
        dic["niavyby"] = [ac2[1]]  # Store Naive Bayes accuracy in the dic dictionary
    
    if 4 in flag:
        dic1["MEAN ABS ERROR"] = ax  # Store mean absolute error in the dic1 dictionary
        dic1["MEAN SQUARE ERROR"] = ay  # Store mean squared error in the dic1 dictionary
        dic1["MEAN root ERROR"] = az  # Store root mean squared error in the dic1 dictionary
        pf2 = pd.DataFrame(data=dic1)  # Create a DataFrame from dic1
        fig2 = pf2.plot.bar().get_figure()  # Plot a bar chart from the DataFrame
        bar2 = FigureCanvasTkAgg(fig2, app)  # Create a Tkinter canvas for the bar chart
        bar2.get_tk_widget().grid(column=9, row=13)  # Place the bar chart in the grid layout of the app

    pf = pd.DataFrame(data=dic)  # Create a DataFrame from dic
    fig = pf.plot.bar().get_figure()  # Plot a bar chart from the DataFrame
    bar1 = FigureCanvasTkAgg(fig, app)  # Create a Tkinter canvas for the bar chart
    bar1.get_tk_widget().grid(column=5, row=13)  # Place the bar chart in the grid layout of the app

####################################################################################################################
# Confusion Matrix Calculation
def confusmtrx():
    # Global variables
    global flag  # Indicator flag
    global xKNN  # Confusion matrix for KNN
    global xLINREG  # Confusion matrix for logistic regression
    global xDECTREE  # Confusion matrix for decision tree
    global xNAIVE  # Confusion matrix for naive Bayes

    if 1 in flag:
        xKNN = ac1[0]  # Assign the KNN confusion matrix to xKNN
        la1 = Label(app, text=f"{xKNN} KNN")  # Create a label to display the KNN confusion matrix
        la1.grid(row=10, column=0)  # Place the label in the grid layout of the app

    if 2 in flag:
        xLINREG = ac[0]  # Assign the logistic regression confusion matrix to xLINREG
        la2 = Label(app, text=f"{xLINREG} logistic")  # Create a label to display the logistic regression confusion matrix
        la2.grid(row=11, column=0)  # Place the label in the grid layout of the app

    if 3 in flag:
        xNAIVE = ac2[0]  # Assign the naive Bayes confusion matrix to xNAIVE
        la3 = Label(app, text=f"{xNAIVE} naive")  # Create a label to display the naive Bayes confusion matrix
        la3.grid(row=12, column=0)  # Place the label in the grid layout of the app

##################################################################################################

x2 = sm.add_constant(X)
models = sm.OLS(y,x2)
result = models.fit()
print (result.summary())

# Creating GUI
# Set the geometry of the app window to 1280x720
app.geometry("1280x720")

# Create a label for the "Project"
label1 = Label(app, text="Project", font=('Times New Roman', '20'))
label1.grid(column=3, row=0)

# Create a label for the "Regression"
label2 = Label(app, text="Regression", font=('Times New Roman', '10'))
label2.grid(column=0, row=2)
# Create a button with the "Regression"
button2 = customtkinter.CTkButton(app, text="click", width=7, height=1, command=Regression)
button2.grid(column=1, row=2)

# Create a label for the "Classification"
label3 = Label(app, text="Classification", font=('Times New Roman', '10'))
label3.grid(column=3, row=2)
# Create a button for the "Classification"
button3 = customtkinter.CTkButton(app, text="click", width=7, height=1, command=Classification)
button3.grid(column=4, row=2)

# Start the main event loop of the app
app.mainloop()
