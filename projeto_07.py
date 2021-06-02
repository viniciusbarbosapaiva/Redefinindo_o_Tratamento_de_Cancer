# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
sns.set(style='white', palette='deep')
import os
width = 0.35

# Funções
def autolabel(rects,ax, df): #autolabel
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{} ({:.2f}%)'.format(height, height*100/df.shape[0]),
                    xy = (rect.get_x() + rect.get_width()/2, height),
                    xytext= (0,3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

# Importing Dataset
path = os.getcwd()
print(path)

documents = os.listdir();documents
with open('training_variants') as f:
    data = f.read()
    print(data)

columns = ['Gene','Variation','Class']
df = pd.read_csv('training_variants',sep=',',usecols=columns)   
df.head(5)     

# Looking for null values
null_values = (np.sum(df.isna())/len(df))*100
null_values = pd.DataFrame(null_values, columns=['% of Null Values'])
null_values

# Data info
df.columns
df.info()
df.describe()
gene_count, variation_count, class_count = [df[i].value_counts() for i in df.columns]

# Correlation with independent Variable (Note: Models like RF are not linear like these)
df.columns
codes_gene, uniques_gene =  pd.factorize(df.iloc[:,0])
codes_variation, uniques_variation =  pd.factorize(df.iloc[:,1])
df2 = df.drop(['Class'], axis=1)
df2['gene_codes'] = codes_gene
df2['variation_codes'] = codes_variation
df2.corrwith(df.Class).plot.bar(
        figsize = (10, 10), title = "Correlation with Class", fontsize = 15,
        rot = 45, grid = True)
# Plot Classes to verify unbalanced class
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
label = df['Class'].unique()
ind = np.arange(1,len(label)+1)
rects = ax.bar(class_count.index, class_count, width)
ax.set_title('Classes Count')
ax.set_xlabel('Classes')
ax.set_ylabel('Frequency of Classes')
ax.set_xticks(ind)
ax.grid(b=True, which='major', linestyle='--')
autolabel(rects,ax,df)

# Defining X and y
X = df.drop('Class', axis=1)
y = df['Class']
y_new = y.apply(lambda x: x-1)

# Get dummies Variables and avoinding dummy traps
X = pd.get_dummies(X, drop_first= True)

# SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE()    
X_sm, y_sm = sm.fit_resample(X,y_new)  
print(X_sm.shape)
print(y_sm.shape)  

# Plot Classes after SMOTE Technique
y_sm_count = y_sm.value_counts()
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
label = y_sm.unique()
ind = np.arange(1,len(label)+1)
rects = ax.bar(y_sm_count.index, y_sm_count, width)
ax.set_title('Classes Count')
ax.set_xlabel('Classes')
ax.set_ylabel('Frequency of Classes (SMOTE)')
ax.set_xticks(ind)
ax.grid(b=True, which='major', linestyle='--')
autolabel(rects,ax,y_sm)

#Splitting the Dataset into the training set and test set
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, stratify=y_sm, random_state=0)
y_train = to_categorical(y_train, num_classes=len(y.unique()))
y_test = to_categorical(y_test, num_classes= len(y.unique()))
y_train = pd.DataFrame(y_train)
y_test = pd.DataFrame(y_test)
X_train.shape
X_test.shape
y_train.shape
y_test.shape

#Importing Keras libraries e packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU
leaky_relu_alpha = 0.1
import time
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
###########################################################################################################################            
#Initialising the ANN
classifier = Sequential()

#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim =int((X_test.shape[1]/2)+1) , init = 'uniform',  input_dim = int(X_test.shape[1]),
                     activation = 'relu'))#kernel_regularizer=regularizers.l1(0.001)
classifier.add(Dropout(p= 0.2))
#classifier.add(LeakyReLU(alpha=leaky_relu_alpha))

#output_dim = average of the sum of the number of variables + 1
#init = Weight. Generates randomly. Always 'uniform'
#activation = rectified linear activation function. The most used.
#Input_dim = number of variables
#Dropout = avoids overfitting. It starts with 0.1. If you continue to overfitting, try 0.2 to 0.5. Never greater than 0.5 otherwise it will be underfitting

#Adding the second hidden layer
classifier.add(Dense(output_dim = 24, init = 'uniform',
                     ))#activation = 'relu', kernel_regularizer=regularizers.l1(0.001)
classifier.add(LeakyReLU(alpha=leaky_relu_alpha))
classifier.add(Dropout(p= 0.2))

#Adding the output layer
classifier.add(Dense(output_dim = len(y.unique()), init = 'uniform', activation = 'softmax'))
#output_dim = As in this case we want more than one class
#init = Weight. Generates randomly. Always 'uniform'
#activation = softmax activation function. The most used for output layer when they are more than one class. 

#Compiling the ANN
adam=Adam(lr=0.0001)
classifier.compile(optimizer = adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])
#optimizer = falgorithm to select the best ANN weight. 'Adam' is one of the most used
#loss =  Algorithm that minimizes losses from the Adam. As the output is binary, it used binary_crossentropy. If there is more than one categorical_crossentropy variable
#metrics = Accuracy

#Early Stoppnig to avoid overfitting
es = EarlyStopping(monitor='val_loss',mode='min', verbose=1 )

#Fit classifier to the training test
t0 = time.time()
history = classifier.fit(X_train, y_train, batch_size = 10, epochs = 100, validation_data=(X_test, y_test), callbacks=[es])
t1 = time.time()
print('The training process took {:.2f}'.format(t1-t0))

#batch_size = it does not have a certain value. 
#epochs  = it does not have a certain value.

#Predicting the test set result
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) #To converto true and false

from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average=None)
rec = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)
    
results_final = pd.DataFrame([['ANN W/ Dropout', acc, prec, rec, f1]],
                      columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('accuracy.png')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('loss.png')
plt.show()   

# Evaluating Results
#Making the confusion matrix
cm = multilabel_confusion_matrix(y_test,y_pred, labels= y_new.unique())
labels = ['Positive', 'Negative']
ind = np.arange(len(labels))
fig = plt.figure(figsize=(10,10))
for i in np.arange(1,len(cm)+1):
    ax = fig.add_subplot(3,3,i)
    ax.set_title('Confusion Matrix for Class {}'.format(i))
    ax.imshow(cm[i-1], cmap='viridis')
    for x in ind:
        for j in ind:
            text = ax.text(j, x, cm[i-1][x, j],
            ha="center", va="center", color="r")
    ax.set_xticks(ind)
    ax.set_yticks(ind)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.xaxis.tick_top()
    plt.tight_layout()  
plt.show() 

#Cheking wheter the model identify all classes. Sometimes, because overfitting, the model doesn't classify any class of y_test.
# All new 'sum' columns must have the number one. Otherwise, the model has failed.
y_pred = pd.DataFrame(y_pred)
y_test['sum'] = [np.sum(y_test.iloc[i,:]) for i in np.arange(0,len(y_test))]
sum_y_test = np.sum(y_test['sum'])
y_pred['sum'] = [np.sum(y_pred.iloc[i,:]) for i in np.arange(0,len(y_pred))]  
sum_y_pred = np.sum(y_pred['sum']) 
print('The model is good' if sum_y_test==sum_y_pred else "The model has failed. It's necessary to train again with diferent parameters.")



         







