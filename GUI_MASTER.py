from subprocess import call
import tkinter as tk
import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageTk
from tkinter import ttk
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest Classifier
from joblib import dump
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix, 
    precision_recall_curve  # Ensure this is imported
)
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import xgboost as xgb
import time
from sklearn.naive_bayes import MultinomialNB 
import warnings
import smtplib
from email.message import EmailMessage
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Bidirectional, LSTM






warnings.filterwarnings("ignore", category=DeprecationWarning) 
root = tk.Tk()
root.title("Intrustion Detection and Prevention System Using Machine Learning and Deep Learning Techniques")

w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.configure(background="black")

image = Image.open('m4.jpg')

image = image.resize((1050,750))

background_image = ImageTk.PhotoImage(image)

background_image=ImageTk.PhotoImage(image)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=0, y=70) #, relwidth=1, relheight=1)


image = Image.open('m4.jpg')

image = image.resize((270,620))

background_image = ImageTk.PhotoImage(image)

background_image=ImageTk.PhotoImage(image)

background_label = tk.Label(root, image=background_image)

background_label.image = background_image

background_label.place(x=1060, y=70) 
# function to change to next image
# function to change to next image
'''def move():
	global x
	if x == 4:
		x = 1
	if x == 1:
		logo_label.config(image=img3)
	elif x == 2:
		logo_label.config(image=img2)
	elif x == 3:
		logo_label.config(image=img3)
	x = x+1
	root.after(2000, move)

# calling the function
move()'''




  # , relwidth=1, relheight=1)
lbl = tk.Label(root, text="Intrustion Detection and Prevention System Using Machine Learning and Deep Learning Techniques", font=('times', 22,' bold '), height=1, width=80,bg="brown",fg="white")
lbl.place(x=0, y=0)
# _+++++++++++++++++++++++++++++++++++++++++++++++++++++++

def Model_Training():
    data = pd.read_csv("test.csv")
    data.head()

    data = data.dropna()

   

    """Feature Selection => Manual"""
    x = data.drop(['Average_Packet_Size','Duration','Label'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['Label']
    print(type(y))
    x.shape
    

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20,random_state=133)

    # from sklearn.svm import SVC
    # svcclassifier = SVC(kernel='linear')
    # svcclassifier.fit(x_train, y_train)
      
    from sklearn.svm import SVC
    start_time = time.time()
    svcclassifier = SVC(kernel='linear')
    svcclassifier.fit(x_train, y_train)
    training_time = time.time() - start_time
    y_pred = svcclassifier.predict(x_test)
    y_pred_prob = svcclassifier.decision_function(x_test)
    print(y_pred)
    
   
    print("=" * 40)
    print("==========")
    print("Classification Report : ",(classification_report(y_test, y_pred)))
    print("Accuracy : ",accuracy_score(y_test,y_pred)*100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Training Time: %.2f seconds" % training_time)
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))
    # Confusion Matrix (Heatmap)
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Not Detected", "Detected"], yticklabels=["Not Detected", "Detected"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


    
    
    label4 = tk.Label(root,text =str(repo),width=45,height=15,bg='seashell2',fg='black',font=("Tempus Sanc ITC",14))
    label4.place(x=250,y=100)
    
    label5 = tk.Label(
        root, 
        text=f"Accuracy: {ACC:.2f}%\nTraining Time: {training_time:.2f} seconds\nModel saved as attack_SVM.joblib", 
        width=45, 
        height=5, 
        bg='khaki', 
        fg='black', 
        font=("Tempus Sanc ITC", 14)
    )
    label5.place(x=250,y=420)
    from joblib import dump
    dump (svcclassifier,"attack_SVM.joblib")
    print("Model saved as attack_SVM.joblib")
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    from itertools import cycle
    
    n_classes = len(y.unique())  # Get number of classes
    y_test_bin = label_binarize(y_test, classes=sorted(y.unique()))  
    y_pred_prob = svcclassifier.decision_function(x_test)  # Get decision function for multi-class

    # Plot ROC Curve for each class
    plt.figure(figsize=(10, 6))
    colors = cycle(["blue", "red", "green", "purple", "orange", "brown"])

    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, label=f"Class {i} (AUC = {roc_auc:.2f})")

    # Plot reference line
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")

    # Labels and Legends
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-Class ROC Curve")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()
    


 # Calculate the mean of each feature
    feature_means = x.mean()

    # Create the bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(feature_means.index, feature_means.values, color="skyblue")
    plt.title("Mean of Each Feature", fontsize=16)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Mean Value", fontsize=14)
    plt.xticks(rotation=45)  # Rotate feature names for better visibility
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    
    
    
    
def Model_Training1():
    data = pd.read_csv("test.csv")
    data.head()

    data = data.dropna()

    """Feature Selection => Manual"""
    x = data.drop(['Average_Packet_Size', 'Duration','Label'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['Label']
    print(type(y))
    x.shape

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=123)

    # Replace SVM with Random Forest Classifier
    start_time = time.time()
    random_forest_classifier = RandomForestClassifier(n_estimators=100, random_state=123)
    random_forest_classifier.fit(x_train, y_train)
    training_time = time.time() - start_time
    y_pred = random_forest_classifier.predict(x_test)
    print(y_pred)

    print("=" * 40)
    print("==========")
    print("Classification Report : ", classification_report(y_test, y_pred))
    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Training Time: %.2f seconds" % training_time)
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))

    label4 = tk.Label(root, text=str(repo), width=45, height=15, bg='seashell2', fg='black', font=("Tempus Sanc ITC", 14))
    label4.place(x=250, y=100)

    label5 = tk.Label(
         root, 
         text=f"Accuracy: {ACC:.2f}%\nTraining Time: {training_time:.2f} seconds\nModel saved as attack_RandomForest.joblib", 
         width=45, 
         height=5, 
         bg='khaki', 
         fg='black', 
         font=("Tempus Sanc ITC", 14)
     )
    label5.place(x=250, y=420)

    dump(random_forest_classifier, "attack_RandomForest.joblib")
    print("Model saved as attack_RandomForest.joblib")
  # Calculate the mean of each feature
    feature_means = x.mean()

     # Create the bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(feature_means.index, feature_means.values, color="skyblue")
    plt.title("Mean of Each Feature", fontsize=16)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Mean Value", fontsize=14)
    plt.xticks(rotation=45)  # Rotate feature names for better visibility
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()   
def Model_Training2():
    data = pd.read_csv("test.csv")
    data.head()

    data = data.dropna()

    """Feature Selection => Manual"""
    x = data.drop(['Average_Packet_Size', 'Duration','Label'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['Label']
    print(type(y))
    x.shape

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=123)

    # Replace SVM with Decision Tree Classifier
    start_time = time.time()
    decision_tree_classifier = DecisionTreeClassifier(random_state=123)
    decision_tree_classifier.fit(x_train, y_train)
    
    y_pred = decision_tree_classifier.predict(x_test)
    
    print(y_pred)
    training_time = time.time() - start_time
    print("=" * 40)
    print("==========")
    print("Classification Report : ", classification_report(y_test, y_pred))
    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Training Time: %.2f seconds" % training_time)
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))

    label4 = tk.Label(root, text=str(repo), width=45, height=15, bg='seashell2', fg='black', font=("Tempus Sanc ITC", 14))
    label4.place(x=250, y=100)

    label5 = tk.Label(
        root, 
        text=f"Accuracy: {ACC:.2f}%\nTraining Time: {training_time:.2f} seconds\nModel saved as attack_DecisionTree.joblib", 
        width=45, 
        height=5, 
        bg='khaki', 
        fg='black', 
        font=("Tempus Sanc ITC", 14)
    )
    label5.place(x=250, y=420)

    dump(decision_tree_classifier, "attack_DecisionTree.joblib")
    print("Model saved as attack_DecisionTree.joblib")
 # Calculate the mean of each feature
    feature_means = x.mean()

    # Create the bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(feature_means.index, feature_means.values, color="skyblue")
    plt.title("Mean of Each Feature", fontsize=16)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Mean Value", fontsize=14)
    plt.xticks(rotation=45)  # Rotate feature names for better visibility
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show() 
def Model_Training3():
    data = pd.read_csv("test.csv")
    data.head()

    data = data.dropna()

    """Feature Selection => Manual"""
    x = data.drop(['Average_Packet_Size', 'Duration','Label'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['Label']
    print(type(y))
    x.shape

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=123)

    # Replace SVM with Multinomial Naive Bayes Classifier
    start_time = time.time()
    naive_bayes_classifier = MultinomialNB()
    naive_bayes_classifier.fit(x_train, y_train)
    
    y_pred = naive_bayes_classifier.predict(x_test)
    
    print(y_pred)
    training_time = time.time() - start_time
    print("=" * 40)
    print("==========")
    print("Classification Report : ", classification_report(y_test, y_pred))
    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Training Time: %.2f seconds" % training_time)
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))

    label4 = tk.Label(root, text=str(repo), width=45, height=15, bg='seashell2', fg='black', font=("Tempus Sanc ITC", 14))
    label4.place(x=250, y=100)

    label5 = tk.Label(
        root, 
        text=f"Accuracy: {ACC:.2f}%\nTraining Time: {training_time:.2f} seconds\nModel saved as attack_NaiveBayes.joblib", 
        width=45, 
        height=5, 
        bg='khaki', 
        fg='black', 
        font=("Tempus Sanc ITC", 14)
    )
    label5.place(x=250, y=420)

    dump(naive_bayes_classifier, "attack_NaiveBayes.joblib")
    print("Model saved as attack_NaiveBayes.joblib")
 # Calculate the mean of each feature
    feature_means = x.mean()

    # Create the bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(feature_means.index, feature_means.values, color="skyblue")
    plt.title("Mean of Each Feature", fontsize=16)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Mean Value", fontsize=14)
    plt.xticks(rotation=45)  # Rotate feature names for better visibility
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()    
def Model_Training4():
    data = pd.read_csv("test.csv")
    data.head()

    data = data.dropna()

    """Feature Selection => Manual"""
    x = data.drop(['Average_Packet_Size', 'Duration','Label'], axis=1)
    data = data.dropna()

    print(type(x))
    y = data['Label']
    print(type(y))
    x.shape

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=123)

    # Replace SVM with XGBoost Classifier
    start_time = time.time()
    xgboost_classifier = xgb.XGBClassifier()
    xgboost_classifier.fit(x_train, y_train)
    training_time = time.time() - start_time
    y_pred = xgboost_classifier.predict(x_test)
    print(y_pred)

    print("=" * 40)
    print("==========")
    print("Classification Report : ", classification_report(y_test, y_pred))
    print("Accuracy : ", accuracy_score(y_test, y_pred) * 100)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("Training Time: %.2f seconds" % training_time)
    ACC = (accuracy_score(y_test, y_pred) * 100)
    repo = (classification_report(y_test, y_pred))

    label4 = tk.Label(root, text=str(repo), width=45, height=15, bg='seashell2', fg='black', font=("Tempus Sanc ITC", 14))
    label4.place(x=250, y=100)

    label5 = tk.Label(
        root, 
        text=f"Accuracy: {ACC:.2f}%\nTraining Time: {training_time:.2f} seconds\nModel saved as attack_XGBoost.joblib", 
        width=45, 
        height=5, 
        bg='khaki', 
        fg='black', 
        font=("Tempus Sanc ITC", 14)
    )
    label5.place(x=250, y=420)

    dump(xgboost_classifier, "attack_XGBoost.joblib")
    print("Model saved as attack_XGBoost.joblib")   
  # Calculate the mean of each feature
    feature_means = x.mean()

     # Create the bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(feature_means.index, feature_means.values, color="skyblue")
    plt.title("Mean of Each Feature", fontsize=16)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Mean Value", fontsize=14)
    plt.xticks(rotation=45)  # Rotate feature names for better visibility
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show() 

def Model_Training_Hybrid():
    data = pd.read_csv("test.csv")
    data = data.dropna()

    # Feature Selection (Manual)
    x = data.drop(['Average_Packet_Size', 'Duration', 'Label'], axis=1)
    y = data['Label']

    # Convert categorical labels to numerical if needed
    y = y.astype(int)

    # Splitting Data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=123)

    # Reshape data for CNN and LSTM (adding channel dimension)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # CNN + BLSTM Model
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training
    start_time = time.time()
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
    training_time = time.time() - start_time

    # Evaluation
    y_pred = (model.predict(x_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)

    # Display results in GUI
    label4 = tk.Label(root, text=str(classification_report(y_test, y_pred)), width=45, height=15, bg='seashell2', fg='black', font=("Tempus Sanc ITC", 14))
    label4.place(x=250, y=100)

    label5 = tk.Label(
        root, 
        text=f"Accuracy: {accuracy * 100:.2f}%\nTraining Time: {training_time:.2f} seconds\nModel saved as attack_CNN.joblib", 
        width=45, 
        height=5, 
        bg='khaki', 
        fg='black', 
        font=("Tempus Sanc ITC", 14)
    )
    label5.place(x=250, y=420)

    model.save("attack_CNN.joblib")
    print("Model saved as attack_CNN_BLSTM.h5")

    # Feature Mean Visualization
    feature_means = x.mean()
    plt.figure(figsize=(12, 6))
    plt.bar(feature_means.index, feature_means.values, color="skyblue")
    plt.title("Mean of Each Feature", fontsize=16)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Mean Value", fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

#Create the bar chart
    





def window():
    root.destroy()
    
def plot():
    from subprocess import call
    call(["python","plot.py"])      

def call_file():
    from subprocess import call
    call(["python","checkk.py"])      

 

# button2 = tk.Button(root, foreground="white", background="black", font=("Tempus Sans ITC", 14, "bold"),
#                     text="Data_Preprocessing", command=Data_Preprocessing, width=15, height=2)
# button2.place(x=5, y=120)

# Adjust x position to fit buttons on the right image
right_image_x = 1100  # Adjust based on image position

button_svm = tk.Button(root, foreground="white", background="#560319", font=("Tempus Sans ITC", 14, "bold"),
                       text="Model_SVM", command=Model_Training, width=15, height=2)
button_svm.place(x=right_image_x, y=100)

button_rf = tk.Button(root, foreground="white", background="#560319", font=("Tempus Sans ITC", 14, "bold"),
                      text="Model_RF", command=Model_Training1, width=15, height=2)
button_rf.place(x=right_image_x, y=170)

button_dt = tk.Button(root, foreground="white", background="#560319", font=("Tempus Sans ITC", 14, "bold"),
                      text="Model_DT", command=Model_Training2, width=15, height=2)
button_dt.place(x=right_image_x, y=240)

button_nb = tk.Button(root, foreground="white", background="#560319", font=("Tempus Sans ITC", 14, "bold"),
                      text="Model_NB", command=Model_Training3, width=15, height=2)
button_nb.place(x=right_image_x, y=310)

button_xgboost = tk.Button(root, foreground="white", background="#560319", font=("Tempus Sans ITC", 14, "bold"),
                           text="Model_xgboost", command=Model_Training4, width=15, height=2)
button_xgboost.place(x=right_image_x, y=380)

button_hybrid = tk.Button(root, foreground="white", background="#560319", font=("Tempus Sans ITC", 14, "bold"),
                          text="Train Hybrid Model", command=Model_Training_Hybrid, width=20, height=2)
button_hybrid.place(x=right_image_x - 15, y=450)  # Placed right after button_xgboost

button_graph = tk.Button(root, foreground="white", background="#560319", font=("Tempus Sans ITC", 14, "bold"),
                         text="Graph", command=plot, width=15, height=2)
button_graph.place(x=right_image_x, y=520)

button_performance = tk.Button(root, foreground="white", background="#560319", font=("Tempus Sans ITC", 14, "bold"),
                               text="Check Performance", command=call_file, width=15, height=2)
button_performance.place(x=right_image_x, y=590)

exit_button = tk.Button(root, text="Exit", command=window, width=15, height=2, font=('times', 15, ' bold '),
                        bg="red", fg="white")
exit_button.place(x=right_image_x, y=660)

root.mainloop()

'''+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''