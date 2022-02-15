# -*- coding: utf-8 -*-
#!/usr/bin/env python
# coding: utf-8

"""
INFORMATION RETRIEVAL AND WEB SEARCH ENGINES TERM PROJECT
DURDANE AVCI 18011040
YAREN ÖZBEY 18011022

"""

import pandas as pd
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def prepare_data(split_rate, file_name):
    data = pd.read_csv(file_name)
    labels = data.pop('Prediction')
    features = data
    email_number = features.pop('Email No.')    
    
    X_train,X_test,y_train,y_test = train_test_split(features,labels,test_size=split_rate,random_state=42)
    
    return X_train,X_test,y_train,y_test


def KNN_predict(split_rate, file_name):
    clf = KNeighborsClassifier()
    X_train,X_test,y_train,y_test = prepare_data(split_rate, file_name)
    
    clf.fit(X_train,y_train)
    test_pred_knn = clf.predict(X_test)
    accuracy = float(accuracy_score(test_pred_knn,y_test))
    confusion_matrix = metrics.confusion_matrix(y_test,  
                                            test_pred_knn)
    
    precision = confusion_matrix[1][1]/( confusion_matrix[0][1]+confusion_matrix[1][1] )    
    recall = confusion_matrix[1][1]/(confusion_matrix[1][0] + confusion_matrix[1][1] )
    f1score = ( 2*precision*recall ) / ( precision+recall )
    
    matrix_df = pd.DataFrame(confusion_matrix)
    #plot the result
    ax = plt.axes()
    sn.set(font_scale=1.3)
    plt.figure(figsize=(10,7))
    sn.heatmap(matrix_df, annot=True, fmt="g", ax=ax, cmap="magma")
    #set axis titles
    ax.set_title('Confusion Matrix - Naive Bayesian')
    ax.set_xlabel("Predicted label", fontsize =15)
    ax.set_ylabel("True Label", fontsize=15)
    plt.show()
    
    ax.figure.savefig("cfm.png")
    return str(accuracy), str(precision), str(recall), str(f1score)


def NaiveBayes_predict(split_rate, file_name):

    clf = GaussianNB()
    X_train,X_test,y_train,y_test = prepare_data(split_rate, file_name)
    
    clf.fit(X_train,y_train)
    test_pred_naive_bayesian = clf.predict(X_test)
    accuracy = float(accuracy_score(test_pred_naive_bayesian,y_test))
    confusion_matrix = metrics.confusion_matrix(y_test,  
                                            test_pred_naive_bayesian)

    precision = confusion_matrix[1][1]/( confusion_matrix[0][1]+confusion_matrix[1][1] )    
    recall = confusion_matrix[1][1]/(confusion_matrix[1][0] + confusion_matrix[1][1] )
    f1score = ( 2*precision*recall ) / ( precision+recall )    
    
    matrix_df = pd.DataFrame(confusion_matrix)
    #plot the result
    ax = plt.axes()
    sn.set(font_scale=1.3)
    plt.figure(figsize=(10,7))
    sn.heatmap(matrix_df, annot=True, fmt="g", ax=ax, cmap="magma")
    #set axis titles
    ax.set_title('Confusion Matrix - Naive Bayesian')
    ax.set_xlabel("Predicted label", fontsize =15)
    ax.set_ylabel("True Label", fontsize=15)
    plt.show()
    
    ax.figure.savefig("cfm.png")
    return str(accuracy), str(precision), str(recall), str(f1score)

def DecisionTree_predict(split_rate, file_name):

    clf = DecisionTreeClassifier(max_depth = 10, random_state = 42)
    X_train,X_test,y_train,y_test = prepare_data(split_rate, file_name)
    
    clf.fit(X_train,y_train)
    test_pred_decision_tree = clf.predict(X_test)
    accuracy = float(accuracy_score(test_pred_decision_tree,y_test))
    confusion_matrix = metrics.confusion_matrix(y_test,  
                                            test_pred_decision_tree)
    
    precision = confusion_matrix[1][1]/( confusion_matrix[0][1]+confusion_matrix[1][1] )    
    recall = confusion_matrix[1][1]/(confusion_matrix[1][0] + confusion_matrix[1][1] )
    f1score = ( 2*precision*recall ) / ( precision+recall )
    
    
    matrix_df = pd.DataFrame(confusion_matrix)
    #plot the result
    ax = plt.axes()
    sn.set(font_scale=1.3)
    plt.figure(figsize=(10,7))
    sn.heatmap(matrix_df, annot=True, fmt="g", ax=ax, cmap="magma")
    #set axis titles
    ax.set_title('Confusion Matrix - Decision-Tree')
    ax.set_xlabel("Predicted label", fontsize =15)
    ax.set_ylabel("True Label", fontsize=15)
    plt.show()
    
    ax.figure.savefig("cfm.png")
    return str(accuracy), str(precision), str(recall), str(f1score)


import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *    
from PyQt5.QtGui import * 



class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1600,900)

        Dialog.setStyleSheet("background-color: #dcdcdc")
  
      #  self.label = QtWidgets.QLabel(Dialog)
      #  self.label.setGeometry(QtCore.QRect(600, 0, 150, 31))
      #  self.label.setText("")
 
        font = QtGui.QFont('Cascadia Code', 12)
        
        self.labelT = QtWidgets.QLabel(Dialog)
        self.labelT.setGeometry(QtCore.QRect(500, 100, 200, 40))
        self.labelT.setText("DOSYA ADI: ")
        self.labelT.setFont(font)

        self.textbox = QLineEdit(Dialog)
        self.textbox.move(500, 150)
        self.textbox.resize(200,40)
        self.textbox.setStyleSheet("background-color: #f5f5f5" )
   
             
        self.labelA = QtWidgets.QLabel(Dialog)
        self.labelA.setGeometry(QtCore.QRect(500, 200, 200, 40))
        self.labelA.setText("ALGORİTMA SEÇİMİ: ")
        self.labelA.setFont(font)
   
        self.comboBox = QComboBox(Dialog)
        self.comboBox.setStyleSheet("background-color: #f5f5f5" )
        self.comboBox.setGeometry(QRect(500, 250, 200, 40))
        self.comboBox.setObjectName(("comboBox"))
        self.comboBox.addItem("KNN")
        self.comboBox.addItem("NAIVE-BAYESIAN")
        self.comboBox.addItem("DESICION TREE")

        
        self.labelS = QtWidgets.QLabel(Dialog)
        self.labelS.setGeometry(QtCore.QRect(500, 300, 200, 40))
        self.labelS.setText("TEST ORANI: (%)")
        self.labelS.setFont(font)

        self.spin = QSpinBox(Dialog)
        self.spin.setStyleSheet("background-color: #f5f5f5" )
        self.spin.setGeometry(500, 350, 200, 40)
      
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(500, 450, 200, 50))
  
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

        # adding signal and slot
        self.pushButton.clicked.connect(self.calculateAccuracy) #butona basıldığında fonksiyonu çalıştırıyor
        self.pushButton.setStyleSheet("background-color: #f5f5f5" )
        self.label_image = QLabel(Dialog)
        self.label_image.setGeometry(0, 0, 400, 900)
        self.label_image.setStyleSheet("background-color: white")
       
        # loading image
        self.pixmap = QPixmap('yeni.jpg')
        self.label_image.setPixmap(self.pixmap.scaled(400, 400, QtCore.Qt.KeepAspectRatio))     
        #self.label_image.resize(self.pixmap.width(), self.pixmap.height())      
       # self.pixmap = pixmap.scaled(50, 50, QtCore.Qt.KeepAspectRatio)
        self.label_cfm = QLabel(Dialog)
        self.label_cfm.setGeometry(800, 0,800, 900)
        self.label_cfm.setStyleSheet("background-color: #6e7b8b")
        
        self.label_sonuc = QtWidgets.QLabel(Dialog)
        self.label_sonuc.setGeometry(QtCore.QRect(850,80, 500, 31))
        self.label_sonuc.setText("")
        self.label_sonuc.setFont(QtGui.QFont('Daytona', 20))
        self.label_sonuc.setStyleSheet("background-color: #6e7b8b; color: white;")
        
        self.label_pmetrics = QtWidgets.QLabel(Dialog)
        self.label_pmetrics.setGeometry(QtCore.QRect(850,730, 500,170))
        self.label_pmetrics.setText("")
        self.label_pmetrics.setFont(QtGui.QFont('Daytona',18))
        self.label_pmetrics.setStyleSheet("background-color: #6e7b8b; color: white;")


    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.pushButton.setText(_translate("Dialog", "Click"))

    def calculateAccuracy(self):
        # slot
        value = self.spin.value()
        secim = self.comboBox.currentIndex()
        print(secim)
        file_name = self.textbox.text();
        
        if(secim == 0):
            accuracy, precision, recall, f1score = KNN_predict(value/100, file_name)   
        elif(secim == 1): 
            accuracy, precision, recall, f1score  = NaiveBayes_predict(value/100, file_name)   
        elif(secim == 2):
            accuracy, precision, recall, f1score  = DecisionTree_predict(value/100, file_name)   
    
        self.label_sonuc.setText("Accuracy: " + accuracy)       
        self.label_pmetrics.setText("Precision: " + precision + "<br>Recall: " + recall +"<br>F1 Score: " +  f1score)

        self.pixmap2 = QPixmap('cfm.png')
        self.label_cfm.setPixmap(self.pixmap2.scaled(800, 800, QtCore.Qt.KeepAspectRatio))       
        


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Dialog()

    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())










