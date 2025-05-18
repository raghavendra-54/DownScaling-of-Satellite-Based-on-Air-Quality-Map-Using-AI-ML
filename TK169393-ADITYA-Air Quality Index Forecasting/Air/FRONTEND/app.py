from flask import *
from tkinter import scrolledtext
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt 
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingClassifier
import plotly.graph_objects as go
import mysql.connector
db=mysql.connector.connect(user="root",password="",port='3306',database='Air1')
cur=db.cursor()
app=Flask(__name__)
app.secret_key="CBJcb786874wrf78chdchsdcv"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about', methods=['POST','GET'])
def about():
    return render_template('about.html')

@app.route('/register',methods=["POST","GET"])
def register():
    if request.method=='POST':
        firstname=request.form['firstname']
        lastname=request.form['lastname']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']        
        address = request.form['address']        
        contact = request.form['contact']
        
        sql="select * from user where useremail='%s' and userpassword='%s'"%(useremail,userpassword)
        cur.execute(sql)
        data=cur.fetchall()
        db.commit()
        print(data)
        if data==[]:            
            sql = "insert into user(firstname,lastname,useremail,userpassword,address,contact)values(%s,%s,%s,%s,%s,%s)"
            val=(firstname,lastname,useremail,userpassword,address,contact)
            cur.execute(sql,val)
            db.commit()
            flash("Registered successfully","success")
            return render_template("login.html")
        else:
            flash("Details are invalid","warning")
            return render_template("register.html")
    return render_template('register.html')

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        email = request.form['mail']
        password = request.form['passw']
        
        user = Register.query.filter_by(email=email, password=password).first()
        
        if user:
            session['user_id'] = user.id
            return redirect(url_for('userhome'))
        else:
            flash("Login failed", "warning")
            return render_template("login.html", msg="Login failed")
    
    return render_template("login.html")

@app.route('/userhome')
def userhome():
    return render_template('userhome.html')

@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test, df, data
    if request.method == "POST":
        df=pd.read_csv(r"city_day.csv")
        df.head()
        ### filiing the null values
        df['PM2.5'].fillna(df['PM2.5'].mean(), inplace=True)
        df['PM10'].fillna(df['PM10'].mean(), inplace=True)
        df['NO'].fillna(df['NO'].mean(), inplace=True)
        df['NO2'].fillna(df['NO2'].mean(), inplace=True)
        df['NOx'].fillna(df['NOx'].mean(), inplace=True)
        df['NH3'].fillna(df['NH3'].mean(), inplace=True)
        df['CO'].fillna(df['CO'].mean(), inplace=True)
        df['SO2'].fillna(df['SO2'].mean(), inplace=True)
        df['O3'].fillna(df['O3'].mean(), inplace=True)
        df['Benzene'].fillna(df['Benzene'].mean(), inplace=True)
        df['Toluene'].fillna(df['Toluene'].mean(), inplace=True)
        df['Xylene'].fillna(df['Xylene'].mean(), inplace=True)
        df['AQI'].fillna(df['AQI'].mean(), inplace=True)

        df["AQI_Bucket"].replace({'Moderate': 1, 'Satisfactory': 1, 'Poor': 0,
                        'Very Poor': 0,  'Good': 2, 'Severe': 2}, inplace=True)
        
        df['AQI_Bucket'].fillna(df['AQI_Bucket'].ffill(), inplace=True)
        df['AQI_Bucket'].fillna(df['AQI_Bucket'].bfill(), inplace=True)

        ### we drop the columns which is not that much importance 
        df.drop(["City", "Date"], axis=1,inplace=True)

        ### Splitting the Data
        x = df.drop("AQI_Bucket", axis=1)
        y = df["AQI_Bucket"]

        # Apply SMOTE 
        smote = SMOTE(random_state=42)
        x,y = smote.fit_resample(x, y)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')

@app.route('/model',methods=["GET","POST"])
def model():
    global acc_rf,acc_gb,acc_adb,x_train
    if request.method =='POST':
        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s = int(request.form['algo'])
        if s == 0:
            return render_template('model.html', msg='Please Choose an Algorithm to Train')
        
        if s == 1:
            from sklearn.ensemble import RandomForestClassifier
            rf = RandomForestClassifier()
            rf = rf.fit(x_train,y_train)
            y_pred = rf.predict(x_test)
            acc_rf=accuracy_score(y_test,y_pred)*100
            msg = 'The accuracy obtained by Random Forest Classifier is ' + str(acc_rf) + str('%')
            print("--------------------------------------------------")
            print(msg)
            return render_template('model.html',msg=msg)
        
        elif s ==2:
            from sklearn.tree import DecisionTreeClassifier
            dt = DecisionTreeClassifier()
            dt = dt.fit(x_train,y_train)
            y_pred = dt.predict(x_test)
            acc_dt=accuracy_score(y_test,y_pred)*100
            msg = 'The accuracy obtained by Decision Tree Classifier is ' + str(acc_dt) + str('%')
            return render_template('model.html',msg=msg)
        
        elif s ==3:
            from sklearn.ensemble import AdaBoostClassifier
            adb = AdaBoostClassifier()
            adb = adb.fit(x_train,y_train)
            y_pred = adb.predict(x_test)
            acc_adb=accuracy_score(y_test,y_pred)*100
            msg = 'The accuracy obtained by AdaBoost Classifier is ' + str(acc_adb) + str('%')
            return render_template('model.html',msg=msg)
        
        elif s ==4:
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier()
            knn = knn.fit(x_train,y_train)
            y_pred = knn.predict(x_test)
            acc_knn=accuracy_score(y_test,y_pred)*100
            msg = 'The accuracy obtained by K-Neighbors Classifier is ' + str(acc_knn) + str('%')
            return render_template('model.html',msg=msg)
    return render_template('model.html')

@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    global x, y, x_train, x_test, y_train, y_test
    msg = ""  # Initialize msg with an empty string
    
    if request.method == "POST":

        l1 = float(request.form['PM2.5'])
        l2 = float(request.form['PM10'])
        l3 = float(request.form['NO'])
        l4 = float(request.form['NO2'])
        l5 = float(request.form['NOx'])
        l6 = float(request.form['NH3'])
        l7 = float(request.form['CO'])
        l8 = float(request.form['SO2'])
        l9 = float(request.form['O3'])
        l10 = float(request.form['Benzene'])
        l11 = float(request.form['Toluene'])
        l12 = float(request.form['Xylene'])
        l13 = float(request.form['AQI'])

        lee=[l1,l2,l3,l4,l5,l6,l7,l8,l9,l10,l11,l12,l13]

        print(lee)

        from sklearn.ensemble import RandomForestClassifier
        model=RandomForestClassifier()
        global x_train,y_train
        model.fit(x_train,y_train)
        result=model.predict([lee])
        print(result)
        if result==0:
            msg="Air Quality Is Poor "
            return render_template('prediction.html', msg=msg)
        elif result==1:
            msg="Air Quality Is Satisfactory"
            return render_template('prediction.html', msg=msg)
        elif result==2:
            msg="Air Quality Is Good"
            return render_template('prediction.html', msg=msg)
    return render_template('prediction.html', msg=msg)
        

if __name__=='__main__':
    app.run(debug=True)