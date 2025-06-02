🚨 Intrusion Detection and Prevention System using Machine Learning & Deep Learning
This project is a comprehensive Intrusion Detection and Prevention System (IDPS) that uses machine learning and deep learning techniques to identify and categorize various types of network intrusions such as DoS, Port Scan, Web Attacks, Brute Force, and more.

🧠 Features
🌐 GUI-based interface using Tkinter

👤 User/Admin login & registration system with SQLite database

📊 Model training with:

SVM

Random Forest

Decision Tree

Naive Bayes

XGBoost

Hybrid CNN + BiLSTM

📈 Visual performance reports using matplotlib and seaborn

✉️ Email alert for Intrusion detection

📹 Video background with TkVideo

📁 Project Structure
.
├── admin.py              # Main launcher GUI (Admin/User panel)
├── Check.py              # Detection input form and prediction logic
├── GUI.py                # User GUI (Login/Register options)
├── GUI_main.py           # Admin GUI (Login/Register options)
├── GUI_MASTER.py         # Model training, hybrid model, graph plots
├── login.py              # Admin login system
├── login1.py             # User login system
├── reg1.py               # User registration system
├── plot.py               # Graph visualization GUI
├── test.csv              # Dataset for training models (must be added manually)
├── *.joblib              # Saved models (generated after training)
└── *.jpg / *.mov / *.png # Media files used in the GUI

🚀 How to Run
pip install -r requirements.txt

Run the main interface
python admin.py

Train Models
Use the GUI to train models by selecting different options (SVM, RF, NB, etc.).

Test Intrusions
From the User/Admin panel, provide input values to check for intrusions.

🔐 Credentials
Databases:

reg.db – for user credentials

evaluation.db – for admin credentials

🛡️ Attack Categories Handled
BENIGN

DOS

PortScan

WebAttack

BruteForce

Intrusion (triggers an email alert)

📬 Email Alert Configuration
Update the sender email and app password in Check.py:
sender_email = "your_email@gmail.com"
app_password = "your_app_password"  # Use an App Password
receiver_email = "receiver_email@gmail.com"

🧰 Requirements
numpy
pandas
scikit-learn
matplotlib
seaborn
tkinter
Pillow
joblib
xgboost
tensorflow

📖 License
This project is developed for academic purposes. Please check with your institution before publishing any derived work.
