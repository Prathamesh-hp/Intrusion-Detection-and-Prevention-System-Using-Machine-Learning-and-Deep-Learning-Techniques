Here's the GitHub README file text you can directly copy and save:

---

# 🚨 Intrusion Detection and Prevention System (IDPS) using Machine Learning & Deep Learning

This project presents a robust Intrusion Detection and Prevention System (IDPS) that leverages machine learning and deep learning techniques to accurately identify and classify various network intrusion types, including DoS, Port Scan, Web Attacks, Brute Force, and more.

## 🧠 Features

* **🌐 GUI-based Interface:** User-friendly interface built with Tkinter.
* **👤 User/Admin System:** Secure login and registration for users and administrators, powered by an SQLite database.
* **📊 Model Training:** Supports training with a variety of powerful algorithms:
    * SVM (Support Vector Machine)
    * Random Forest
    * Decision Tree
    * Naive Bayes
    * XGBoost
    * Hybrid CNN + BiLSTM (Convolutional Neural Network + Bidirectional Long Short-Term Memory)
* **📈 Visual Performance Reports:** Generate insightful performance graphs using Matplotlib and Seaborn.
* **✉️ Email Alert System:** Configurable email alerts for detected intrusions.
* **📹 Video Background:** Enhanced GUI with video backgrounds using TkVideo.

## 📁 Project Structure

```
.
├── admin.py            # Main launcher GUI (Admin/User panel)
├── Check.py            # Detection input form and prediction logic
├── GUI.py              # User GUI (Login/Register options)
├── GUI_main.py         # Admin GUI (Login/Register options)
├── GUI_MASTER.py       # Model training, hybrid model, graph plots
├── login.py            # Admin login system
├── login1.py           # User login system
├── reg1.py             # User registration system
├── plot.py             # Graph visualization GUI
├── test.csv            # Dataset for training models (must be added manually)
├── *.joblib            # Saved models (generated after training)
└── *.jpg / *.mov / *.png # Media files used in the GUI
```

## 🚀 How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Main Interface:**
    ```bash
    python admin.py
    ```

3.  **Train Models:**
    Use the GUI to train models by navigating through the options (SVM, Random Forest, Naive Bayes, etc.).

4.  **Test Intrusions:**
    From either the User or Admin panel, provide input values to check for network intrusions.

## 🔐 Credentials

This project uses SQLite databases for credential management:

* `reg.db`: Stores user credentials.
* `evaluation.db`: Stores admin credentials.

## 🛡️ Attack Categories Handled

The system is designed to detect and categorize the following types of network traffic:

* `BENIGN`
* `DOS`
* `PortScan`
* `WebAttack`
* `BruteForce`
* `Intrusion` (triggers an email alert)

## 📬 Email Alert Configuration

To enable email alerts, update the sender email and an **App Password** in `Check.py`:

```python
sender_email = "your_email@gmail.com"
app_password = "your_app_password"  # Use an App Password for security
receiver_email = "receiver_email@gmail.com" # The email address to receive alerts
```

**Important:** For Gmail, you'll need to generate an App Password. Do not use your regular Gmail account password directly in the code.

## 🧰 Requirements

The project relies on the following Python libraries:

* `numpy`
* `pandas`
* `scikit-learn`
* `matplotlib`
* `seaborn`
* `tkinter`
* `Pillow`
* `joblib`
* `xgboost`
* `tensorflow`

## 📖 License

This project is developed for academic purposes. Please ensure you comply with your institution's policies before publishing any derived work.
