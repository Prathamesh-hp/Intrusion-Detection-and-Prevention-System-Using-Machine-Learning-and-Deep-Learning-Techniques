from tkinter import *
import tkinter as tk
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from joblib import load
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Dictionary to track attack counts
attack_counts = {"DOs": 0, "PortScan": 0, "Intrusion": 0, "WebAttack": 0, "BruteForce": 0, "BENIGN": 0}

def Train():
    root = tk.Tk()
    root.geometry("900x800")
    root.title("Intrusion Detection Using ML")
    root.configure(background="brown")
    
    Total_Fwd_Packets = tk.IntVar()
    Total_Backward_Packets = tk.IntVar()
    Down_Up_Ratio = tk.IntVar()
    act_data_pkt_fwd = tk.IntVar()
    min_seg_size_forward = tk.IntVar()
    
    prevention_label = tk.Label(root, text="", background="brown", foreground="white", font=('times', 14, 'bold'), wraplength=700)
    prevention_label.place(x=50, y=600)

    def send_email_alert(subject, body):
        sender_email = "truptipisal9130@gmail.com"
        receiver_email = "margaleprathamesht@gmail.com"
        app_password = "shlr cpxr wcxj gvit"

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(sender_email, app_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
            server.quit()
            print("Email alert sent successfully!")
        except Exception as e:
            print("Error sending email:", e)

    def show_preventions(attack_type):
        prevention_measures = {
            "DOs": "Mitigation: Use rate limiting, network segmentation, and firewall rules.",
            "PortScan": "Mitigation: Enable intrusion detection, restrict unnecessary ports, and monitor network activity.",
            "Intrusion": "Mitigation: Use strong authentication, monitor logs, and employ SIEM solutions.",
            "WebAttack": "Mitigation: Keep software updated, use WAFs, and sanitize user inputs.",
            "BruteForce": "Mitigation: Implement account lockouts, use CAPTCHAs, and enforce strong passwords."
        }
        prevention_label.config(text=prevention_measures.get(attack_type, "No specific prevention measures available."))

    def update_graph():
        attack_types = list(attack_counts.keys())
        counts = list(attack_counts.values())

        ax.clear()
        ax.bar(attack_types, counts, color=['red', 'blue', 'green', 'purple', 'orange', 'gray'])
        ax.set_title("Attack Detection Frequency")
        ax.set_ylabel("Count")
        ax.set_xlabel("Attack Type")
        ax.set_xticklabels(attack_types, rotation=30)

        canvas.draw()

    def Detect():
        e1 = Total_Fwd_Packets.get()
        e2 = Total_Backward_Packets.get()
        e3 = Down_Up_Ratio.get()
        e4 = act_data_pkt_fwd.get()
        e5 = min_seg_size_forward.get()

        model = load('C:/Users/iprat/Videos/BE Project Implementation/Final Code/attack_RandomForest.joblib')
        prediction = model.predict([[e1, e2, e3, e4, e5]])

        attack_labels = {
            1: "DOs",
            2: "PortScan",
            3: "Intrusion",
            4: "WebAttack",
            5: "BruteForce"
        }

        attack_type = attack_labels.get(prediction[0], "BENIGN")

        attack_counts[attack_type] += 1  # Update count
        update_graph()  # Refresh graph

        result_label = tk.Label(root, text=f"{attack_type} Attack Detected" if attack_type != "BENIGN" else "No Attack Detected",
                                background="green", foreground="white", font=('times', 20, 'bold'), width=32)
        result_label.place(x=200, y=500)

        if attack_type == "Intrusion":
            threading.Thread(target=send_email_alert, args=("🚨 Intrusion Alert!", "An Intrusion attack has been detected."), daemon=True).start()

        show_preventions(attack_type)

    labels = ["Total_Fwd_Packets", "Total_Backward_Packets", "Down_Up_Ratio", "act_data_pkt_fwd", "min_seg_size_forward"]
    vars = [Total_Fwd_Packets, Total_Backward_Packets, Down_Up_Ratio, act_data_pkt_fwd, min_seg_size_forward]

    for i, (label_text, var) in enumerate(zip(labels, vars)):
        tk.Label(root, text=label_text, background="seashell2", font=('times', 20, 'bold'), width=20).place(x=5, y=50 + i * 50)
        tk.Entry(root, bd=2, width=5, font=("TkDefaultFont", 20), textvar=var).place(x=400, y=50 + i * 50)

    tk.Button(root, text="Submit", command=Detect, font=('times', 20, 'bold'), width=10).place(x=300, y=400)

    # Matplotlib graph setup
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_title("Attack Detection Frequency")
    ax.set_ylabel("Count")
    ax.set_xlabel("Attack Type")

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().place(x=700, y=30)

    root.mainloop()

Train()
