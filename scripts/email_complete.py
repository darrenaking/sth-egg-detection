import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

def send_training_notification(subject, body):
    creds_path = "/workspace/.email_creds"

    if os.path.exists(creds_path):
        with open(creds_path, 'r') as f:
            email, password = f.read().splitlines()
    
    # Construct the email
    msg = MIMEMultipart()
    msg['From'] = email
    msg['To'] = email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        # Connect to Google's secure SMTP server on Port 465
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(email, password)
        server.send_message(msg)
        server.quit()
        print("Success: Notification email sent.")
    except Exception as e:
        print(f"Failed to send email: {e}")