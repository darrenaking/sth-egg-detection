import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_training_notification(subject, body):
    sender_email = ""
    receiver_email = ""
    app_password = "" 
    
    # Construct the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        # Connect to Google's secure SMTP server on Port 465
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender_email, app_password)
        server.send_message(msg)
        server.quit()
        print("Success: Notification email sent.")
    except Exception as e:
        print(f"Failed to send email: {e}")