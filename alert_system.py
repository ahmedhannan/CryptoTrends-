# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart
# import datetime
# from config import (
#     ENABLE_EMAIL_ALERTS, ALERT_EMAIL_FROM, ALERT_EMAIL_TO, LOG_FILE,
#     ALERT_SMTP_SERVER, ALERT_SMTP_PORT, ALERT_SMTP_USER, ALERT_SMTP_PASSWORD, TARGET_CRYPTOS
# )

# def send_alert(subject_line, alert_body_text, severity_level="INFO", crypto_context=None):
#     """
#     Emails are sent only for Price Predictions, Anomalies, and Portfolio Actions.
#     """
#     timestamp_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     full_subject = f"CryptoAgent [{severity_level}]"
#     if crypto_context:
#         full_subject += f" ({crypto_context})"
#     full_subject += f": {subject_line}"

#     console_log_message = f"{timestamp_now} {full_subject}\n   Body: {alert_body_text.replace('\\n', '\\n         ')}"
#     print(console_log_message)

#     try:
#         with open(LOG_FILE, 'a') as f_log:
#             f_log.write(console_log_message + "\n\n")
#     except Exception as e_log:
#         print(f"ERROR: Could not write to log file {LOG_FILE}: {e_log}")

#     if ENABLE_EMAIL_ALERTS:
#         # Send emails only for specific cases
#         should_send_email = (
#             (severity_level == "INFO" and subject_line == "Price Prediction") or
#             (severity_level == "CRITICAL" and "Anomaly Detected" in subject_line) or
#             (severity_level == "ACTION" and "Portfolio Rebalance" in subject_line)
#         )
#         if not should_send_email:
#             print(f"INFO: Skipping email for {full_subject} (not a prediction, anomaly, or portfolio action).")
#             return

#         try:
#             email_msg = MIMEMultipart()
#             email_msg['From'] = ALERT_EMAIL_FROM
#             if isinstance(ALERT_EMAIL_TO, list):
#                 email_msg['To'] = ", ".join(ALERT_EMAIL_TO)
#                 recipients_list = ALERT_EMAIL_TO
#             else:
#                 email_msg['To'] = ALERT_EMAIL_TO
#                 recipients_list = [ALERT_EMAIL_TO]
#             email_msg['Subject'] = full_subject

#             email_body_content = f"Timestamp: {timestamp_now}\nSeverity: {severity_level}\n"
#             if crypto_context:
#                 email_body_content += f"Context: {crypto_context}\n"
#             email_body_content += "\nDetails:\n" + alert_body_text
#             email_msg.attach(MIMEText(email_body_content, 'plain'))

#             with smtplib.SMTP(ALERT_SMTP_SERVER, ALERT_SMTP_PORT) as smtp_server:
#                 smtp_server.ehlo()
#                 smtp_server.starttls()
#                 smtp_server.ehlo()
#                 smtp_server.login(ALERT_SMTP_USER, ALERT_SMTP_PASSWORD)
#                 smtp_server.sendmail(ALERT_EMAIL_FROM, recipients_list, email_msg.as_string())
#             print(f"INFO: Email alert sent successfully to {recipients_list}.")
#         except Exception as e_email:
#             print(f"ERROR: Failed to send email alert. Subject: '{full_subject}'. Error: {e_email}")
#             try:
#                 with open(LOG_FILE, 'a') as f_log_err:
#                     f_log_err.write(f"{timestamp_now} [EMAIL_ERROR] Failed to send email: {full_subject}. Error: {e_email}\n")
#             except:
#                 pass

# if __name__ == '__main__':
#     print("--- Testing Alert System ---")
#     # Test alerts for each cryptocurrency in TARGET_CRYPTOS
#     for crypto in TARGET_CRYPTOS:
#         send_alert(
#             subject_line="Price Prediction",
#             alert_body_text=f"{crypto} predicted to reach $200.00 in 1 hour.\nCurrent Price: $180.00",
#             severity_level="INFO",
#             crypto_context=crypto
#         )
#         send_alert(
#             subject_line="Anomaly Detected!",
#             alert_body_text=f"{crypto} experienced an unusual spike in volume.\nReconstruction Error: 0.85",
#             severity_level="CRITICAL",
#             crypto_context=crypto
#         )
#     # Test portfolio rebalance (no specific crypto context)
#     send_alert(
#         subject_line="Portfolio Rebalance Suggestions",
#         alert_body_text="Target allocations updated.\n" + "\n".join(
#             [f"{crypto}: 15%" for crypto in TARGET_CRYPTOS]
#         ),
#         severity_level="ACTION"
#     )
#     # Test a non-email alert
#     send_alert(
#         subject_line="Model Training Started",
#         alert_body_text=f"Training anomaly detector for {TARGET_CRYPTOS[0]}.",
#         severity_level="INFO",
#         crypto_context=TARGET_CRYPTOS[0]
#     )







#Updated Code
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import datetime
from config import (
    ENABLE_EMAIL_ALERTS,
    ALERT_EMAIL_FROM,
    ALERT_EMAIL_TO,
    LOG_FILE,
    ALERT_SMTP_SERVER,
    ALERT_SMTP_PORT,
    ALERT_SMTP_USER,
    ALERT_SMTP_PASSWORD,
    TARGET_CRYPTOS
)

def send_alert(subject_line, alert_body_text, severity_level="INFO", crypto_context=None):
    """
    Emails are sent only for Price Predictions, Anomalies, and Portfolio Actions.
    """
    # 1) Timestamp
    timestamp_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 2) Construct full subject
    full_subject = f"CryptoAgent [{severity_level}]"
    if crypto_context:
        full_subject += f" ({crypto_context})"
    full_subject += f": {subject_line}"

    # 3) Prepare indented body for console and log readability
    indented_body = alert_body_text.replace("\n", "\n         ")
    console_log_message = (
        f"{timestamp_now} {full_subject}\n"
        f"   Body: {indented_body}"
    )
    print(console_log_message)

    # 4) Append to log file
    try:
        with open(LOG_FILE, 'a') as f_log:
            f_log.write(console_log_message + "\n\n")
    except Exception as e_log:
        print(f"ERROR: Could not write to log file {LOG_FILE}: {e_log}")

    # 5) Email alerts based on severity and subject
    if ENABLE_EMAIL_ALERTS:
        should_send_email = (
            (severity_level == "INFO" and subject_line == "Price Prediction") or
            (severity_level == "CRITICAL" and "Anomaly Detected" in subject_line) or
            (severity_level == "ACTION" and "Portfolio Rebalance" in subject_line)
        )
        if not should_send_email:
            print(
                f"INFO: Skipping email for {full_subject} "
                "(not a prediction, anomaly, or portfolio action)."
            )
            return

        # 6) Build and send email
        try:
            email_msg = MIMEMultipart()
            email_msg['From'] = ALERT_EMAIL_FROM
            if isinstance(ALERT_EMAIL_TO, list):
                email_msg['To'] = ", ".join(ALERT_EMAIL_TO)
                recipients_list = ALERT_EMAIL_TO
            else:
                email_msg['To'] = ALERT_EMAIL_TO
                recipients_list = [ALERT_EMAIL_TO]
            email_msg['Subject'] = full_subject

            email_body_content = (
                f"Timestamp: {timestamp_now}\n"
                f"Severity: {severity_level}\n"
            )
            if crypto_context:
                email_body_content += f"Context: {crypto_context}\n"
            email_body_content += "\nDetails:\n" + alert_body_text
            email_msg.attach(MIMEText(email_body_content, 'plain'))

            with smtplib.SMTP(ALERT_SMTP_SERVER, ALERT_SMTP_PORT) as smtp_server:
                smtp_server.ehlo()
                smtp_server.starttls()
                smtp_server.ehlo()
                smtp_server.login(ALERT_SMTP_USER, ALERT_SMTP_PASSWORD)
                smtp_server.sendmail(
                    ALERT_EMAIL_FROM,
                    recipients_list,
                    email_msg.as_string()
                )
            print(f"INFO: Email alert sent successfully to {recipients_list}.")
        except Exception as e_email:
            print(f"ERROR: Failed to send email alert. Subject: '{full_subject}'. Error: {e_email}")
            try:
                with open(LOG_FILE, 'a') as f_log_err:
                    f_log_err.write(
                        f"{timestamp_now} [EMAIL_ERROR] Failed to send email: {full_subject}. Error: {e_email}\n"
                    )
            except:
                pass

if __name__ == '__main__':
    print("--- Testing Alert System ---")
    # Test alerts for each cryptocurrency in TARGET_CRYPTOS
    for crypto in TARGET_CRYPTOS:
        send_alert(
            subject_line="Price Prediction",
            alert_body_text=(
                f"{crypto} predicted to reach $200.00 in 1 hour.\n"
                f"Current Price: $180.00"
            ),
            severity_level="INFO",
            crypto_context=crypto
        )
        send_alert(
            subject_line="Anomaly Detected!",
            alert_body_text=(
                f"{crypto} experienced an unusual spike in volume.\n"
                f"Reconstruction Error: 0.85"
            ),
            severity_level="CRITICAL",
            crypto_context=crypto
        )
    # Test portfolio rebalance (no specific crypto context)
    send_alert(
        subject_line="Portfolio Rebalance Suggestions",
        alert_body_text=(
            "Target allocations updated.\n" +
            "\n".join([f"{crypto}: 15%" for crypto in TARGET_CRYPTOS])
        ),
        severity_level="ACTION"
    )
    # Test a non-email alert
    send_alert(
        subject_line="Model Training Started",
        alert_body_text=(
            f"Training anomaly detector for {TARGET_CRYPTOS[0]}."
        ),
        severity_level="INFO",
        crypto_context=TARGET_CRYPTOS[0]
    )
