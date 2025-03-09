import nest_asyncio

nest_asyncio.apply()

from flask import Flask, request, render_template, jsonify, session
import imaplib
import email
import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Logging setup
logging.basicConfig(level=logging.INFO)

# Configuration
IMAP_URL = "imap.gmail.com"
DATASET_FILE = "mail_datasets.csv"


# Train the spam detection model
def train_model():
    df = pd.read_csv(DATASET_FILE)
    data = df.where(pd.notnull(df), '')

    data.loc[data['Category'] == 'spam', 'Category'] = 0
    data.loc[data['Category'] == 'ham', 'Category'] = 1

    X = data['Message']
    Y = data['Category'].astype('int')

    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_features = feature_extraction.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_features, Y)

    return model, feature_extraction


# Train model once at startup
spam_model, vectorizer = train_model()


# Fetch latest email from the target sender
def fetch_latest_email(user, password, target_email):
    try:
        with imaplib.IMAP4_SSL(IMAP_URL) as mail:
            mail.login(user, password)
            mail.select("Inbox")

            search_criterion = f'FROM "{target_email}"'
            _, data = mail.search(None, search_criterion)

            mail_ids = data[0].split()
            if not mail_ids:
                logging.info(f"No emails found from {target_email}.")
                return []

            latest_email_id = mail_ids[-1]
            _, msg_data = mail.fetch(latest_email_id, "(RFC822)")
            email_bodies = []

            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            body = part.get_payload(decode=True).decode(part.get_content_charset(), errors="ignore")
                            email_bodies.append(body.strip())

            return email_bodies

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return []


# Classify emails as spam or ham
def classify_emails(email_bodies, model, feature_extraction):
    results = []
    for body in email_bodies:
        input_data_features = feature_extraction.transform([body])
        prediction = model.predict(input_data_features)[0]

        # Show email body only for spam, else display "It's a ham message!"
        if prediction == 0:  # Spam
            results.append(f'<p class="spam-text">SPAM 🚨: {body[:200]}...</p>')
        else:  # Ham
            results.append('<p class="ham-text">It\'s a HAM message! ✅</p>')

    return results


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        email_address = request.form["email"]
        app_password = request.form["password"]
        target_email = request.form["target_email"]

        # Store credentials temporarily in session
        session["email"] = email_address
        session["password"] = app_password

        emails = fetch_latest_email(email_address, app_password, target_email)
        predictions = classify_emails(emails, spam_model, vectorizer)

        return jsonify({"target_email": target_email, "predictions": predictions})

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
