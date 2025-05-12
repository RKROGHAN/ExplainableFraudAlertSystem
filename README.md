
1. Import Required Libraries
import random
from sklearn.ensemble import IsolationForest
from transformers import pipeline
•	random: (Included but unused) typically used to add randomness to simulations or sampling.
•	IsolationForest: A machine learning algorithm from Scikit-learn for unsupervised anomaly detection.
•	pipeline: Hugging Face utility to easily access pre-trained NLP models (like GPT-2 for text generation).
________________________________________
2. Define Sample Transaction Data
transactions = [
    {"amount": 120.5, "location": "New York", "time": 14},
    {"amount": 20.0, "location": "New York", "time": 9},
    {"amount": 3050.0, "location": "Houston", "time": 2},
    {"amount": 150.0, "location": "Chicago", "time": 12},
    {"amount": 10.0, "location": "New York", "time": 10},
    {"amount": 5000.0, "location": "Los Angeles", "time": 1},
]
Each transaction includes:
•	amount: Amount of the transaction
•	location: Where the transaction occurred
•	time: Hour of transaction (24-hour format)
________________________________________
3. Define Trusted Locations
trusted_users = ["New York", "Chicago"]
•	These locations are considered usual or trusted for the user.
•	This list helps build context-based rules for trust scoring.
________________________________________
4. Feature Extraction Function
def extract_features(tx):
    return [tx["amount"], tx["time"]]
•	Simplifies the transaction into a 2D feature vector: [amount, time].
•	Used as input for the Isolation Forest model.
________________________________________
5. Prepare Feature Matrix
X = [extract_features(tx) for tx in transactions]
•	Applies the extract_features() function to all transactions.
•	This creates the dataset used for training the anomaly detection model.
________________________________________
6. Train Isolation Forest Anomaly Detector
model = IsolationForest(contamination=0.3, random_state=42)
model.fit(X)
•	Detects abnormal transactions assuming 30% are potential anomalies.
•	random_state ensures reproducibility.
•	fit(X) trains the model on transaction behavior patterns.
________________________________________
7. Load NLP Model for Explanation
nlp_model = pipeline('text-generation', model='distilgpt2')
•	Loads a lightweight GPT-2 model for text generation.
•	Will be used to generate plain English explanations for flagged transactions.
________________________________________
8. Trust Score Function
def trust_score(transaction):
    score = 100
    if transaction["location"] not in trusted_users:
        score -= 30
    if transaction["time"] < 6 or transaction["time"] > 22:
        score -= 20
    if transaction["amount"] > 1000:
        score -= 30
    return max(score, 0)
•	Starts with a perfect trust score of 100.
•	Deducts based on risky patterns:
o	Unfamiliar location: −30
o	Odd hours (before 6 or after 22): −20
o	High amount (>1000): −30
•	Returns a final trust score between 0 and 100.
________________________________________
9. Generate Natural Language Explanation
def generate_explanation(transaction, score):
    prompt = (
        f"The transaction of ${transaction['amount']} at {transaction['time']}h "
        f"in {transaction['location']} was flagged. Trust score: {score}/100. "
        f"Explain why it may be suspicious in simple terms:\n"
    )
    output = nlp_model(prompt, max_length=100, num_return_sequences=1)
    return output[0]["generated_text"]
•	Constructs a text prompt including transaction details and trust score.
•	Passes it to the GPT-2 model.
•	Returns a simple, readable explanation.
________________________________________
10. Analyze and Explain Transactions
for i, tx in enumerate(transactions):
    score = trust_score(tx)
    prediction = model.predict([extract_features(tx)])
    is_anomaly = prediction[0] == -1
•	Iterates through each transaction.
•	Computes the trust score.
•	Predicts whether the transaction is an anomaly (-1) or normal (1).
________________________________________
11. Output Results
print(f"\n--- Transaction #{i+1} ---")
print(f"Details: {tx}")
if is_anomaly:
    print(f"Flagged as suspicious.")
    print(f"Trust Score: {score}/100")
    explanation = generate_explanation(tx, score)
    print("Explanation:")
    print(explanation)
else:
    print("Transaction appears normal.")
•	For each transaction:
o	Prints details.
o	If flagged as suspicious, shows:
	Trust score
	Natural language explanation
o	Else, confirms it appears normal.
________________________________________
Summary
This system combines machine learning and natural language processing to:
•	Detect fraudulent transactions using Isolation Forest.
•	Score trust with a custom, human-understandable logic.
•	Explain flagged transactions using GPT-2-based language generation.
It’s designed to bridge the gap between AI predictions and user transparency, making fraud detection explainable, fair, and easy to understand.
________________________________________

