from sklearn.ensemble import IsolationForest

# Sample transaction data
transactions = [
    {"amount": 120.5, "location": "New York", "time": 14},
    {"amount": 20.0, "location": "New York", "time": 9},
    {"amount": 3050.0, "location": "Houston", "time": 2},
    {"amount": 150.0, "location": "Chicago", "time": 12},
    {"amount": 10.0, "location": "New York", "time": 10},
    {"amount": 5000.0, "location": "Los Angeles", "time": 1},
]

trusted_locations = {"New York", "Chicago"}

# Extract numerical features for anomaly detection
def extract_features(tx):
    return [tx["amount"], tx["time"]]

X = [extract_features(tx) for tx in transactions]

# Train Isolation Forest
detector = IsolationForest(contamination=0.3, random_state=42)
detector.fit(X)

# Rule-based trust scoring system
def calculate_trust_score(tx):
    score = 100
    if tx["location"] not in trusted_locations:
        score -= 30
    if tx["time"] < 6 or tx["time"] > 22:
        score -= 20
    if tx["amount"] > 1000:
        score -= 30
    return max(score, 0)

# Rule-based explanation generator
def generate_explanation(tx, score):
    reasons = []
    if tx["location"] not in trusted_locations:
        reasons.append("The transaction occurred from an untrusted location.")
    if tx["time"] < 6 or tx["time"] > 22:
        reasons.append("It was made at an unusual time.")
    if tx["amount"] > 1000:
        reasons.append("The transaction amount is unusually high.")
    
    explanation = f"Transaction flagged. Trust score: {score}/100.\n"
    if reasons:
        explanation += "Reason(s):\n" + "\n".join(f"- {r}" for r in reasons)
    else:
        explanation += "No strong indicators of fraud, but model marked it as anomalous."
    return explanation

# Evaluation loop
for i, tx in enumerate(transactions):
    score = calculate_trust_score(tx)
    prediction = detector.predict([extract_features(tx)])
    is_suspicious = prediction[0] == -1

    print(f"\n--- Transaction #{i+1} ---")
    print(f"Details: {tx}")
    if is_suspicious:
        print("Flagged as suspicious.")
        print(generate_explanation(tx, score))
    else:
        print("Transaction appears normal.")
