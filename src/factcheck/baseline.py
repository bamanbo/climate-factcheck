from collections import Counter 

def majority_class(y_train):
    """Return the most frequent label."""
    return Counter(y_train).most_common(1)[0][0] #e.g., 'SUPPORTS'

def predict_majority(claims, majority_label):
    """Predict the majority label for every claim."""
    return [majority_label] * len(claims)