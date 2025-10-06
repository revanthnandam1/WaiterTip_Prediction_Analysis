import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    """Preprocess the dataset for modeling."""
    # Convert categorical variables to numerical
    label_encoders = {}
    for column in ["sex", "smoker", "day", "time"]:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    
    # Drop the 'size' column if it exists
    if "size" in data.columns:
        data = data.drop(columns=["size"])
    
    return data, label_encoders