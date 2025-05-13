# X as 1D
import pandas as pd
from sklearn.model_selection import train_test_split

# Sample dataset
data = {
    "email_text": ["Buy now!", "Hello friend", "Limited offer!", "Meeting at 3 PM"],
    "spam": [1, 0, 1, 0],
}  # 1 = Spam, 0 = Not Spam
df = pd.DataFrame(data)

X = df["email_text"]
y = df["spam"]
