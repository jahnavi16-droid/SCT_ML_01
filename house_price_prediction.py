# house_price_prediction.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -------------------------------
# 1. Generate Dataset Automatically
# -------------------------------

np.random.seed(42)
n = 100

square_feet = np.random.randint(500, 3000, n)
bedrooms = np.random.randint(1, 6, n)
bathrooms = np.random.randint(1, 4, n)

# Price formula with noise
price = (
    square_feet * 3000 +
    bedrooms * 50000 +
    bathrooms * 30000 +
    np.random.randint(-100000, 100000, n)
)

data = pd.DataFrame({
    "SquareFeet": square_feet,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Price": price
})

print("\nSample Dataset:\n")
print(data.head())

# -------------------------------
# 2. Train Linear Regression Model
# -------------------------------

X = data[["SquareFeet", "Bedrooms", "Bathrooms"]]
y = data["Price"]

model = LinearRegression()
model.fit(X, y)

print("\nModel Training Completed!")

# -------------------------------
# 3. Take User Input
# -------------------------------

print("\nEnter House Details to Predict Price")

sqft = float(input("Enter Square Feet: "))
bed = int(input("Enter Number of Bedrooms: "))
bath = int(input("Enter Number of Bathrooms: "))

# FIXED: Use DataFrame instead of NumPy array
user_input = pd.DataFrame(
    [[sqft, bed, bath]],
    columns=["SquareFeet", "Bedrooms", "Bathrooms"]
)

predicted_price = model.predict(user_input)

print("\nPredicted House Price: ₹", round(predicted_price[0], 2))

# -------------------------------
# 4. Visual Graph
# -------------------------------

plt.figure()
plt.scatter(data["SquareFeet"], data["Price"])
plt.xlabel("Square Feet")
plt.ylabel("Price")
plt.title("House Price Prediction (Square Feet vs Price)")

# Mark predicted point
plt.scatter(sqft, predicted_price[0], marker='x', s=150)

plt.show()