# 🚖 Uber Ride Analytics & Prediction (Python | Pandas | Random Forest)

## 📘 Project Overview
This project performs **data analysis and machine learning** on an **Uber Ride Booking dataset** to uncover insights about ride behavior, customer preferences, payment trends, and predict booking status outcomes using a **Random Forest Classifier**.

The dataset was obtained from the **Kaggle Uber Ride Analytics Dashboard**, containing detailed information about bookings, ride distances, payment methods, and customer-driver ratings.

---

## 🧠 Objectives
- Analyze ride booking data and customer patterns.
- Clean and preprocess missing and inconsistent data.
- Visualize booking trends and key metrics using Seaborn and Matplotlib.
- Build a **Random Forest Model** to predict the **Booking Status** (e.g., Completed, Cancelled, No Driver Found).
- Identify the most important features influencing booking outcomes.

---

## 🧰 Technologies Used
| Tool / Library | Purpose |
|----------------|----------|
| **Python 3.11+** | Core programming language |
| **Pandas** | Data cleaning & manipulation |
| **NumPy** | Numerical computations |
| **Matplotlib / Seaborn** | Data visualization |
| **Scikit-learn** | Machine learning (RandomForest, Train/Test split, Pipelines) |

---

## 📂 Dataset Information
**Source:** `/kaggle/input/uber-ride-analytics-dashboard/ncr_ride_bookings.csv`  
**Rows:** 150,000  
**Columns:** 21  

### Key Columns
| Column Name | Description |
|--------------|-------------|
| `Date`, `Time` | Ride booking timestamp |
| `Booking Status` | Completed / Cancelled / No Driver Found / Incomplete |
| `Vehicle Type` | Auto, Sedan, Bike, etc. |
| `Pickup Location`, `Drop Location` | Ride origin and destination |
| `Booking Value` | Total cost of ride |
| `Ride Distance` | Distance covered in kilometers |
| `Driver Ratings`, `Customer Rating` | Feedback ratings |
| `Payment Method` | Cash, Card, UPI, Wallet, etc. |

---

## 🧹 Data Cleaning & Preprocessing
1. **Removed irrelevant columns:**
   - Cancelled rides and incomplete ride reason columns.
2. **Handled missing values:**
   - Used median for `Booking Value`, `Ride Distance`, `Avg VTAT`, `Avg CTAT`.
   - Used mean for `Driver Ratings`, `Customer Rating`.
   - Used mode for `Payment Method`.
3. **Converted columns:**
   - `Date` → datetime format  
   - `Time` → time format  
   - Converted categorical columns to dummy variables.
4. **Handled nulls and missing entries with SimpleImputer.**

---

## 📊 Exploratory Data Analysis (EDA)

### 1. **Booking Status Distribution**
Visualized the count of completed and cancelled rides.
```python
df['Booking Status'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')



2. Popular Vehicle Types
df['Vehicle Type'].value_counts().plot(kind='bar', color='orange', edgecolor='black')


🚗 Insight: Auto, Go Mini, and Go Sedan are the most used ride options.

3. Payment Method Distribution
df['Payment Method'].value_counts().plot(kind='pie', autopct='%1.1f%%')


💳 Insight: Over 60% of users prefer UPI payments, followed by Cash.

4. Revenue by Vehicle Type
df.groupby('Vehicle Type')['Booking Value'].sum().sort_values(ascending=False).plot(kind='bar')


💰 Insight: Go Sedan and Premier Sedan contribute most to total revenue.

5. Customer vs Driver Ratings
plt.hist(df['Customer Rating'], bins=5, alpha=0.5, label='Customer')
plt.hist(df['Driver Ratings'], bins=5, alpha=0.5, label='Driver')


⭐ Insight: Ratings are generally high (4.0–5.0) with minor variation between customers and drivers.

6. Monthly Revenue Trend
df.groupby('YearMonth')['Booking Value'].sum().plot(kind='line', marker='o')


📅 Insight: Ride revenue fluctuates with noticeable peaks around August–November 2024.

🤖 Machine Learning Model
Model: RandomForestClassifier

A Random Forest model was built to predict ride booking status based on features like vehicle type, ride distance, payment method, etc.

⚙️ Workflow
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('model', RandomForestClassifier(n_estimators=100, random_state=42))
])


Train/Test Split: 80/20

Stratified Sampling used to balance classes

Accuracy Achieved: 96.46%
