# Data Analytics & Data Science: The "Edge-Case" Interview Guide
**15 Highly Unconventional, Deep-Dive Questions for Senior Analysts**

*Forget "What is a Left Join?" and "How do you calculate the Mean?". This guide tests your understanding of statistical paradoxes, SQL silent killers, and Pandas memory traps.*

---

## 1. Statistical Paradoxes & A/B Testing

### Q1: Your A/B test shows that Variant B has a higher conversion rate for Mobile users AND a higher conversion rate for Desktop users. However, when you look at the total combined users, Variant A has a higher conversion rate. How is this mathematically possible?
**Answer:**
This is a classic example of **Simpson's Paradox**. It occurs when the groups (Mobile vs. Desktop) have drastically different sample sizes and different baseline conversion rates. 
For example, Variant A might have been exposed mostly to Desktop users (who naturally have a 20% conversion rate), while Variant B was exposed mostly to Mobile users (who naturally have a 2% conversion rate). Even if Variant B slightly improved the Mobile rate to 3%, the massive volume of low-converting Mobile users pulls the overall average of Variant B down below Variant A.
*Takeaway:* Always ensure equal traffic splits across sub-segments in randomized controlled trials.

### Q2: You are running an A/B test. On Day 3, you check the dashboard and the p-value is 0.02 (Statistically Significant). You immediately stop the test and declare a winner. Why is this a massive analytical error?
**Answer:**
This is known as the **"Peeking Problem"** or Continuous Monitoring Trap. 
If you calculate the p-value continuously and stop the experiment the first time it dips below 0.05, you vastly inflate your False Positive Rate (Type I Error). Because A/B test metrics fluctuate wildly in the early days due to small sample sizes, almost *every* A/A test (where both variants are identical) will temporarily show statistical significance at some point. You must pre-determine the required sample size and *only* read the p-value once that sample size is reached.

### Q3: You are testing 20 different button colors at the same time against a control. One of them (Neon Pink) shows a statistically significant lift with p = 0.04. Should you deploy it?
**Answer:**
Probably not. This is the **Multiple Comparisons Problem**. If you test 20 different random colors with an $\alpha$ of 0.05, the probability of getting at least one False Positive purely by random chance is $1 - (1 - 0.05)^{20} \approx 64\%$. 
To fix this, you must apply the **Bonferroni Correction** (divide your target $\alpha$ by the number of tests, so you need $p < 0.0025$) or control the False Discovery Rate using the Benjamini-Hochberg procedure.

---

## 2. SQL "Silent Killers"

### Q4: You write the following query to find all users who do not live in London:
```sql
SELECT * FROM users WHERE city != 'London'
```
**Why will this query accidentally exclude users who have not set a city?**
**Answer:**
In SQL, `NULL` represents an unknown value. Any comparison against `NULL` (including `!=` or `=`) yields `UNKNOWN` (essentially `FALSE` for `WHERE` clauses). 
If a user's city is `NULL`, `NULL != 'London'` evaluates to `UNKNOWN`, and the row is excluded. 
*Fix:* You must write `WHERE city != 'London' OR city IS NULL`.

### Q5: Explain how a `CROSS JOIN` combined with an aggregation can cause a "Fan-Out" explosion that crashes the data warehouse.
**Answer:**
If you have a `users` table with 1,000,000 rows and a `transactions` table with 5,000,000 rows, and you accidentally forget the `ON` condition in your `JOIN` (making it a `CROSS JOIN` or Cartesian Product), the database will attempt to create a temporary table with $1,000,000 \times 5,000,000 = 5,000,000,000,000$ (5 Trillion) rows. If you attempt to run a `SUM()` over this, it will exhaust the cluster's memory and crash the query.

### Q6: What is the difference between `COUNT(*)` and `COUNT(column_name)`?
**Answer:**
`COUNT(*)` counts the absolute number of rows in the table or group, regardless of what the rows contain.
`COUNT(column_name)` counts the number of rows where `column_name` is **NOT NULL**. 
If you want to find the total number of users, `COUNT(*)` is correct. If you use `COUNT(phone_number)` on users where half haven't provided a phone, your user count will be wrong by 50%.

---

## 3. Python & Pandas Traps

### Q7: Why does Pandas throw a `SettingWithCopyWarning` when you do `df[df['age'] > 30]['salary'] = 50000`, and how do you fix it?
**Answer:**
Pandas is warning you that it doesn't know if `df[df['age'] > 30]` returned a *View* (a reference to the original memory) or a *Copy* (a new slice in memory). 
By chaining brackets `[][]`, you are modifying a temporary DataFrame that gets instantly garbage collected, meaning the original `df` is completely unmodified!
*Fix:* Always use `.loc` for assignment:
```python
df.loc[df['age'] > 30, 'salary'] = 50000
```

### Q8: You have a DataFrame with 100 million rows. One column is `country` (containing only 10 unique strings like "USA", "UK"). Loading this DataFrame takes 5GB of RAM. How do you drop the RAM usage by 90% without losing any data?
**Answer:**
Convert the column `dtype` from `object` (string) to `category`. 
Strings in Python are heavy objects. When stored as `object`, Pandas stores 100 million string pointers. By converting to `category`, Pandas stores an underlying integer array (0 to 9) and a single lookup dictionary of 10 strings.
```python
df['country'] = df['country'].astype('category')
```

### Q9: What is Anscombe's Quartet, and why does it mandate Data Visualization?
**Answer:**
Anscombe's Quartet is a set of four datasets that have nearly identical descriptive statistics: the exact same Mean, Variance, Correlation Coefficient, and Linear Regression Line.
However, when graphed, they look entirely different: one is a normal scatter plot, one is a non-linear curve, one is a straight line with a massive outlier, and one is a vertical line with an outlier. 
It proves that relying solely on aggregate metrics (like Pearson Correlation) without visualizing the scatter plot can completely mislead a data analyst.
