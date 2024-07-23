# Feature Engineering

---

- Label Encoding
- One-hot Encoding
- Count Encoding
- Mean Encoding
- Weight of Evidence Encoding
- Feature Interaction
- Date-time Features

---

## Label Encoding

**Definition**: Label encoding is a technique to convert categorical features into numerical values so that machine learning algorithms can process them.

### Key Points:

- **Consistency**: Ensure the encoding is performed simultaneously on both training and testing datasets to avoid inconsistencies.
- **Example Mapping**: For a dataset, "Sun" might be mapped to 3 and "Moon" to 2.

### Steps:

1. **Label Encoding**:

   - Convert categorical features to numerical values.
   - Use `LabelEncoder` from `sklearn.preprocessing`.

2. **Steps in sklearn**:

   - Create a `LabelEncoder` object.
   - Fit on raw features to establish mapping.
   - Transform the features to numerical values.

3. **Steps in Pandas**:

   - Convert column to `category` type.
   - Use `cat.codes` for label encoding.

4. **Important Considerations**:
   - Perform encoding consistently across both train and test datasets.
   - Verify the encoded values to ensure correct mappings.

---

## One-hot Encoding

**Definition**: One-hot encoding is a technique to convert categorical features into numerical features by creating new columns for each unique categorical value, assigning a value of 1 or 0 to indicate the presence or absence of that category.

### Key Points:

- **One-Hot Encoding**:

  - Creates new binary columns for each unique categorical value.
  - Each column represents the presence (1) or absence (0) of the categorical value.

- **Example**: For a column "Planet" with values "Earth" and "Mars":

  - "Earth" -> [1, 0]
  - "Mars" -> [0, 1]

- **Advantages Over Label Encoding**:
  - Avoids misleading ordinal relationships.
  - Preserves the true categorical nature of the data.
  - Avoids misleading patterns by treating each category independently.

### Steps:

1. **Steps in sklearn**:

   - Create `OneHotEncoder` object.
   - Fit `OneHotEncoder` to integer-coded features.
   - Transform features to one-hot encoded format.

2. **Steps in Pandas**:

   - Use `get_dummies` function to one-hot encode a DataFrame.
   - Specify the column(s) and prefix for new column names.

3. **Important Considerations**:
   - Ensure consistent application across train and test datasets.
   - Verify the encoded output to ensure correctness.

---

## Count Encoding

**Definition**: Count encoding transforms categorical features into numerical features based on the frequency of each category's occurrence in the dataset.

### Key Points:

- **Advantages**:

  - Simple and effective for many models, especially tree-based models.

- **Limitations**:

  - Not suitable for new categories in the test set not present in the training set.
  - Can create conflicts if different categories have the same count.

- **Common Use**:

  - Popular in data science competitions and practical applications for tree-based models.

### Steps:

1. **Steps in Pandas**:

   - Use `value_counts` to generate a dictionary of category counts.
   - Map these counts to create a new encoded column.

2. **Implementation Example**:

   ```python
   value_counts_dict = df["col1"].value_counts().to_dict()
   df["encoded_col"] = df["col1"].map(value_counts_dict)
   ```

---

## Mean Encoding

**Definition**: Mean encoding transforms categorical features into numerical features by replacing each category with the mean of the target value (or any other numerical feature) for that category.

### Key Points:

- **Advantages**:

  - Provides more informative encoding by leveraging the target value.

- **Variations**:

  - Instead of mean, other statistical measures like standard deviation (std), variance (var), or maximum (max) can also be used.

- **Powerful Method**:

  - Captures the mean of a category value in the entire dataset, providing more informative encoding.

- **Example**:
  - For a "Type" column and a "Price" target: "Sun" might be replaced by the mean price for all "Sun" entries.

### Steps:

1. **Steps in Pandas**:

   - Group by the categorical column and calculate the mean of the target value.
   - Map these mean values to create a new encoded column.

2. **Implementation Example**:
   ```python
   mean_dict = df.groupby("col1")["target"].mean().to_dict()
   df["encoded_col"] = df["col1"].map(mean_dict)
   ```

---

## Weight of Evidence (WOE) Encoding

**Definition**: Weight of Evidence (WOE) encoding is a technique used to encode categorical features in binary classification tasks. It measures the ratio of the probabilities of positive and negative outcomes for each category, providing an indication of how each category affects the likelihood of the target variable.

### Formula:

\[ \text{WOE} = \ln \left( \frac{p(1)}{p(0)} \right) \]

- \( p(1) \) is the probability of the positive label (1) for a given category.
- \( p(0) \) is the probability of the negative label (0) for the same category.

### Key Points:

- **Advantages**:

  - Helps in understanding the impact of each category on the outcome.
  - Can improve model performance by providing a meaningful encoding.

- **Limitations**:
  - Requires careful handling of categories with zero occurrences to avoid division by zero.
  - Assumes that the relationship between categorical features and the target is linear.

### Steps:

1. **Steps in Pandas**:

   - Calculate the probability of positive and negative labels for each category.
   - Compute WOE using the formula \( \text{WOE} = \ln \left( \frac{p(1)}{p(0)} \right) \).
   - Map these WOE values to the original categorical features.

2. **Implementation Example**:

   ```python
   # Calculate probabilities
   p1 = df[df["target"] == 1].groupby("col1").size() / df[df["target"] == 1].shape[0]
   p0 = df[df["target"] == 0].groupby("col1").size() / df[df["target"] == 0].shape[0]

   # Calculate WOE
   woe = np.log(p1 / p0)

   # Map WOE values to original column
   df["encoded_col"] = df["col1"].map(woe)
   ```

---

## Feature Interaction

**Definition**: Feature interaction involves creating new features by combining two or more existing features. This technique helps to capture complex relationships and interactions between features, enhancing the model's predictive power.

### Advantages of Feature Interaction:

- **Expanded Feature Space**:

  - Increases the number of feature combinations, enhancing the model's ability to capture complex patterns.

  - **Example**: Combining a feature with 20 values and another with 30 values results in 600 possible interactions.

- **Finer Sample Distribution**:

  - Allows for more detailed differentiation between samples, improving model accuracy.

- **Introduces Nonlinearity**:

  - Enhances the model's capacity to capture nonlinear relationships between features.

### Practical Tips:

- **Second-Order Interactions**:

  - Commonly used and often sufficient for many tasks.

- **Higher-Order Interactions**:

  - Can be useful but may add excessive complexity. Use judiciously based on task and data.

- **Domain Knowledge**:

  - Apply interactions based on a deep understanding of the domain and task to avoid unnecessary complexity.

### Feature Interaction in Pandas:

1. **Concatenate Features**:

   - **Example**: `df["fea1_fea2"] = df["fea1"].astype('str') + "_" + df["fea2"].astype('str')`

2. **Multiple Columns**:

   - **Example**: `df["fea1_fea2_fea3"] = df["fea1"].astype('str') + "_" + df["fea2"].astype('str') + "_" + df["fea3"].astype('str')`

### Generalized Steps:

- Identify features to interact.
- Combine features using concatenation or arithmetic operations.
- Create and add the new feature to the dataset.
- Consider higher-order interactions with caution.

---

## Handling DateTime Features

**Definition**: DateTime features capture temporal information from data. Extracting and manipulating these features can reveal patterns related to time, such as seasonal behaviors or time intervals between events.

### Key Points:

- **Extracting Components**:

  - **Month, Week, Hour**: Useful for identifying patterns based on time of year, week, or day.
  - **Example**: Different user behaviors in summer vs. winter, or peak activity hours.

- **Time Intervals**:

  - **Definition**: Time intervals capture the duration between events, which can be critical for behavior analysis.
  - **Example**: Time between user logins to predict churn rates.

- **Advantages**:

  - Reveal seasonal and time-of-day patterns.
  - Provide critical information for behavior prediction and analysis.

### Practical Tips:

- **Unit of Time**:

  - Ensure correct unit specification (e.g., seconds for UNIX timestamps).
  - **Example**: `pd.to_datetime(df["timestamp_col"], unit='s')`

- **Handling Multiple Time Components**:

  - Extract multiple components as needed to gain more insights.
  - **Example**: Month, week, hour from the same datetime object.

### Steps:

1. **Steps in Pandas**:

   - Convert UNIX timestamp to datetime object.
   - Extract components like month, week, and hour.
   - Calculate time intervals between events.

2. **Implementation Example**:

   ```python
   # Convert UNIX timestamp to datetime
   df["datetime_col"] = pd.to_datetime(df["timestamp_col"], unit='s')

   # Extract time components
   df["month"] = df["datetime_col"].dt.month
   df["week"] = df["datetime_col"].dt.isocalendar().week
   df["hour"] = df["datetime_col"].dt.hour

   # Calculate time intervals
   df["time_interval"] = (df["end_datetime"] - df["start_datetime"]).dt.days
   ```
