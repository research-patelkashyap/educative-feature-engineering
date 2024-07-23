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

- **Count Encoding**:

  - Converts each categorical value to the count of its occurrences in the dataset.
  - Simple and effective, particularly for tree-based models like XGBoost.

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
