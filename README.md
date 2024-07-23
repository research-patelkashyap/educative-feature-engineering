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
