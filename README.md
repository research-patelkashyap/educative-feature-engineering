# Feature Engineering
---

* Label Encoding
* One-hot Encoding
* Count Encoding
* Mean Encoding
* Weight of Evidence Encoding
* Feature Interaction
* Date-time Features

---

## Label Encoding

**Definition**: Label encoding is a technique to convert categorical features into numerical values so that machine learning algorithms can process them.

### Steps:

1. **Create LabelEncoder Object**:
   - **Definition**: Instantiate an object from the `LabelEncoder` class in the `sklearn.preprocessing` module.
   - **Example**: `label_encoder = LabelEncoder()`

2. **Fit the LabelEncoder**:
   - **Definition**: Use the `fit` method on the raw categorical features to establish the mapping between categorical and numerical values.
   - **Example**: `label_encoder.fit(raw_features)`

3. **Transform the Categorical Features**:
   - **Definition**: Apply the `transform` method to convert categorical values to their corresponding numerical values.
   - **Example**: `encoded_features = label_encoder.transform(raw_features)`

### Using Pandas for Label Encoding:

1. **Convert Column to Category Type**:
   - **Definition**: Change the data type of a column to `category`, which is a special type in pandas that facilitates label encoding.
   - **Example**: `df['column_name'] = df['column_name'].astype('category')`

2. **Use Categorical Values for Encoding**:
   - **Definition**: Access the `codes` attribute of the categorical column to get the numerical representation.
   - **Example**: `df['encoded_column'] = df['column_name'].cat.codes`

### Key Points:

- **Consistency**: Ensure the encoding is performed simultaneously on both training and testing datasets to avoid inconsistencies.
- **Example Mapping**: For a dataset, "Sun" might be mapped to 3 and "Moon" to 2.

---

