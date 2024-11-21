import hashlib

# TODO: Write a function which creates a sample split based in some id_column and training_frac.
# Optional: If the dtype of id_column is a string, we can use hashlib to get an integer representation.
def create_sample_split(df, id_column, training_frac=0.8):
    """Create sample split based on ID column.

    Parameters
    ----------
    df : pd.DataFrame
        Training data
    id_column : str
        Name of ID column4
    training_frac : float, optional
        Fraction to use for training, by default 0.9

    Returns
    -------
    pd.DataFrame
        Training data with sample column containing train/test split based on IDs.
    """
    def hash_to_split(id_value):
         # Create a hash from the ID
        hash_value = int(hashlib.md5(str(id_value).encode()).hexdigest(), 16)
        # Map hash to train/test based on training_frac
        return 'train' if hash_value % 100 < training_frac * 100 else 'test'
    
    # Apply the hash function to create the 'sample' column
    df['sample'] = df[id_column].apply(hash_to_split)

    return df
