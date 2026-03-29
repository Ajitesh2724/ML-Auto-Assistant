from sklearn.preprocessing import StandardScaler

def scale_features(df):

    scaler = StandardScaler()

    numeric_columns = df.select_dtypes(include=['int64','float64']).columns

    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    df[numeric_columns] = df[numeric_columns].round(2)

    print("\nFeature scaling completed!")

    return df