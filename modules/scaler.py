from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def scale_features(df, method="standard"):

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns

    if method == "standard":

        scaler = StandardScaler()

    elif method == "minmax":

        scaler = MinMaxScaler()

    elif method == "robust":

        scaler = RobustScaler()

    else:

        print("Invalid scaling method")

        return df

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    df[numeric_cols] = df[numeric_cols].round(3)

    print("\nScaling done using:", method)

    return df