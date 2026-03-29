import pandas as pd


class DataProfiler:
    def __init__(self, df):
        self.df = df

    def get_column_types(self):
        """
        Identify column types more intelligently
        """
        numeric_cols = []
        categorical_cols = []
        id_cols = []

        for col in self.df.columns:
            if self.df[col].nunique() == len(self.df):
                id_cols.append(col)
                continue

            if self.df[col].dtype == "object":
                categorical_cols.append(col)

            elif self.df[col].dtype in ["int64", "float64"]:
                # detect categorical disguised as numeric
                if self.df[col].nunique() < 15:
                    categorical_cols.append(col)
                else:
                    numeric_cols.append(col)

            else:
                categorical_cols.append(col)

        return numeric_cols, categorical_cols, id_cols

    def summary(self):
        numeric_cols, categorical_cols, id_cols = self.get_column_types()

        print("\n[DataProfiler]")
        print("Numeric Columns:", numeric_cols)
        print("Categorical Columns:", categorical_cols)
        print("ID Columns:", id_cols)

        return numeric_cols, categorical_cols, id_cols