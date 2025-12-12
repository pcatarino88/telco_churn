import pandas as pd
from utils.data_preparation import cleaning, feature_engineering, encoding, scaling
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline


IMPORTANT_FEATURES = [
    'Contract Duration',
    'Dependents',
    'Internet Type',
    'Monthly Charge',
    'Number of Referrals_bins',
    'Paperless Billing',
    'Payment Method_Credit Card',
    'Senior Citizen',
    'Tenure in Months',
    'Total Extra Data Charges per Month',
    'Total Long Distance Charges per Month',
    'Unlimited Data',
]


class PreprocessTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = RobustScaler()
        self.encoded_columns_ = None

    def fit(self, X, y=None):
        Xp = cleaning(X.copy())
        Xp = feature_engineering(Xp)
        Xp = encoding(Xp)

        self.encoded_columns_ = Xp.columns.tolist()
        self.scaler.fit(Xp)

        return self

    def transform(self, X):
        Xp = cleaning(X.copy())
        Xp = feature_engineering(Xp)
        Xp = encoding(Xp)

        Xp = Xp.reindex(columns=self.encoded_columns_, fill_value=0)

        Xs = pd.DataFrame(
            self.scaler.transform(Xp),
            columns=self.encoded_columns_,
            index=Xp.index
        )

        return Xs[IMPORTANT_FEATURES]