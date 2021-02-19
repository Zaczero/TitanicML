import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline, make_union
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from tpot.builtins import StackingEstimator
from xgboost import XGBClassifier
from tpot import TPOTClassifier
from tpot.export_utils import set_param_recursive


def prep_drop(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop(['PassengerId', 'Name', 'Ticket'], axis=1)


def prep_fillna(data: pd.DataFrame) -> pd.DataFrame:
    config = {
        'Age': np.nanmedian,
        'Embarked': 'S',
    }

    for colName, colReplace in config.items():
        if callable(colReplace):
            # noinspection PyPep8Naming
            colReplace = colReplace(data[colName])

        data[colName] = data[colName].fillna(colReplace)

    return data


def prep_round(data: pd.DataFrame) -> pd.DataFrame:
    config = ['Age']

    for colName in config:
        data[colName] = data[colName].round()

    return data


def prep_count(data: pd.DataFrame) -> pd.DataFrame:
    config = ['Cabin']

    for colName in config:
        data[colName] = data[colName].apply(lambda text: len(re.findall(r'\w+', text if text is not np.nan else '')))

    return data


def prep_scale(data: pd.DataFrame) -> pd.DataFrame:
    config = {
        'Age': 100,
        'Cabin': 4,
    }

    for colName, colScale in config.items():
        data[colName] = data[colName] / colScale

    return data


def prep_bin(data: pd.DataFrame) -> pd.DataFrame:
    config = {
        'SibSp': {
            'bins': [-1, 0, 1, np.inf],
            'names': ['none', 'single', 'many']
        },
        'Parch': {
            'bins': [-1, 0, 1, np.inf],
            'names': ['none', 'single', 'many']
        },
        'Fare': {
            'percs': [0, .25, .75, 1],
            'names': ['low', 'med', 'hi']
        }
    }

    for colName, colConfig in config.items():
        if 'bins' in colConfig:
            bins = colConfig['bins']
        else:
            bins = data[colName].quantile(colConfig['percs'])

        data[colName] = pd.cut(data[colName], bins, labels=colConfig['names'])

    return data


def prep_categorize(data: pd.DataFrame) -> pd.DataFrame:
    config = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']

    for colName in config:
        data[colName] = data[colName].astype('category')

    return data


def prep_one_hot(data: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(data, drop_first=True)


# Average CV score on the training set was: 0.8507375557089951
pipe_tpot = make_pipeline(
    StackingEstimator(estimator=SGDClassifier(alpha=0.0, eta0=0.1, fit_intercept=False, l1_ratio=1.0, learning_rate="constant", loss="squared_hinge", penalty="elasticnet", power_t=50.0)),
    XGBClassifier(learning_rate=0.1, max_depth=10, min_child_weight=4, n_estimators=100, n_jobs=1, subsample=0.9500000000000001, verbosity=0)
)


# There is a reason I have not merged those both together
pipe = Pipeline([
    ('prep_drop', FunctionTransformer(prep_drop)),
    ('prep_fillna', FunctionTransformer(prep_fillna)),
    ('prep_round', FunctionTransformer(prep_round)),
    ('prep_count', FunctionTransformer(prep_count)),
    # ('prep_scale', FunctionTransformer(prep_scale)),
    ('prep_bin', FunctionTransformer(prep_bin)),
    ('prep_categorize', FunctionTransformer(prep_categorize)),
    ('prep_one_hot', FunctionTransformer(prep_one_hot))
])


# print all pandas columns
pd.set_option('display.max_columns', None)

df = pd.read_csv('data/train.csv', header=0)
df = pipe.transform(df)

y = df['Survived']
X = df.drop('Survived', axis=1)

pipe_tpot.fit(X, y)

df_test = pd.read_csv('data/test.csv', header=0)

preds = pipe_tpot.predict(
    pipe.transform(df_test)
)

submission = pd.DataFrame({
    'PassengerId': df_test['PassengerId'],
    'Survived': preds,
})

submission.to_csv('data/submission.csv', index=False)

# tpot = TPOTClassifier(generations=100,
#                       population_size=30,
#                       cv=5,
#                       n_jobs=15,
#                       random_state=111,
#                       periodic_checkpoint_folder='checkpoints',
#                       early_stop=20,
#                       verbosity=2)
#
# tpot.fit(X, y)
