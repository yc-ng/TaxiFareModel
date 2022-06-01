# preprocessing and modeling
from black import Line
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# model selection
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
# model storage
import os.path
import joblib
# ML iteration
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
# internal modules
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer
from TaxiFareModel.utils import compute_rmse

class Trainer():
    mlflow_uri = "https://mlflow.lewagon.ai/"
    experiment_name = "[SG] [Singapore] [yc-ng] taxifare_predict_v1"

    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self, model_name: str, model):
        '''returns a pipeline integrating the specified model'''
        dist_pipe = Pipeline([
            ('dist_trans', DistanceTransformer()),
            ('stdscaler', StandardScaler())
        ])
        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])
        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")
        pipe = Pipeline([
            ('preproc', preproc_pipe),
            (model_name, model)
        ])
        return pipe

    def run(self, model_name: str, model):
        """sets the pipeline with model, then fits it on X and y"""
        self.pipeline = self.set_pipeline(model_name, model)
        self.pipeline.fit(self.X, self.y)
        return self.pipeline

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return rmse

    def cross_val(self, model_name: str, model):
        """evaluates the pipeline by K-fold cross-validation and
        returns the mean RMSE of the cross-validations"""
        self.pipeline = self.set_pipeline(model_name, model)
        cv_scores = cross_val_score(self.pipeline, self.X, self.y, cv=5,
                                    scoring='neg_root_mean_squared_error',
                                    verbose=1, n_jobs=-3)
        # metric is negative, multiply by -1 to get positive
        return cv_scores.mean() * -1

    def save_model(self, model_name):
        """ Save the trained model into a model.joblib file """
        filepath = os.path.join(
            os.path.dirname(__file__), 'models', f'{model_name}.joblib'
        )
        joblib.dump(self.pipeline, filepath)
        return None

    #############################################
    # Upload runs with params/metrics to MLFlow #
    #############################################

    # memoized as self.mlflow_client
    @memoized_property
    def mlflow_client(self):
        """sets up the MLflow client"""
        mlflow.set_tracking_uri(self.mlflow_uri)
        return MlflowClient()

    # memoized as self.mlflow_experiment_id
    @memoized_property
    def mlflow_experiment_id(self):
        """generates experiment ID for the MLflow client"""
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    # memoized as self.mlflow_run
    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        """Logs the parameters for a """
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":

    from TaxiFareModel.data import get_data, clean_data
    from sklearn.model_selection import train_test_split
    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    X = df.drop(columns='fare_amount')
    y = df['fare_amount']
    # hold out
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)

    # iterate over different estimators
    models_list = [
        ('linear_regression', LinearRegression()),
        ('ridge_regression', Ridge(alpha=1)),
        ('lasso_regression', Lasso(alpha=1)),
        ('sv_regressor', SVR(C=1)),
        ('random_forest', RandomForestRegressor()),
        ('knn_regressor', KNeighborsRegressor()),
        ('gboost_regressor', GradientBoostingRegressor())
    ]

    upload = True # flag to determine uploading to mlflow
    for model_name, model in models_list:
        # train the model
        trainer = Trainer(X, y)
        trainer.run(model_name, model)
        # cross-validate the model and store the metric
        rmse = trainer.cross_val(model_name, model)
        print(f"RMSE for {model_name} is {rmse}")
        # fit save the model
        trainer.run(model_name, model)
        trainer.save_model(model_name=model_name)
        print(f"{model_name} model saved.")

        # if uploading, log the user, model and CV metric
        if upload:
            trainer.mlflow_log_param("student_name", "YC")
            trainer.mlflow_log_param("model", model_name)
            trainer.mlflow_log_metric("rmse", rmse)
            print(f"Metrics for {model_name} uploaded.")

    if upload:
        experiment_id = trainer.mlflow_experiment_id
        print(f"experiment URL: https://mlflow.lewagon.ai/#/experiments/{experiment_id}")
