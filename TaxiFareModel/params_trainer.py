from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from TaxiFareModel.trainer import Trainer
from TaxiFareModel.data import get_data, clean_data


if __name__ == "__main__":

    # get data
    df = get_data()
    # clean data
    df = clean_data(df)
    # set X and y
    X = df.drop(columns='fare_amount')
    y = df['fare_amount']

    # declare the models to be used
    model_name = 'gboost_regressor'
    model = GradientBoostingRegressor()

    # initialize and train the model
    trainer = Trainer(X, y)

    # declare the parameter search space
    gboost_params = {
        f'{model_name}__learning_rate': [0.05, 0.1, 0.2],
        f'{model_name}__max_depth': [2, 3, 4],
        f'{model_name}__n_estimators': [25, 50, 75],
    }

    # grid search the parameter space and
    # obtain the best model, params, score
    best_model, best_params, best_score = \
        trainer.gridsearch_params(model_name, model, gboost_params)

    for param, value in best_params.items():
        param_short = param[len(model_name)+2:] # remove model_name prefix
        print(f"best {param_short} is {value}")
    print(f"best RMSE is {best_score}")
    print(best_model)

    # refit the pipeline with the best parameters and save it
    save = True
    if save:
        trainer.pipeline = best_model
        trainer.pipeline.fit(X, y)
        trainer.save_model(model_name=f"{model_name}_tuned")
        print(f"{model_name} model saved.")

    # upload to MLflow
    upload = True
    if upload:
        trainer.mlflow_log_param("student_name", "YC")
        trainer.mlflow_log_param("model", model_name)
        for param, value in best_params.items():
            param_short = param[len(model_name)+2:] # remove model_name prefix
            trainer.mlflow_log_param(param_short, value)
        trainer.mlflow_log_metric("rmse", best_score)
        print(f"Params and metrics for {model_name} uploaded.")
