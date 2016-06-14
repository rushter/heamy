=====
Usage
=====

To use heamy in a project:

.. code:: python

    from heamy.dataset import Dataset
    from heamy.estimator import Regressor, Classifier
    from heamy.pipeline import ModelsPipeline


Stacking
--------

.. code:: python

    # load boston dataset from sklearn
    from sklearn.datasets import load_boston
    data = load_boston()
    X, y = data['data'], data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=111)

    # create dataset
    dataset = Dataset(X_train,y_train,X_test)

    # initialize RandomForest & LinearRegression
    model_rf = Regressor(dataset=dataset, estimator=RandomForestRegressor, parameters={'n_estimators': 50},name='rf')
    model_lr = Regressor(dataset=dataset, estimator=LinearRegression, parameters={'normalize': True},name='lr')

    # Stack two models
    # Returns new dataset with out-of-fold predictions
    pipeline = ModelsPipeline(model_rf,model_lr)
    stack_ds = pipeline.stack(k=10,seed=111)

    # Train LinearRegression on stacked data (second stage)
    stacker = Regressor(dataset=stack_ds, estimator=LinearRegression)
    results = stacker.predict()
    # Validate results using 10 fold cross-validation
    results = stacker.validate(k=10,scorer=mean_absolute_error)

Blending
--------

.. code:: python

    # load boston dataset from sklearn
    from sklearn.datasets import load_boston
    data = load_boston()
    X, y = data['data'], data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=111)

    # create dataset
    dataset = Dataset(X_train,y_train,X_test)

    # initialize RandomForest & LinearRegression
    model_rf = Regressor(dataset=dataset, estimator=RandomForestRegressor, parameters={'n_estimators': 50},name='rf')
    model_lr = Regressor(dataset=dataset, estimator=LinearRegression, parameters={'normalize': True},name='lr')

    # Stack two models
    # Returns new dataset with out-of-fold predictions
    pipeline = ModelsPipeline(model_rf,model_lr)
    stack_ds = pipeline.blend(proportion=0.2,seed=111)

    # Train LinearRegression on stacked data (second stage)
    stacker = Regressor(dataset=stack_ds, estimator=LinearRegression)
    results = stacker.predict()
    # Validate results using 10 fold cross-validation
    results = stacker.validate(k=10,scorer=mean_absolute_error)


Weighted average
----------------

.. code:: python

    dataset = Dataset(preprocessor=boston_dataset)

    model_rf = Regressor(dataset=dataset, estimator=RandomForestRegressor, parameters={'n_estimators': 151},name='rf')
    model_lr = Regressor(dataset=dataset, estimator=LinearRegression, parameters={'normalize': True},name='lr')
    model_knn = Regressor(dataset=dataset, estimator=KNeighborsRegressor, parameters={'n_neighbors': 15},name='knn')

    pipeline = ModelsPipeline(model_rf,model_lr,model_knn)

    weights = pipeline.find_weights(mean_absolute_error)
    result = pipeline.weight(weights)