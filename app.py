# Base libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io, os, glob, time

# Pandas profiling
# import pandas_profiling
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

# Data preparation
from sklearn.model_selection import train_test_split

# Model building
from flaml import AutoML

# Performance metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlxtend.plotting import plot_confusion_matrix

# Web Data App
import streamlit as st
import streamlit.components.v1 as components

def main():
    st.title("Data Workbench v0.1")

    def load_data(data_file, data_name):
        if 'current' not in st.session_state or st.session_state['current'] != data_name:
            dataset = pd.read_csv(data_file, skipinitialspace=True)
            # New dataset loaded. Relinquish all the previous states
            for key in st.session_state.keys():
                del st.session_state[key]
        else:
            dataset = st.session_state['dataset']
        st.session_state['dataset'] = dataset
        st.session_state['current'] = data_name
        return dataset

    def preprocess(dataset, drop_cols, target):
        dataset = dataset.drop(drop_cols, axis=1)
        X, y = dataset.drop([target], axis=1), dataset[[target]]
        features = dataset.columns
        return X.reset_index(drop=True), y, features
    
    def check_status(checker):
        if checker not in st.session_state:
            return None
        return st.session_state[checker]

    def profile_report(dataset, num_rows):
        if num_rows != len(dataset):
            dataset = dataset[:num_rows]

        reported = st.session_state['current'] + '_reported'
        report = check_status(reported)
        if not report:
            # report = dataset.profile_report()
            report = ProfileReport(dataset)
            st.session_state[reported] = report
        st_profile_report(report)

    def check_task(inspect):
        non_numbers = inspect.select_dtypes(exclude='number').columns
        if len(non_numbers) > 0:
            return "classification"
        select_ints = inspect.select_dtypes(include='int')
        if len(select_ints.columns) == 0:
            return "regression"
        value_count = len(inspect[select_ints.columns[0]].value_counts())
        if value_count > 30:        # Too many integral labels. Consider it as regression
            return "regression"
        return "classification"

    ### 
    # st.subheader("Load/Upload dataset file")
    for _ in range(2):
        st.sidebar.write("")
    st.sidebar.header("Dataset Loader")
    from_upload = st.sidebar.file_uploader('Upload a local CSV/XLSX file')
    from_data = st.sidebar.selectbox(
        "Choose from preloaded datasets", [""] + glob.glob('data/*.{}'.format("csv")))
    from_url = st.sidebar.text_input("Or, Get from the specified URL")
    data_file = from_upload if from_upload else from_data if from_data else from_url
    if not data_file:
        return
    data_name = data_file if type(data_file)==str else data_file.name
    dataset = load_data(data_file, data_name)

    st.subheader(f'Dataset: {data_name}')
    st.dataframe(dataset.head())
    st.write(f"Dataset shape: {dataset.shape}")

    max_cells = 10000
    num_rows, num_cols = dataset.shape
    num_rows = max_cells // num_cols
    profile_report(dataset, num_rows)

    ###
    st.header('A Taste of Machine Learning')
    target = st.selectbox(
        'Select target feature', sorted(list(dataset.columns), key=str.casefold))
    drop_cols = st.multiselect(
        'Select columns to drop',
        sorted(list(dataset.columns), key=str.casefold)
    )
    check_preprocess = st.session_state['current'] + '_processed'
    if st.button("Do the Magic"):
        if target:
            X, y, features = preprocess(dataset, drop_cols, target)
            st.session_state['X'] = X
            st.session_state['y'] = y
            st.session_state['features'] = features
            st.session_state['target'] = target
            st.session_state[check_preprocess] = True
        else:
            st.write('You should select at least one target!')

    if check_status(check_preprocess):
        st.write('Selected features: ' +  ', '.join(st.session_state['features']))
        removed_features = sorted(set(st.session_state['dataset'].columns) - \
                                  set(st.session_state['features']))
        st.write('Dropped features: ' +  ', '.join(removed_features))

        st.write('Transformed dataset')
        xformed_dataset = pd.concat([st.session_state['X'], st.session_state['y']], axis=1)
        st.dataframe(xformed_dataset.head())

        st.write(f"Dataset shape: {xformed_dataset.shape}")

        test_size = 0.25

        X = st.session_state['X']
        y = st.session_state['y']
        target = st.session_state['target']
        task = check_task(y)
        # st.markdown(f'Problem type is **{task}**.')
        st.subheader(f'Problem type is {task}.')
        X_train, X_test, y_train, y_test = train_test_split(X, y.values, test_size=test_size)
        model = AutoML(task=task, time_budget=20, max_iter=200)

        with st.spinner("Training a model..."):
            start_time = time.time()
            model.fit(X_train, y_train, estimator_list=['xgboost'], retrain_full=False)
            time_taken = int(time.time() - start_time)
            st.write(f"Train took {time_taken} seconds.")

        # Output
        st.subheader('Predictions')
        y_pred = model.predict(X_test).flatten()
        actuals_vs_preds = pd.DataFrame({'actual_' + target: y_test.flatten(),
                                         'predicted_' + target: y_pred.flatten()})
        st.dataframe(pd.concat([X_test.reset_index(drop=True), actuals_vs_preds], axis=1))

        # Performance
        st.subheader('Model Performance')
        if task == 'classification':
            st.write(f"Classification report")
            fig = plt.figure(figsize=(12, 8))
            clf_report = classification_report(y_test, y_pred, output_dict=True)
            sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
            st.pyplot(fig)

            st.write(f"Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig = plt.figure(figsize=(12, 8))
            # st.write(cm)
            labels = sorted(np.unique(y_train))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
            plt.ylabel("Actual")
            plt.xlabel("Predicted")
            st.pyplot(fig)
        elif task == 'regression':
            # Scatter plot
            fig = plt.figure(figsize=(12, 12))
            plt.scatter(y_pred, y_test, color="#33AADE")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.subplots_adjust(bottom=0.2)
            st.pyplot(fig)

            # Key metrics
            mae_res = mean_absolute_error(y_test, y_pred)
            rmse_res = mean_squared_error(y_test, y_pred) ** 0.5
            r_2 = r2_score(y_test, y_pred)
            st.markdown(f'**MAE**: {str(round(mae_res, 3))}')
            st.markdown(f'**RMSE**: {str(round(rmse_res, 3))}')
            st.markdown(f'**R^2**: {str(round(r_2, 3))}')

        st.subheader('Feature Importances')
        try:
            fig = plt.figure(figsize=(12, 8))
            plt.barh(model.model.estimator.feature_names_in_,
                     model.model.estimator.feature_importances_)
            st.pyplot(fig)
        except:
            print("Feature importances extraction failed")

        st.session_state[check_preprocess] = False


if __name__ == '__main__':
    main()
