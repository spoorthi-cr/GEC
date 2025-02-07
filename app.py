import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def main():
    st.title("Iris Species Prediction App")

    # 1. Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    # 2. Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Train a Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model (test set accuracy)
    accuracy = model.score(X_test, y_test)

    st.write(f"Model Test Accuracy: *{accuracy:.2f}*")

    # 4. Create sliders for user input
    st.sidebar.header("Input Feature Values")
    def user_input_features():
        sepal_length = st.sidebar.slider(
            "Sepal length (cm)",
            float(X[:, 0].min()),
            float(X[:, 0].max()),
            float(X[:, 0].mean())
        )
        sepal_width = st.sidebar.slider(
            "Sepal width (cm)",
            float(X[:, 1].min()),
            float(X[:, 1].max()),
            float(X[:, 1].mean())
        )
        petal_length = st.sidebar.slider(
            "Petal length (cm)",
            float(X[:, 2].min()),
            float(X[:, 2].max()),
            float(X[:, 2].mean())
        )
        petal_width = st.sidebar.slider(
            "Petal width (cm)",
            float(X[:, 3].min()),
            float(X[:, 3].max()),
            float(X[:, 3].mean())
        )
        data = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    # Display the user inputs
    st.subheader("User Input Features")
    st.write(input_df)

    # 5. Generate Predictions
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader("Prediction")
    st.write(f"Predicted class: *{target_names[prediction][0]}*")

    st.subheader("Prediction Probabilities")
    pred_proba_df = pd.DataFrame(prediction_proba, columns=target_names)
    st.write(pred_proba_df)

if _name_ == "_main_":
    main()
