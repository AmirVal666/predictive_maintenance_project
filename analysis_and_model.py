import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler

def analysis_and_model_page():
    st.title("Анализ данных и модель")

    uploaded_file = st.file_uploader("Загрузите CSV", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        data = data.drop(columns=['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], errors='ignore')
        data['Type'] = LabelEncoder().fit_transform(data['Type'])
        numerical = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        scaler = StandardScaler()
        data[numerical] = scaler.fit_transform(data[numerical])

        X = data.drop(columns=['Machine failure'])
        y = data['Machine failure']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        st.subheader("Метрики модели")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        st.text("Classification Report:\n" + classification_report(y_test, y_pred))

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        st.subheader("ROC-кривая")
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.2f}")
        ax.plot([0, 1], [0, 1], '--')
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Предсказание по новым данным")
        with st.form("predict_form"):
            type_input = st.selectbox("Type", ["L", "M", "H"])
            air_temp = st.number_input("Air Temperature (K)", 290.0, 320.0)
            process_temp = st.number_input("Process Temp (K)", 300.0, 330.0)
            rotation = st.number_input("Rotational speed (rpm)", 1200, 2000)
            torque = st.number_input("Torque (Nm)", 10.0, 70.0)
            wear = st.number_input("Tool wear (min)", 0, 250)
            submit = st.form_submit_button("Предсказать")

            if submit:
                type_encoded = {"L": 0, "M": 1, "H": 2}[type_input]
                new_data = pd.DataFrame([[type_encoded, air_temp, process_temp, rotation, torque, wear]],
                                        columns=['Type'] + numerical)
                new_data[numerical] = scaler.transform(new_data[numerical])
                pred = model.predict(new_data)
                proba = model.predict_proba(new_data)[:, 1]
                st.write(f"Предсказание: {'Отказ' if pred[0] else 'Нет отказа'}")
                st.write(f"Вероятность отказа: {proba[0]:.2f}")
