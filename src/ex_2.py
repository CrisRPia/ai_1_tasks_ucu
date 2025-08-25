from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from data import train
import pandas as pd


def main():
    df = train.copy()
    print(f"{df.isna() = }")
    return
    print(f"{df = }")

    # 🚫 PASO 1: Manejar valores faltantes (imputación)
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])  # Valor más común
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())              # Mediana
    df['Age'] = df['Age'].fillna(df.groupby(['Sex','Pclass'])['Age'].transform('median'))

    # 🆕 PASO 2: Crear nuevas features útiles
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
    rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
    df['Title'] = df['Title'].replace(rare_titles, 'Rare')

    # 🔄 PASO 3: Preparar datos para el modelo
    features = ['Pclass','Sex','Age','Fare','Embarked','FamilySize','IsAlone','Title','SibSp','Parch']
    X = pd.get_dummies(df[features], drop_first=True)
    y = df['Survived']

    print(f"{X.shape, y.shape = }")
    

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dummy = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy.fit(X_train, y_train)
    baseline_pred = dummy.predict(X_test)

    lr = LogisticRegression(max_iter=1000, solver='liblinear', random_state=42)
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)

    print('Baseline acc:', accuracy_score(y_test, baseline_pred))
    print('LogReg acc  :', accuracy_score(y_test, pred))

    print('\nClassification report (LogReg):')
    print(classification_report(y_test, pred))

    print('\nConfusion matrix (LogReg):')
    print(confusion_matrix(y_test, pred))

if __name__ == "__main__":
    main()
