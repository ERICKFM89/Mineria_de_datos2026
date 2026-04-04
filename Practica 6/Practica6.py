import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def codificar_categorias(data: pd.DataFrame) -> pd.DataFrame: 
    le = LabelEncoder()
    data['Platform'] = le.fit_transform(data['Platform'])
    data['Genre'] = le.fit_transform(data['Genre'])
    data['Publisher'] = le.fit_transform(data['Publisher'])
    return data

def crear_variable_objetivo(data: pd.DataFrame) -> pd.DataFrame: 
    median_sales = data['Global_Sales'].median()
    data['Sales_Class'] = data['Global_Sales'].apply(lambda x: 1 if x > median_sales else 0)
    return data

def preparar_datos(data: pd.DataFrame) -> pd.DataFrame: 
    features = ['Platform', 'Year', 'Genre', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    X = data[features]
    y = data['Sales_Class']
    return train_test_split(X, y, test_size=0.3, random_state=42)

def escalar_datos(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def entrenar_y_evaluar_knn(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f'Accuracy del modelo KNN: {accuracy:.2f}')
    print('Reporte de clasificación:')
    print(report)
    print('Matriz de confusión:')
    print(conf_matrix)

def main():
    # Lectura de datos
    data = pd.read_csv("Practica 1/videogamesales/videogamesales_clean.csv")
    data = codificar_categorias(data)
    data = crear_variable_objetivo(data)
    
    X_train, X_test, y_train, y_test = preparar_datos(data)
    
    X_train_scaled, X_test_scaled = escalar_datos(X_train, X_test)
    
    entrenar_y_evaluar_knn(X_train_scaled, X_test_scaled, y_train, y_test)

if __name__ == "__main__":
    main()