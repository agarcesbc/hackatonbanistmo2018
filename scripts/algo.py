from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import etl as etl

class algoritmos():
    def __init__(self):
        data = etl.leer_datos()
        self.preparar_data(data)
        self.modelo()
        #print(self.X_train_real.describe())

    def preparar_data(self,data):
        X_train_real, X_test_real = train_test_split(data, test_size=0.20)
        X_train_real = X_train_real.drop('Unnamed: 0', axis=1)
        X_train_real = X_train_real.drop('v_0', axis=1)
        X_test_real = X_test_real.drop('Unnamed: 0', axis=1)
        X_test_real = X_test_real.drop('v_0', axis=1)
        self.X_train = normalize(X_train_real)
        self.X_test = normalize(X_test_real)
        self.X_test_real = X_test_real
        self.X_train_real = X_train_real

    def modelo(self):
        self.results = KMeans(n_clusters=2, random_state=0).fit(self.X_train)
        self.training_train = self.results.predict(self.X_train)
        self.training_test = self.results.predict(self.X_test)
        self.X_test_real['result'] = self.training_test
        self.X_train_real['result'] = self.training_train
        self.X_test_real.to_csv("result_test.csv")
        self.X_train_real.to_csv("result_train.csv")
