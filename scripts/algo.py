from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import etl as etl


class algoritmos():
    def __init__(self):
        '''
        inicializar todos los atributos resultantes del algoritmo, y ejecutar el proceso de etl para preparar el modelo y calibrarlo
        '''

        #llamar la funcion de lectura de datos
        self.data = etl.leer_datos()
        #llama la funcion para preparar la data
        self.preparar_data(self.data)
        #llama la funcion de calibracion del modelo
        self.modelo()

    def preparar_data(self,data):
        '''
        se realiza la limpieza de la data previamente generada en el etl y se separa la data para
        usar hold out como metodo de validacion (se sapara en 80/20 por pareto ya que el volumen de datos es adecuada)
        y se almacenan como atributos de la clase
        :param data:
        :return:
        '''

        #separar la data de entrenamiento y pruebas.
        X_train_real, X_test_real = train_test_split(data, test_size=0.20)

        #retirar las columnas de indicador de fila e identificador universal.
        X_train_real = X_train_real.drop('Unnamed: 0', axis=1)
        X_train_real = X_train_real.drop('v_0', axis=1)
        X_test_real = X_test_real.drop('Unnamed: 0', axis=1)
        X_test_real = X_test_real.drop('v_0', axis=1)

        #normalizar los rangos de las variables para no afectar el desempeño del modelo.
        self.X_train = normalize(X_train_real[['v_1', 'v_4', 'v_5', 'v_8', 'v_9']])
        self.X_test = normalize(X_test_real[['v_1', 'v_4', 'v_5', 'v_8', 'v_9']])
        self.X_test_real = X_test_real
        self.X_train_real = X_train_real


    def modelo(self):
        '''
        se genera los objetos que mapean los resultados del modelo y guardan los csv donde estan los resultados de las dos faces
        tanto entenamiento y test
        :return:
        '''

        #construccion del modelo
        self.results = KMeans(n_clusters=2, random_state=0).fit(self.X_train)

        #predecir los valores de entrenamiento y de prueba
        self.training_train = self.results.predict(self.X_train)
        self.training_test = self.results.predict(self.X_test)

        #añadir los resultados a los datasets originales sin normalizar
        self.X_test_real['result'] = self.training_test
        self.X_train_real['result'] = self.training_train

        #escribir el resultado en un csv
        self.X_test_real.to_csv("../result/result_test.csv")
        self.X_train_real.to_csv("../result/result_train.csv")
