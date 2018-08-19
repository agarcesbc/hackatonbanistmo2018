import pandas as pd
import matplotlib.pyplot as plt


def leer_datos():
    '''
    esta funcion el trabajo que realiza es cargar la informacion en la variable data, e imprimir las relaciones entre las columnas
    'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12'
    :return:
    '''

    #lectura del archivo
    data = pd.read_csv("../dataset/hackaton_training_v1.csv")
    # generar un grafico con la correlacion de las variables
    plt.matshow(data[['v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12']].corr())
    plt.show()
    #imprimir en consola la matriz de correlacion
    print(data[['v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12']].corr())
    #devolver la data leida
    return data
