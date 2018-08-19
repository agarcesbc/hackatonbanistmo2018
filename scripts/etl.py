import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def leer_datos():
    '''
    esta funcion el trabajo que realiza es cargar la informacion en la variable data, e imprimir las relaciones entre las columnas
    'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12'
    :return:
    '''

    # lectura del archivo
    data = pd.read_csv("../dataset/hackaton_training_v1.csv")
    # generar un grafico con la correlacion de las variables
    plt.matshow(data[['v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12']].corr())
    plt.show()
    # imprimir en consola la matriz de correlacion
    print(data[['v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12']].corr())



    #se genera el dataset con nuevas variables dependiendo de su distribucion y sus parametros encontrados en
    #las investigaciones
    np.random.seed(1234)
    data_extended = data[['Unnamed: 0', 'v_0']]

    # data_extended['actividad_economica']
    # data_extended['nivel_academico']
    desv_activos = 553638482.2073354
    m_activos = 242157.1
    data['activos'] = data_extended['activos'] = np.random.gumbel(m_activos, desv_activos, 5000)
    desv_pasivos = 310087989.61633337
    m_pasivos = 8271.5
    data['pasivo'] = data_extended['pasivo'] = np.random.gumbel(m_pasivos, desv_pasivos, 5000)
    m_patrimonio = 150000
    desv_patrimonio = 382050840.36867464
    data['patrimonio'] = data_extended['patrimonio'] = np.random.gumbel(m_patrimonio, desv_patrimonio, 5000)
    m_estado_civil = 2
    desv_estado_civil = 0.8429593542892511
    # 7 valores posibles (poisson)
    data['estado_civil'] = data_extended['estado_civil'] = np.random.poisson(m_estado_civil, 5000)
    # data_extended['personas_a_cargo']

    data_extended.to_csv("../dataset/extrainformation.csv", index=False)
    # devolver la data leida
    return data

leer_datos()
