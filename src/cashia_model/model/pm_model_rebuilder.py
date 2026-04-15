import sys
import math
import pickle
import pickle


import cashia_core.data_channel.pm_datapipeline as pmpipe

from cashia_core.common_tools.utils import *
from cashia_core.common_tools.configuration.cashiaconstants import *
from cashia_core.ponderation.pm_amount_finder import *
from cashia_core.ponderation.pm_low_risk_client_ponderation import *
from cashia_core.common_tools.serialization import read_pickle


def can_convert_to_float(value):
    try:
        # Verificar si el valor puede ser convertido a float
        float_value = float(value)
        # Verificar si no es NaN
        if math.isnan(float_value):
            return False
        return True
    except (ValueError, TypeError):
        return False

class ModelsContainer:
    def __init__(self, name: str, model_keys: dict, storage):
        """
        Carga una colección de modelos a partir de sus keys lógicas.

        Args:
            name (str): Nombre del contenedor de modelos.
            model_keys (dict): Diccionario {model_name: resource_key}.
            storage: Instancia de LocalStorage o S3Storage.
        """
        self.name = name
        self.model = {}
        self.model_keys = model_keys
        self.storage = storage

        for model_name, model_key in model_keys.items():
            self.model[model_name] = read_pickle(self.storage, model_key)

            print(f"Reading model {model_name} ({self.name})")
            print(f"Model key : {model_key}")
            print(
                f"Training file : "
                f"{self.model[model_name].data_parameters['SourceFile']} "
                f"Date: {self.model[model_name].data_parameters['Date']}"
            )
            print(
                f"Threshold: "
                f"{self.model[model_name].best_gain_threshold:.2f}\n"
            )

# class ModelsContainer:
#      def __init__(self, name:str, models_filenames: list):
#          """ Carga una lista de modelos a partir de una lista de nombres de archivos.

#          Args:
#              models_filenames (list): Lista con los nombres de los archivos de modelos.
#          """
#          self.name = name
#          self.model = {}
#          for name in models_filenames:
#             with open(models_filenames[name], "rb") as f:
#                 self.model[name] = pickle.load(f)
#                 print(f"Reading model {name} {self.name}")
#                 print(f"Model file : {models_filenames[name]}")
#                 print(f"Training file : {self.model[name].data_parameters['SourceFile'] } Date:{self.model[name].data_parameters['Date']}")
#                 print(f"Threshold: {self.model[name].best_gain_threshold:.2f} \n")
                
class ModelRebuilder:

    def __init__(self, data_features: set, model=None, 
                 model_filename: str | None = None, stop_on_error=False):
        
        """Reconstruye un modelo y crea el pipeline para el tratamiento de los datos.

        Args:
            data_features (set): Conjunto de elementos (columnas) del juego de datos.
            cp_manager (MX_CP_manager): Un manejador de coódigos postales.
            model (ModelToTest, optional): Modelo a usarse para el tratamiento de los datos. Defaults to None.
            model_filename (str, optional): Nombre del archivo del modelo. Defaults to None.
            stop_on_error (bool, optional): Indica si debe detenerse la ejecución si se encuentra un error. 
                                            Defaults to False.
        """                       
            # Leer el modelo disponible
        self.model = None

        if model_filename != None:
            with open(model_filename, "rb") as f:
                self.model = pickle.load(f)
                print(f"Reading model")
                print(f"Model file : {model_filename}")
                print(f"Training file : {self.model.data_parameters['SourceFile'] } Date:{self.model.data_parameters['Date']}")
                print(f"Threshold: {self.model.best_gain_threshold} \n")
        elif model != None:
                self.model = model
        else:
            print("No model or model filename provided")
            sys.exit()

        # print("Model trainad with data from ",self.model.data_parameters['Training init date'],
        #       " to ",self.model.data_parameters['Trainig en date'])

        #self.model = pickle.load(open(model_filename, 'rb'))

        self.pipeline = pmpipe.DataPipeline(self.model.features,
                                            self.model.conf_column,
                                            data_features, 
                                            self.model.data_parameters,
                                            stop_on_error)
        
    def max_amount(self, row, amounts, p):
        previous_amount = 0
        for amount in amounts:
            if row[amount] >= p:
                return(int(previous_amount))
            previous_amount = amount
        return int(previous_amount)

    def prepare_data(self, data):
        
        """Verifica y prepara los datos para ser procesados por el modelo.

        Args:
            data (dataframa): Conjunto de datos a ser procesados.

        Returns:
            lista de errores, dataframe: Lista de errores encontrados al procesar 
                                         los datos y el dataframe procesado.
        """            
        errors, data = self.pipeline.process_data(data)

        # print_message("Feature enginnering")
        # self.feature_engineering = pm_eng.FeatureEngineering(self.model.min_max_stats, self.model.columns)

        # # Onehot encoding
        # data = self.feature_engineering.one_hot_encoding(data, self.pipeline.categorical_features)

        # # Estandarización/Normalización de los datos
        # data = self.feature_engineering.normalize(data, list(self.pipeline.quantitative_features), 
        #                                              self.model.feature_engineering)

        return errors, data
    
    def predict_proba(self, data):
        """Predice la probabilidad para la variable objetivo.

        Args:
            data (dataframe): dataframe usado para calcular la probabilidad.

        Returns:
            float: probabilidad de que la variable objetivo tenga un valor igual a 1.
        """        
        return self.model.pipeline.predict_proba(data)
        
    def predict_max_amount(self,
                           data,
                           original_data,
                           model_pond,
                           model_key,
                           approval_labels=[0, 1],
                           add_ponderation_flag=True):
        
        """Calcula el monto máximo que se le puede asignar a un prospecto en base a su información.

        Args:
            data (dataframe): Datos del prospecto tratados (estandarizados, one hot encoding, etc).
            original_data (datarame): Datos en su estado original.

        Returns:
            tuple: una tumpla con los siguientes elementos:
                    - dataframe: Los datos originales con columnas adicionales del máximo monto y
                                de si asignar un crédito o no.
                    - dictionary: Diccionarios con los elementos'aprove','default_probability' y 'max_amount'.
        """        

        print_message("Computing scores")

        data, original_data = self.predict_for_amounts(data,
                                                       original_data,
                                                       model_pond,
                                                       model_key, 
                                                       approval_labels,
                                                       add_amounts=True,
                                                       add_ponderation_flag=add_ponderation_flag
                                                       )

        aprove = int(original_data.loc[0, 'Aprobar'])
        amount = float(original_data.loc[0,"Monto máximo"])
        ponderated = bool(original_data.loc[0,"Ponderado"])

        if amount > 0:
            probability = float(original_data.loc[0, str(original_data.loc[0,"Monto máximo"])])
        else:
            min_amount = model_pond[original_data.loc[0,"Unidad"]].min_amount
            probability = float(original_data.loc[0, str(min_amount)])

        returnvalue = {'approve':aprove,'default_probability':probability, 
                       'max_amount':amount, 'ponderated':ponderated}

        return original_data, returnvalue
        
    def predict_for_amounts(self, 
                            data, 
                            original_data,
                            model_pond,
                            model_key,
                            approval_labels,
                            add_amounts=False, 
                            add_max_proba=False, 
                            add_threshold=False,
                            add_ponderation_flag=True):
        """Calcula para diferentes montos la probabilidad de que la variable objetivo sea igual a 1.

        Args:
            data (dataframe): Datos de los prospectos tratados y listos para ser evaluados (estandarizados,
                              one hot encoding, etc).
            original_data (dataframe): Datos en el estado original que los provió el usuario.
            min_amount (int): Mínimo moonto a ser evaluado.
            max_amount (int): Máximo monto a ser evaluado.
            increment (_type_): Incremento entre los montos.

        Returns:
            tuple: Tupla con los datos tratados y los datos originales con las columnas agregadas
                   de "Monto máximo" y "Aprobar".
        """        

        # Calculamos en donde detenemos los cálculos de probabilidad para cada monto
        stop_amount = MAX_AMOUNT+EXPLORE_INCREMENT

        amounts = []
        for monto in range(MIN_EXPLORE_AMOUNT, stop_amount, EXPLORE_INCREMENT):
            # Comentado por que ahora tenemos el feauture enginering en el pipe del modelo
            data['Monto'] = monto
            y_pred = self.predict_proba(data)
            prob_col_name = str(monto)
            amounts.append(prob_col_name)
            original_data[prob_col_name] = y_pred[:,1]

        # Obtenemos la probabilidad máxima obtenida
        original_data["MaxProba"] = original_data[amounts].max(axis=1)

        # Obtenemos pa probabilidad promedio obtenida
        original_data["MeanProba"] = original_data[amounts].mean(axis=1)

        original_data['Threshold'] = original_data.apply(lambda row: model_pond[row['Unidad']].threshold, axis=1)

        # Si su probabilidad máxima no pasa el umbral (threshold) entonces se ponderara
        original_data["Ponderado"] = original_data['MaxProba'] < original_data['Threshold']
     
        # Buscamos que monto máximo se le puede asignar
        original_data["Monto máximo"] = original_data.apply(lambda row: model_pond[row['Unidad']].find_max_amount(row), axis=1)

        # Se aprueba si se le ha podido asignar un monto máximo
        original_data['Aprobar'] = original_data.apply(lambda row: approval_labels[1] if row["Monto máximo"] > 0 
                                                       else approval_labels[0], axis=1)

        # Si se le asigó un monto máximo, regresamos la probabilidad de ese moto, de otra forma regresamos
        # la probabilidad del primer monto que fue explorado es decir de min_amount
        original_data["Proba"] = original_data.apply(lambda row: row[str(row["Monto máximo"])] if row["Monto máximo"] 
                                                     else row[str(model_pond[row['Unidad']].min_amount)], axis=1)

        if not add_max_proba: 
            original_data = original_data.drop("MaxProba", axis=1)

        if not add_amounts: 
            original_data = original_data.drop(columns=amounts)
        
        if not add_threshold: 
            original_data = original_data.drop('Threshold', axis=1)

        if not add_ponderation_flag:
            original_data = original_data.drop(columns='Ponderado')

        return data, original_data