import joblib
import pickle
from cashia_core.common_tools.configuration.resource_keys import get_resource_path

class TrainModel:
    def __init__(self, name, parameters, is_probabilistic = True, model_conf =""):
        self.name = name
        self.parameters = parameters
        self.pipeline = None
        self.probabilistic = is_probabilistic
        self.model_conf = model_conf
        self.feature_engineering = None
        self.best_accurracy_threshold = None
        self.best_gain_threshold = None
        self.min_max_stats = None
        self.columns = None
        self.features = None
        self.conf_column = None
        self.data_parameters = dict()

    def min_value(self, characteristic: str):
        return self.get_param_value(characteristic, "Min")
    
    def max_value(self, characteristic: str):
        return self.get_param_value(characteristic, "Max")

    def get_param_value(self, characteristic:str , parameter:str):
        # Obtener el rango de edades desde el modelo
        features = self.features
        value = features.loc[features['Característica'] == characteristic, parameter].values[0]
        value_type = features.loc[features['Característica'] == characteristic, 'Sub tipo'].values[0]

        if parameter in ["Min", "Max"]:
            if value_type == "Entero":
                return int(value)
            elif value_type == "Real":
                return float(value)
        
        return str(value)


    def save(self):
        filename = self.name+".sav"
        joblib.dump(self, filename)

    def save_with_pickle(self, storage):
        """Guarda el modelo serializado en el almacenamiento configurado.

        Args:
            storage: Instancia de LocalStorage o S3Storage.
        """

        models_dict = {"Random_Forest": "M01",
                       "Random_Forest_SMOTE": "M02",
                       "Regresion": "M03",
                       "Naive Bayes": "M04",
                       "Ada Boost": "M05",
                       "SVC": "M06",
                       "KNN": "M07"
                       }

        prefix_name = models_dict.get(self.name, "UnknownModel")

        filename = f"{prefix_name}_{self.model_conf}.sav"

        output_key = (
            f"{get_resource_path('models')}{filename}"
        )

        storage.write_bytes(output_key, pickle.dumps(self))


    def __str__(self):
        features = "Model name: "+self.name+"\n"
        features += "Feature engineering: "+self.feature_engineering+"\n"
        features += "Best gain threshold: "+str(self.best_gain_threshold)+"\n"

        return str(features)
    
    def __repr__(self):
        features = "Model name: "+self.name+"\n"
        features = "Feature engineering: "+self.feature_engineering+"\n"
        features = features+"Best gain threshold: "+str(self.best_gain_threshold)+"\n"

        return str(features)