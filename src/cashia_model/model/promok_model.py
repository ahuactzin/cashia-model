import joblib
import pickle

class ModelToTest:
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

    def save_with_pickle(self):
        filename = self.name+"_"+self.model_conf+".sav"
        pickle.dump(self, open("./models/"+filename, 'wb'))

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