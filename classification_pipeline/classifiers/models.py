import json
from sklearn.svm import SVC

class ClassificationModels(object):
    def __init__(self, model_config_path=None):
        if model_config_path is not None:
            with open(model_config_path) as w:
                self.model_def = json.load(w)

    def create_model(self, model_name=None, params=None):
        print(f"Model Name:{model_name}")
        print(f"Model Parameters:{params}")
        if model_name is not None and params is None:
            return None
        if model_name == 'SVC' and params is not None:
            return SVC(**params)

    def get_models(self):
        models = []
        for model_definition in self.model_def:
            defined_model = self.create_model(
                model_name=model_definition['model'],
                params=model_definition['params']
                )
            models.append(defined_model)
        return models