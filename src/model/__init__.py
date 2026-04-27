import os
from importlib import import_module

model_class_dict = {}

def regist_model(model_class):

    model_name = model_class.__name__.lower()
    if model_name in model_class_dict:
        return model_class 
    model_class_dict[model_name] = model_class
    return model_class

def get_model_class(model_name: str):

    return model_class_dict[model_name.lower()]

current_dir = os.path.dirname(__file__)
for file_name in os.listdir(current_dir):
    if file_name == '__init__.py' or not file_name.endswith('.py'):
        continue
    
    module_name = file_name[:-3]

    try:
        import_module('src.model.{}'.format(module_name))
    except ImportError as e:
        print(f"Failed to import {module_name}: {e}")

if 'file_name' in locals():
    del file_name