#Gestion del modelo
import torch
#Modelo preentrenado de T5
from transformers import T5ForConditionalGeneration, T5Tokenizer
#Carga del dataset
from torch.utils.data import DataLoader, Dataset
#Barras de progreso
from tqdm import tqdm
#Manipular datos de csv
import pandas as pd
