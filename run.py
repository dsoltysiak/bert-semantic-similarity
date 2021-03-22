import os.path
from test import test_eval
from train import train_model
from dataset import data_preprocessing
from tensorflow.keras.models import load_model

train_df, valid_df, test_df, y_train, y_valid, y_test = data_preprocessing()

if not os.path.isfile("./Model/saved_model.pb"):
  model = train_model(train_df, valid_df, y_train, y_valid)
  model.save("./Model")
else:
  model = load_model("./Model")

test_eval(model, test_df, y_test)