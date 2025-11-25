from tensorflow.keras.models import load_model

model = load_model("backend/model_weights.h5")
print(model.input_shape)
