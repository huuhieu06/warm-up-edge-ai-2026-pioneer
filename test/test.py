from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense

def build_edge_cnn(input_shape=(96, 96, 3), num_classes=10):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        ReLU(),
        
        SeparableConv2D(64, (3, 3), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        ReLU(),
        
        SeparableConv2D(128, (3, 3), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        ReLU(),
        
        SeparableConv2D(256, (3, 3), strides=(2, 2), padding='same', use_bias=False),
        BatchNormalization(),
        ReLU(),
        
        SeparableConv2D(256, (3, 3), strides=(1, 1), padding='same', use_bias=False),
        BatchNormalization(),
        ReLU(),
        
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])
    return model

model = build_edge_cnn()
model.summary() # Tổng tham số khoảng 118k