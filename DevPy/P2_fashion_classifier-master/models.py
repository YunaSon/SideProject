from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications import MobileNetV2
from keras.models import Model


class CustomModel:
    def __init__(self, name, weights='imagenet'):
        self._name = name
        self._weights = weights

    @property
    def custom_mobile_net_v2(self):
        base_model = MobileNetV2(weights=self._weights, include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(512, activation='relu')(x)
        preds = Dense(120, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=preds)
        return model

    def __call__(self):
        if self._name.lower() == "mobilenetv2":
            return self.custom_mobile_net_v2


if __name__ == '__main__':
    my_model = CustomModel("mobilenetv2")()
