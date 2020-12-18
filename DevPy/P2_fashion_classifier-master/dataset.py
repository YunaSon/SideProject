from os import path
from keras.preprocessing.image import ImageDataGenerator


class Dataset:
    def __init__(self, src=path.join(path.dirname(path.abspath(__file__)), "dataset")):
        self._src = src
        self._train = None
        self._validation = None
        self._test = None

    def generate_dataset(self, target_size=(224, 224), batch_size=4, class_mode="categorical", shuffle=True, seed=42):
        img_generator = ImageDataGenerator(rescale=1. / 255,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True,
                                           validation_split=0.2)

        self._train = img_generator.flow_from_directory(
            self._src,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=shuffle,
            seed=seed,
            subset='training')

        self._validation = img_generator.flow_from_directory(
            self._src,
            target_size=target_size,
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=shuffle,
            seed=seed,
            subset='validation')

    @property
    def train(self):
        return self._train

    @property
    def validation(self):
        return self._validation

    @property
    def test(self):
        return self._test

    @train.setter
    def train(self, dataset):
        # todo : dataset validation
        self._train = dataset

    @validation.setter
    def validation(self, dataset):
        # todo : dataset validation
        self._validation = dataset

    @test.setter
    def test(self, dataset):
        # todo : dataset validation
        self._test = dataset

    def make_generator_info(self, subset='train'):
        target = None
        if subset == 'train':
            target = self.train
        if subset == 'validation':
            target = self.validation
        elif subset == 'test':
            target = self.test

        if target is None:
            return ""

        generator_info = f"""{"=" * 10} {subset} dataset generator {"=" * 10}\n
            Class count : {target.num_classes}\n
            Class mode : {target.class_mode}\n
            Image shape : {target.image_shape}\n
            Data type : {target.dtype}\n
            Batch size : {target.batch_size}\n
            Shuffle : {target.shuffle}\n
            Seed : {target.seed}\n
            {"=" * 50}
            """.split('\n')

        return "\n".join([info.strip() for info in generator_info if info])

    def __str__(self):
        generator_info = ""
        for subset in ["train", "validation", "test"]:
            generator_info += self.make_generator_info(subset)
        return generator_info
