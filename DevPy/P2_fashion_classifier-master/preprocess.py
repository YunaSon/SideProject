from dataset import Dataset


class PreProcess:
    def __init__(self):
        self.dataset = Dataset()
        self.dataset.generate_dataset()

    def print_dataset_info(self):
        print(self.dataset)


if __name__ == '__main__':
    pp = PreProcess()