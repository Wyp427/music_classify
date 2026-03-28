class LabelMapper:
    def __init__(self, labels=None):
        default_labels = [
            "布鲁斯",
            "古典",
            "乡村",
            "迪斯科",
            "嘻哈",
            "爵士",
            "金属",
            "流行",
            "雷鬼",
            "摇滚",
        ]
        self.labels = list(labels or default_labels)
        self.mapping = {index: label for index, label in enumerate(self.labels)}

    def get_label(self, encoded_label):
        return self.mapping.get(encoded_label)

    def get_labels(self):
        return self.labels.copy()


class GTZANLabelMapper(LabelMapper):
    def __init__(self):
        super().__init__()