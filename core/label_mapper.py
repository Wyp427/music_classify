class GTZANLabelMapper:
    def __init__(self):
        # 定义 GTZAN 编码与中文标签的映射关系
        self.mapping = {
            0: "布鲁斯",
            1: "古典",
            2: "乡村",
            3: "迪斯科",
            4: "嘻哈",
            5: "爵士",
            6: "金属",
            7: "流行",
            8: "雷鬼",
            9: "摇滚"
        }

    def get_label(self, encoded_label):
        """
        根据编码标签获取对应的中文标签
        :param encoded_label: 编码标签
        :return: 中文标签
        """
        return self.mapping.get(encoded_label)
