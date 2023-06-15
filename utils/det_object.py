# import parse


class DetBbox:
    format_str = "({:0.3f}, {:0.3f}, {:0.3f}, {:0.3f})"

    def __init__(self, x0, y0, x1, y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    #     @property
    #     def xmin(self):
    #         return self.xc - self.w / 2

    #     @property
    #     def xmax(self):
    #         return self.xc + self.w / 2

    #     @property
    #     def ymin(self):
    #         return self.yc - self.h / 2

    #     @property
    #     def ymax(self):
    #         return self.yc + self.h / 2
    @property
    def box(self):
        return [self.x0, self.y0, self.x1, self.y1]

    def __str__(self) -> str:
        return self.format_str.format(self.x0, self.y0, self.x1, self.y1)

    # @classmethod
    # def from_string(cls, bbox_str):
    #     xc, yc, w, h = parse.parse(cls.format_str, bbox_str)
    #     return cls(xc, yc, w, h)


class DetObject:
    format_str = "class_id={}, score={:0.3f}, bbox={}"

    def __init__(self, bbox: DetBbox, score: float, class_id: int):
        self.bbox = bbox
        self.score = score
        self.class_id = class_id

    @property
    def label(self):
        return "{}:{:0.2f}".format(self.class_id, self.score)

    def __str__(self) -> str:
        return self.format_str.format(self.class_id, self.score, self.bbox)

    # @classmethod
    # def from_string(cls, obj_str):
    #     name, score, bbox_str = parse.parse(cls.format_str, obj_str)
    #     bbox = DetBbox.from_string(bbox_str)
    #     obj = cls(bbox, score, name)
    #     return obj
