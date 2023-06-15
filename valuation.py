import json
import os
import shutil
import sys
import cv2
import torch
from utils.yolov5_detector import Yolov5TSDetector
from utils.general import logger
from utils.general import get_yaml
from utils.plots import Annotator
from utils.ImageFailedShower import FailedIconHtml
from pathlib import Path

FILE = Path(__file__).resolve()

ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))


def model_pt_to_ts(export_model,use_model_name = "exp"):
    # Parameters
    saved_model_path = f"{ROOT}/runs/train/{use_model_name}/weights/best.pt"
    img = torch.zeros((1, 3, 640, 640))  # image size, (1, 3, 320, 192) iDetection
    if not os.path.exists("./model_package"):
        os.makedirs("./model_package")
    # Load pytorch model
    model = torch.load(saved_model_path, map_location=torch.device("cpu"))["model"].float()
    model.eval()

    # Export to onnx
    model.model[-1].export = True  # set Detect() layer export=True
    _ = model(img)  # dry run
    traced_script_module = torch.jit.trace(model, img, strict=False)
    d = {"shape": img.shape, "stride": int(max(model.stride)), "names": model.names}
    extra_files = {"config.txt": json.dumps(d)}
    traced_script_module.save(export_model, _extra_files=extra_files)


class ModelValuation:
    def __init__(self,config, yolov5_ts, data_version):
        self.model_config = config["model_config"]
        self.test_camera_path = config["test_camera_path"]
        self.original_path = config["original_path"]
        self.val_image_num = None
        self._model_versions = data_version
        self.show_image = config["show_image"]
        self.model_version = config["export_model"]
        self.save_precision_file = config["save_precision_file"]
        self.original_image_save_path = config["original_image_save_path"]
        self.camera_image_save_path = config["camera_image_save_path"]
        self.val_file = config["val_file"].replace("data_version", data_version)
        self.icons_path = config["icons_path"]
        self.failed_icons_path = config["failed_icons_path"]
        self.val_file_coco = config["val_file_coco"].replace("data_version", data_version)
        self.failed_icons_image = config["failed_icons_image"]
        self.yolov5_ts = yolov5_ts

        with open(self.val_file, "r") as f:
            self.val_file_data = json.load(f)
            f.close()
    def achieve_mard_information(self):
        image_infor = {}
        for key, values in self.val_file_data.items():
            s = []
            for value in values:
                image_in = {}
                key = value["category_id"]
                image_in[f"{key}"] = value["bbox"]
                s.append(image_in)
                image_name = value["image_name"]
            image_infor[f"{image_name}"] = s
        logger.info("Raw data processing completed")
        return image_infor

    def load_yolov5_categories(self):
        with open(self.val_file_coco) as file:
            json_str = json.load(file)
        self.val_image_num = len(json_str['images'])
        logger.info(f"YOLOV5 datasets: images({len(json_str['images'])}), categories({len(json_str['categories'])})")
        categories_list = json_str['categories']
        categories = dict()
        for item in categories_list:
            categories[item["id"]] = item["name"]

        return categories


    def image_detect(self):
        num = 0
        image_infor = self.achieve_mard_information()
        # invalid_image = ["369.png","13.png","301.png","354.png","21.png","232.png","65.png","244.png","196.png","195.png","161.png","102.png"]
        invalid_image = []
        failed_icons = []
        dec_failed_icons = []
        images_outputs = []
        failed_icons_id = []
        failed_image_infor = []
        categories = self.load_yolov5_categories()
        print(self.val_image_num)
        lamp_icon_id = list(categories.keys())[list(categories.values()).index("icon_945")]
        for d in os.listdir(self.test_camera_path):
            if "checkpoint" in d:
                continue
            ori_path = self.original_path + d[:-4] + '.png'
            img_infor = image_infor[d[:-4] + '.png']
            image_name = d[:-4] + '.png'
            file_path = self.test_camera_path + d
            img = cv2.imread(file_path)
            # print("file_path:", file_path)
            output = self.yolov5_ts.detect(img)
            # print("output:",output)
            ori_labels = []
            for inf in img_infor:
                for key, _ in inf.items():
                    ori_labels.append(int(key))
            ori_labels.sort(reverse=False)
            camera_labels = []
            for obj in output:
                label = obj.label
                label = label.split(":")[0]
                if label == lamp_icon_id:
                    continue
                camera_labels.append(int(label))
            camera_labels.sort(reverse=False)
            dif_ori = list(set(ori_labels).difference(set(camera_labels)))  # 返回集合的差集，第一个集合有第二个集合没有
            dif_out = list(set(camera_labels).difference(set(ori_labels)))
            if 3 > len(dif_ori) > 0 and 3 > len(dif_out):
                if d not in invalid_image:
                    # if dif_ori == [427] or dif_ori == [30]: #or dif_ori == [186] or dif_ori == [340]:  # or dif_ori == [203,337]:
                    #     continue
                    failed_icons.append({d:dif_ori})
                    dec_failed_icons.append({d:dif_out})
                    images_outputs.append({d:output})
                    failed_image_infor.append({image_name: image_infor[d[:-4] + '.png']})
                    [failed_icons_id.append(icon_id) for icon_id in dif_ori]
                    num += 1
                    logger.info(f"This is the {num} failed picture")
                    if self.show_image:
                        self.image_display(dif_ori,dif_out,ori_path,file_path,img_infor,output)

        trans = self.remove_failed_icon(failed_icons, failed_icons_id,failed_image_infor,dec_failed_icons,images_outputs)
        return 1 - round((trans) / self.val_image_num , 2)


    def image_display(self,dif_ori,dif_out,ori_path,file_path,img_infor,output):
        ori_img = cv2.imread(ori_path)
        img = cv2.imread(file_path)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotator = Annotator(ori_img, line_width=4, font_size=8)
        for inf in img_infor:
            for key, value in inf.items():
                box = [torch.tensor(value[0]), torch.tensor(value[1]),
                       torch.tensor(value[0]) + torch.tensor(value[2]),
                       torch.tensor(value[1]) + torch.tensor(value[3])]
                label = key
            if int(label) not in dif_ori:
                continue
            annotator.box_label(box, label, color=(255, 0, 255), txt_color=(255, 255, 255))

        annotator1 = Annotator(img, line_width=5, font_size=13)
        for obj in output:
            label = obj.label
            label = label.split(":")[0]
            if int(label) not in dif_out:
                continue
            annotator1.box_label(obj.bbox.box, label, color=(86, 255, 86), txt_color=(255, 0, 0))
        if not os.path.exists(self.camera_image_save_path):
            os.makedirs(self.camera_image_save_path)
        if not os.path.exists(self.original_image_save_path):
            os.makedirs(self.original_image_save_path)
        ori_image_save_path = os.path.join(self.original_image_save_path, ori_path.split("/")[-1])
        camera_fail_image_path = os.path.join(self.camera_image_save_path,file_path.split("/")[-1])
        annotator1 = annotator1.result()
        annotator = annotator.result()
        cv2.imwrite(ori_image_save_path, ori_img)
        cv2.imwrite(camera_fail_image_path,img)



    def get_icon_md5(self,failed_icons):
        id_and_md5 = {}
        for data in failed_icons:
            for data_key, data_values in data.items():
                for data_value in data_values:
                    for key, values in self.val_file_data.items():
                        if data_key.split(".")[0] == key:
                            for value in values:
                                try:
                                    if value["category_id"] == data_value:
                                        id_and_md5[data_value] = value["md5"]
                                except:
                                    continue

        return id_and_md5

    def get_icon_name(self,failed_icons_id):
        id_to_name = {}
        with open(self.val_file_coco,"r") as f:
            datas = json.load(f)
            f.close()
        for key,values in datas.items():
            if key == "categories":
                for id in failed_icons_id:
                    for value in values:
                        if int(id) == int(value["id"]):
                            id_to_name[id] = value["name"]
        return id_to_name

    def cal_iou_xyxy(self,box1, box2):
        x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
        x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2] + box2[0], box2[3] + box2[1]
        # 计算两个框的面积
        s1 = (y1max - y1min + 1.0) * (x1max - x1min + 1.0)
        s2 = (y2max - y2min + 1.0) * (x2max - x2min + 1.0)

        # 计算相交部分的坐标
        xmin = max(x1min, x2min)
        ymin = max(y1min, y2min)
        xmax = min(x1max, x2max)
        ymax = min(y1max, y2max)

        inter_h = max(ymax - ymin + 1, 0)
        inter_w = max(xmax - xmin + 1, 0)

        intersection = inter_h * inter_w
        union = s1 + s2 - intersection

        # 计算iou
        iou = intersection / union
        return iou



    def str_to_list(self,s):
        s_list = []
        s = str(s)
        s = s[1:-1]
        s = s.split(", ")
        for i in range(len(s)):
            if i == 2:
                s[i] = float(s[i]) - float(s[0])
            elif i == 3:
                s[i] = float(s[i]) - float(s[1])
            s_list.append(float(s[i]))
        return s_list

    def save_failed_images(self,md5_id, image_name, icon_name, failed_image_infor,dec_failed_icons,images_outputs):
        # print("22:",len(failed_image_infor))
        categories = self.load_yolov5_categories()
        ori_image_path = os.path.join(self.original_path,image_name)
        camera_image_path = os.path.join(self.test_camera_path, image_name)
        ori_img = cv2.imread(ori_image_path)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        camera_img = cv2.imread(camera_image_path)
        camera_img = cv2.cvtColor(camera_img, cv2.COLOR_BGR2RGB)
        print(len(failed_image_infor),len(dec_failed_icons),len(images_outputs))
        annotator = Annotator(ori_img, line_width=3, font_size=6)
        for i in range(len(failed_image_infor)):
            for name, values in failed_image_infor[i].items():
                # print("image_name,name:", image_name,name)
                if image_name == name:
                    failed_image_save_path = os.path.join(self.failed_icons_image, image_name)
                    ori_image_save_path = os.path.join(failed_image_save_path, "ori_")
                    ori_image_save_path = ori_image_save_path + icon_name
                    out_image_save_path_camera = os.path.join(failed_image_save_path, icon_name)
                    out_image_save_path_camera = out_image_save_path_camera + "_out_" + icon_name
                    for single_value in values:
                        for key, value in single_value.items():
                            label_ori = key
                        # print("label,md5_id:",label,md5_id)
                        if int(label_ori) == md5_id:
                            box = [torch.tensor(value[0]), torch.tensor(value[1]),
                                   torch.tensor(value[0]) + torch.tensor(value[2]),
                                   torch.tensor(value[1]) + torch.tensor(value[3])]
                            box_ori = [(value[0]), (value[1]),
                                       (value[0]) + (value[2]),
                                       (value[1]) + (value[3])]
                            annotator.box_label(box, icon_name, color=(255, 0, 255), txt_color=(255, 255, 255))
                            break
                    annotator1 = Annotator(camera_img, line_width=3, font_size=6)
                    if len(dec_failed_icons[i][image_name]):
                        for output in images_outputs[i][image_name]:
                            label = output.label
                            label = label.split(":")[0]
                            if int(label) not in dec_failed_icons[i][image_name]:
                                continue
                            box_out = self.str_to_list(str(output.bbox))
                            iou = self.cal_iou_xyxy(box_ori, box_out)
                            if iou > 0.2:
                                icon_name_out = categories[int(label)]
                                annotator1.box_label(output.bbox.box, icon_name_out, color=(86, 255, 86), txt_color=(255, 0, 0))
                                out_image_save_path_camera = os.path.join(failed_image_save_path, icon_name)
                                out_image_save_path_camera = out_image_save_path_camera + "_out_" + icon_name_out

                    if not os.path.exists(ori_image_save_path):
                        os.makedirs(ori_image_save_path)
                    else:
                        shutil.rmtree(ori_image_save_path)
                        os.makedirs(ori_image_save_path)

                    if not os.path.exists(out_image_save_path_camera):
                        os.makedirs(out_image_save_path_camera)
                    else:
                        shutil.rmtree(out_image_save_path_camera)
                        os.makedirs(out_image_save_path_camera)
                    failed_images_save_path = os.path.join(ori_image_save_path, image_name)
                    failed_images_save_path_out = os.path.join(out_image_save_path_camera, image_name)
                    annotator = annotator.result()
                    annotator1 = annotator1.result()
                    cv2.imwrite(failed_images_save_path, annotator)
                    cv2.imwrite(failed_images_save_path_out, annotator1)


    def remove_failed_icon(self,failed_icons,failed_icons_id,img_infor,dec_failed_icons,images_outputs):
        failed_id = []
        trans = 1
        id_and_md5 = self.get_icon_md5(failed_icons)
        id_to_name = self.get_icon_name(failed_icons_id)
        [failed_id.append(id) for id,name in id_to_name.items()]
        name_ids = []
        [name_ids.append(name_id) for name_id in id_to_name.keys()]
        for md5_id in id_and_md5.keys():
            if md5_id in name_ids:
                icon_name = id_to_name[md5_id]
                md5_name = id_and_md5[md5_id]
                for data in failed_icons:
                    for key,value in data.items():
                        if md5_id in value:
                            print(key)
                            image_name = key
                self.save_failed_images(md5_id, image_name, icon_name, img_infor,dec_failed_icons,images_outputs)
                origin_icon_path = self.icons_path + icon_name + "/" + md5_name + ".png"
                origin_image_path = self.test_camera_path + image_name
                fail_icons_path = self.failed_icons_path + icon_name
                if not os.path.exists(fail_icons_path):
                    os.makedirs(fail_icons_path)
                failed_icon_path = self.failed_icons_path + icon_name + "/" + md5_name + ".png"
                try:
                    shutil.copyfile(origin_icon_path, failed_icon_path)
                    logger.info(f"{trans} icon transferring")
                    trans += 1
                except:
                    logger.info(f"{trans} icon has been transferred ")
                    trans += 1
                    continue
        return trans

    def failed_image_show(self,icon_category_folder):
        self.load_yolov5_categories()
        obj = FailedIconHtml()
        obj.output_html(icon_category_folder,self.val_image_num,self._model_versions)



def val(config, data_version,icon_category_folder):
    if os.path.exists("../../scc/dataset/data/dataset/latest/failed_images/"):
        shutil.rmtree("../../scc/dataset/data/dataset/latest/failed_images/")
    if os.path.exists("../../scc/dataset/data/dataset/latest/failed/"):
        shutil.rmtree("../../scc/dataset/data/dataset/latest/failed/")
    model_config = config["model_config"]
    model_version = config["export_model"]
    save_precision_file = config["save_precision_file"]
    yolov5_ts = Yolov5TSDetector(model_config)
    valuation_model = ModelValuation(config, yolov5_ts, data_version)
    precision = valuation_model.image_detect()
    valuation_data = {model_version : str(precision)}
    valuation_data = json.dumps(valuation_data)
    # valuation_model.failed_image_show(icon_category_folder)
    file = open(f"{save_precision_file}","w")
    file.write(valuation_data)
    file.close()
    logger.info(f"The accuracy of model verification is {precision}")

def show_failed_images(config, data_version,icon_category_folder):
    model_config = config["model_config"]
    yolov5_ts = Yolov5TSDetector(model_config)
    valuation_model = ModelValuation(config, yolov5_ts, data_version)
    valuation_model.failed_image_show(icon_category_folder)

def switch(export_model, use_model_name):
    # export_model = config["export_model"]
    # use_model_name = config["use_model_id"]
    model_pt_to_ts(export_model,use_model_name)
    logger.info(f"The ts model is saved in {export_model}")
    logger.info("The conversion of model type is succeeded!")


if __name__ == "__main__":
    yaml_path = "data/detect_config.yaml"
    config = get_yaml(yaml_path)
    val(config)

