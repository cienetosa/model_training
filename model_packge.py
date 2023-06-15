import csv
import os
import shutil
import sys
import hashlib
import datetime
import json
from pathlib import Path
import pandas as pd
from pandas import DataFrame
from utils.general import get_yaml
from utils.general import logger
import random
import argparse

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

up_path = Path.cwd().parent
up_path = Path(os.path.relpath(up_path, Path.cwd()))

def parse_ope(known = False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir", type=str, default="../../scc/dataset/data/resources", help="package model information for model_info.js"
    )
    parser.add_argument(
        "--download_dir", type=str, default="../../scc/dataset/test_dataset/",
        help="package model information for model_info.js"
    )
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def copy_camera_images(all_camera_path):
    if not os.path.exists(all_camera_path):
        os.makedirs(all_camera_path)

class ValDatasetEngine:
    @staticmethod
    def get_valid_dir(name, dirs):
        for i in dirs:
            if name.lower() in i.lower():
                return i

    @staticmethod
    def copy_images(directory):
        all_test_camera = "model_data/all_test_camera"
        if not os.path.exists(all_test_camera):
            os.makedirs(all_test_camera)
        images_names = [i[:-4] for i in os.listdir(directory) if i.endswith(".png")]
        json_names = [i.replace(".json", "") for i in os.listdir(directory) if i.endswith(".json")]
        for name in set(images_names) & set(json_names):
            shutil.copy(f"{directory}/{name}.png", f"{all_test_camera}/{name}.png")


    def generate_val_dataset(self):
        source_dir = "../../scc/dataset/data/resources"

        for path in os.listdir(source_dir):
            if not os.path.isdir(f"{source_dir}/{path}"):
                continue
            elif "NODE" in path:
                continue
            elif path in ["Incremental_0.1.0", "Incremental_0.2.0"]:
                for root, dirs, files in os.walk(f"{source_dir}/{path}"):
                    camera = self.get_valid_dir("camera", dirs)
                    if camera:
                        camera_dir = f"{root}/{camera}"
                        self.copy_images(camera_dir)
            else:
                for root, dirs, files in os.walk(f"{source_dir}/{path}"):
                    for per_dir in dirs:
                        images_dir = f"{root}/{per_dir}"
                        self.copy_images(images_dir)
    def get_val_data(self,config, version):
        val_file_coco = config["val_file_coco"].replace("data_version", version)
        all_camera_path = config["all_camera_path"]
        new_origin_path = config["new_origin_path"]
        new_camera_path = config["new_camera_path"]
        all_adb_images = config["all_adb_images"].replace("data_version", version)

        self.generate_val_dataset()
        copy_camera_images(all_camera_path)
        all_camera_images = []
        image_num = 0
        [all_camera_images.append(image) for image in os.listdir(all_camera_path)]
        if not os.path.exists(new_origin_path):
            os.makedirs(new_origin_path)
        if not os.path.exists(new_camera_path):
            os.makedirs(new_camera_path)
        if os.path.exists(new_origin_path):
            shutil.rmtree(new_origin_path)
            os.makedirs(new_origin_path)
        if os.path.exists(new_camera_path):
            shutil.rmtree(new_camera_path)
            os.makedirs(new_camera_path)
        with open(val_file_coco, "r") as f:
            datas = json.load(f)
            f.close()
        for keys, values in datas.items():
            if keys == "images":
                for value in values:
                    original_path = value["original_path"]
                    original_images = original_path.split("/")[-1]
                    if original_images in all_camera_images:
                        image_num += 1
                        image_name = value["file_name"].replace(".png", ".jpg")
                        original_image_path = os.path.join(all_adb_images, image_name)
                        original_camera_image_path = os.path.join(all_camera_path, original_images)
                        val_original_image_path = os.path.join(new_origin_path, value["file_name"])
                        val_camera_image_path = os.path.join(new_camera_path, value["file_name"])
                        shutil.copyfile(original_image_path, val_original_image_path)
                        shutil.copyfile(original_camera_image_path, val_camera_image_path)
                        if image_num % 10 == 0:
                            logger.info(f"Copying picture {image_num}")



def get_map(exp_path,use_model_num = "exp",maps = "metrics/mAP_0.5:0.95"):
    model_csv_path = os.path.join(exp_path,use_model_num)
    model_csv_name = os.path.join(model_csv_path,"results.csv")
    frame=DataFrame(pd.read_csv(model_csv_name))
    temporary = []
    for j in frame.index:
        temporary.append(frame[maps][j])
        temporary.sort(reverse=True)
    return temporary[0]


def get_model_version(export_model):
    model_name = export_model.split("/")[-1]
    return model_name


def get_md5_file(relace_mode):
    model_path = relace_mode
    with open(model_path, 'rb') as f:
        model_md5 = hashlib.md5(f.read()).hexdigest()
        return model_md5



def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_hpy_current(exp_path,model_information,use_model_num = "exp"):
    current_train_hyp = f"{exp_path}/{use_model_num}/hyp.yaml"
    hpy_datas = {"hpy":get_yaml(current_train_hyp)}
    with open(model_information,"a+") as f:
        hpy_datas = json.dumps(hpy_datas, ensure_ascii=True, indent=4)
        # f.write("/n")
        f.write(hpy_datas)

def get_model_precison(save_precision_file,export_model):
    json_data = []
    for line in open(save_precision_file,"r"):
        json_data.append(json.loads(line))
    for data in json_data:
        precison = data[f"{export_model}"]
    return precison


def copy_and_delete(config,version):
    package_path = config["package_path"]
    val_data_path = config["val_data_path"]
    release_path = config["release_path"]
    val_file_coco = config["val_file_coco"].replace("data_version", version)
    json_name = "instances.json"
    copy_json_path = os.path.join(package_path,json_name)
    shutil.copyfile(val_file_coco, copy_json_path)
    shutil.copyfile("Content_Compare_Failed_Image_List.html", "../../scc/dataset/data/Content_Compare_Failed_Image_List.html")
    for root,dirs,files in os.walk(package_path):
        if root == package_path:
            for file in files:
                file_path = os.path.join(package_path,file)
                copy_file_path = os.path.join(release_path,file)
                shutil.copyfile(file_path,copy_file_path)
    if os.path.exists("../../scc/dataset/model"):
        shutil.rmtree("../../scc/dataset/model")
    shutil.copytree("../icon-detector/model","../../scc/dataset/model")
    shutil.rmtree(package_path)
    shutil.rmtree(val_data_path)


def find_new_data(all_dir_path):
    single_json_files = []
    for root,dir,files in os.walk(all_dir_path):
        all_json_files = []
        if len(files):
            for file in files:
                if ".json" in file and ".meta" not in file:
                    all_json_files.append(file)
            if len(all_json_files):
                file = random.sample(all_json_files,1)[0]
                file = os.path.join(root, file)
                single_json_files.append(file)
    single_json_files = sorted(single_json_files, key=lambda x: os.path.getmtime(x))
    # print(single_json_files)
    new_data = os.path.dirname(single_json_files[-1])
    if "download" in new_data.split("/")[-1]:
        new_data = new_data.split("download")[0] + "download"
    else:
        if "resources" in new_data:
            logger.info(f"The latest data path is {new_data}")
            resources_path = new_data.split("resources")[0] + "resources"
            resources_location = new_data.split("resources")[1].split("/")[1]
            new_data = os.path.join(resources_path, resources_location)
    return new_data



def get_new_category(download_dir):
    new_categories_list = []
    new_download = find_new_data(download_dir)
    new_mapping_json = os.path.join(new_download,"icon_name_mapping_table.json")
    old_maping_json = "data/icon_name_mapping_table.json"
    new_json_data = []
    file = open(new_mapping_json, "r")
    new_json_data = json.load(file)
    # print(new_json_data)
    file = open(old_maping_json, "r")
    old_json_data = json.load(file)
    new_categories_list = list(set(new_json_data.values())-set(old_json_data.values()))
    print("new_categories_list:",new_categories_list)
    return new_categories_list

def save_all_failed_icons(config):
    detect_new_data = config["detect_new_data"]
    try:
        file = open(detect_new_data, "r")
        if file.read() == "success":
            failed_icons_path = config["failed_icons_path"]
            header = []
            new_data = {}
            failed_icons_path = os.path.abspath(failed_icons_path)
            file_save_path = os.path.abspath(os.path.join(failed_icons_path, "../../"))
            failed_icons_file = os.path.join(file_save_path, "all_failed_file.csv")
            all_icons = os.listdir(failed_icons_path)
            f = open(os.path.join("model_package","model.json"),"r")
            datas = json.load(f)
            f.close()
            for key,value in datas.items():
                if key == "current_time":
                    key = "model_create_time"
                    new_data[key] = value
                else:
                    new_data[key] = value
                header.append(key)
            header.append("failed_icons")
            new_data["failed_icons"] = all_icons
            with open(failed_icons_file, 'a+', newline="\n", encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=header)  # 提前预览列名，当下面代码写入数据时，会将其一一对应。
                writer.writeheader()  # 写入列名
                writer.writerow(new_data)  # 写入数据
            print("All data has been written successfully！！！")
        else:
            print("have no new datas")
    except Exception as e:
        print("Exception in modify_failed_icons is {} !".format(repr(e)))

def modify_failed_icons(config):
    failed_icons_path = config["failed_icons_path"]
    all_icon_num_json_path = config["all_icon_num_json_path"]
    manual_icon_num_json_path = config["manual_icon_num_json_path"]
    data_all = []
    # f_all_filed = open(all_icon_num_json_path, "w")
    # f_all_filed.write("{}")
    file = open("failed_important_icons.txt", "w")
    if os.path.exists(os.path.join(all_icon_num_json_path)):
        f = open(os.path.join(all_icon_num_json_path), "r")
        datas = f.read()
        if len(datas):
            icon_counts = json.loads(datas)
            f.close()
            failed_icons = os.listdir(failed_icons_path)
            try:
                icon_counts = icon_counts["case_counts"]
                print(icon_counts)
                all_important_icons = [key for key,value in icon_counts.items() if value > 100]
                failed_important_icons = set(failed_icons) & set(all_important_icons)
                if len(failed_important_icons):
                    file.write("failed")
                else:
                    file.write("success")
            except Exception as e:
                print("Exception in modify_failed_icons is {} !".format(repr(e)))
        else:
            print("had no datas")
            file.write("success")
    else:
        file.write("success")

def get_package(config, model_version, use_model_id):
    cfg = parse_ope(True)
    new_categories = get_new_category(cfg.download_dir)
    exp_path = config["exp_path"]
    export_model = config["export_model"]
    save_precision_file = config["save_precision_file"]
    model_information = config["model_information"]
    current_time = get_time()
    model_md5 = get_md5_file(export_model)
    model_map = get_map(exp_path,use_model_id,maps = "metrics/mAP_0.5:0.95")
    resources_Incremental = []
    for line in open("../icon-category/Incremental_list.txt", "r"):
        data = line.strip("\n")
        resources_Incremental.append(data)
    model_precision = get_model_precison(save_precision_file, export_model)
    datas = {"current_time": current_time, "model_version": model_version,"resources_Incremental":resources_Incremental,"new_categories":new_categories,
             "model_map": model_map, "model_md5": model_md5, "model_precision": model_precision}
    datas = json.dumps(datas, ensure_ascii=False, indent=4)
    with open(model_information, "w") as f:
        f.write( datas + "\n")
        f.close()
    save_all_failed_icons(config)
    modify_failed_icons(config)
    logger.info("package for model_package")

if __name__ == "__main__":
    cfg = parse_ope(True)
    model_information = "model.json"
    resources_Incremental = []
    for line in open("../icon-category/Incremental_list.txt", "r"):
        data = line.strip("\n")
        resources_Incremental.append(data)
    new_categories = get_new_category(cfg.download_dir)
    datas = {"current_time": "current_time", "model_version": "model_version",
             "resources_Incremental": resources_Incremental,"new_categories": new_categories,
             "model_map": "model_map", "model_md5": "model_md5", "model_precision": "model_precision"}
    datas = json.dumps(datas, ensure_ascii=False, indent=4)
    with open(model_information, "w") as f:
        f.write(datas + "\n")
        f.close()