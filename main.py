import argparse
import shutil
import sys
import time
from loguru import logger

from valuation import *
from train import run
from model_packge import *

def parse_ope(known = False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", action="store_true", help="train new model"
    )
    parser.add_argument(
        "--epoch",type=str, default= "", help="train new model"
    )
    parser.add_argument(
        "--copy_val_data", action="store_true", help="copy new data to train"
    )
    parser.add_argument(
        "--get_ts_model", action="store_true", help="switch type of model pt to ts"
    )
    parser.add_argument(
        "--use_model_id", type=str,default= "exp2", help="The location of the trained model"
    )
    parser.add_argument(
        "--valuation", action="store_true", help="valuation camera images precision"
    )
    parser.add_argument(
        "--version", type=str, default=None, help="data version name"
    )
    parser.add_argument(
        "--icon_category_folder", type=str, default="../../scc/dataset/data/dataset/latest/failed_images", help="package model information for model_info.js"
    )
    parser.add_argument(
        "--package", action="store_true", help="package model information for model_info.js"
    )
    parser.add_argument(
        "--yaml_path", type=str, default=ROOT / "data/detect_config.yaml", help="initial weights path"
    )
    parser.add_argument(
        "--copy_release_package", action="store_true", help="package model information for model_info.js"
    )
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def main():
    start_time = time.time()
    opt = parse_ope(True)
    yaml_path = opt.yaml_path
    config = get_yaml(yaml_path)
    if opt.train:
        if opt.epoch:
            if os.path.exists("runs"):
                shutil.rmtree("runs")
                print("folder had delete")
            os.system(f"python train.py --epochs {opt.epoch} --noval --version {sys.argv[4]}")
            os.system(
                f"python train.py --weights runs/train/exp/weights/best.pt --hyp data/hyps/hyp.finetune.yaml --epochs {opt.epoch} --noval --version {sys.argv[4]}")
        else:
            if os.path.exists("runs"):
                shutil.rmtree("runs")
                print("folder had delete")
            os.system(f"python train.py --epochs 300 --noval --version {sys.argv[2]}")
            os.system(
                f"python train.py --weights runs/train/exp/weights/best.pt --hyp data/hyps/hyp.finetune.yaml --epochs 200 --version {sys.argv[2]}")
    if opt.get_ts_model:
        switch(config["export_model"], opt.use_model_id)
    if opt.valuation:
        val(config, opt.version, opt.icon_category_folder)
    if opt.package:
        get_package(config, opt.version, opt.use_model_id)
    if opt.copy_release_package:
        val_data = ValDatasetEngine()
        val_data.get_val_data(config, opt.version)
        switch(config["export_model"], opt.use_model_id)
        val(config, opt.version, opt.icon_category_folder)
        show_failed_images(config, opt.version, opt.icon_category_folder)
        get_package(config, opt.version, opt.use_model_id)
        copy_and_delete(config,opt.version)
    end_time = time.time()
    logger.info("The total time is %s min" % (str(round((end_time - start_time) / 60, 3))))



if __name__ == "__main__":
    main()

