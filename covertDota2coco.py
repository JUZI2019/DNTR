import json
import os
import glob
import cv2

def dota_to_coco(dota_path, output_path):
    # COCO 格式的 JSON 结构
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "ship", "supercategory": "object"}]
    }

    annotation_id = 1  # 用于给每个标注分配唯一ID

    # 处理每个 DOTA 标注文件
    for img_id, txt_file in enumerate(glob.glob(os.path.join(dota_path, "*.txt"))):
        # 获取对应的图像文件路径并读取图像信息
        image_path = txt_file.replace("labelTxt", "images").replace(".txt", ".jpg")
        image = cv2.imread(image_path)
        if image is None:
            continue
        height, width = image.shape[:2]

        # 添加图像信息到 coco_data
        image_info = {
            "id": img_id,
            "file_name": os.path.basename(image_path),
            "width": width,
            "height": height
        }
        coco_data["images"].append(image_info)

        # 读取 DOTA 标注文件
        with open(txt_file, "r") as f:
            for line in f.readlines():
                data = line.strip().split()
                # 解析四点坐标
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, data[:8])

                # 将四点坐标转换为 COCO 格式的水平边界框
                x_coords = [x1, x2, x3, x4]
                y_coords = [y1, y2, y3, y4]
                xmin, ymin = min(x_coords), min(y_coords)
                xmax, ymax = max(x_coords), max(y_coords)
                bbox_width = xmax - xmin
                bbox_height = ymax - ymin
                area = bbox_width * bbox_height

                # 创建标注信息
                annotation = {
                    "id": annotation_id,
                    "image_id": img_id,
                    "category_id": 1,  # ship类别的ID
                    "bbox": [xmin, ymin, bbox_width, bbox_height],
                    "area": area,
                    "iscrowd": 0
                }
                coco_data["annotations"].append(annotation)
                annotation_id += 1

    # 保存为 COCO 格式的 JSON 文件
    with open(output_path, "w") as json_file:
        json.dump(coco_data, json_file, indent=4)
    print(f"COCO JSON file saved at {output_path}")

# 使用示例
dota_to_coco("/workstation/fyy/sen1ship_dota_vhbg_608_single_2/vh/train/labelTxt", "/workstation/fyy/sen1ship_dota_vhbg_608_single_2/vh/train/sen1ship_train.json")
dota_to_coco("/workstation/fyy/sen1ship_dota_vhbg_608_single_2/vh/test/labelTxt", "/workstation/fyy/sen1ship_dota_vhbg_608_single_2/vh/test/sen1ship_test.json")
dota_to_coco("/workstation/fyy/sen1ship_dota_vhbg_608_single_2/vh/inshore_test/labelTxt", "/workstation/fyy/sen1ship_dota_vhbg_608_single_2/vh/inshore_test/sen1ship_offshore_test.json")
dota_to_coco("/workstation/fyy/sen1ship_dota_vhbg_608_single_2/vh/offshore_test/labelTxt", "/workstation/fyy/sen1ship_dota_vhbg_608_single_2/vh/offshore_test/sen1ship_offshore_test.json")

