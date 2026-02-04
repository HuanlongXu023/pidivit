#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析COCO格式JSON文件中有标注和无标注的类别
"""

import json
import sys


def analyze_annotations(json_file):
    # 读取JSON文件
    print(f"正在读取文件: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 获取所有定义的类别
    all_categories = {}
    for category in data['categories']:
        all_categories[category['id']] = category['name']

    # 获取所有标注中出现的类别ID
    annotated_category_ids = set()
    for annotation in data['annotations']:
        annotated_category_ids.add(annotation['category_id'])

    # 分离有标注和无标注的类别
    annotated_categories = {}
    unannotated_categories = {}

    for cat_id, cat_name in all_categories.items():
        if cat_id in annotated_category_ids:
            annotated_categories[cat_id] = cat_name
        else:
            unannotated_categories[cat_id] = cat_name

    # 打印结果
    print("\n" + "=" * 60)
    print("分析结果")
    print("=" * 60)

    print(f"\n总类别数: {len(all_categories)}")
    print(f"有标注的类别数: {len(annotated_categories)}")
    print(f"无标注的类别数: {len(unannotated_categories)}")
    print(f"总标注数: {len(data['annotations'])}")

    print("\n" + "=" * 60)
    print("有标注的类别:")
    print("=" * 60)
    for cat_id in sorted(annotated_categories.keys()):
        cat_name = annotated_categories[cat_id]
        # 统计该类别的标注数量
        count = sum(1 for ann in data['annotations'] if ann['category_id'] == cat_id)
        print(f"ID: {cat_id:3d} | 名称: {cat_name:20s} | 标注数: {count:5d}")

    if unannotated_categories:
        print("\n" + "=" * 60)
        print("无标注的类别:")
        print("=" * 60)
        for cat_id in sorted(unannotated_categories.keys()):
            cat_name = unannotated_categories[cat_id]
            print(f"ID: {cat_id:3d} | 名称: {cat_name}")
    else:
        print("\n所有类别都有标注!")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    json_file = "/root/data1/xfr/coco14_10shot_filtered.json"
    #json_file = "/root/data1/xfr/CDFSOD/datasets/coco/annotations/fs_coco14_base_train.json"
    # 如果命令行提供了参数,使用命令行参数
    if len(sys.argv) > 1:
        json_file = sys.argv[1]

    try:
        analyze_annotations(json_file)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {json_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误: {json_file} 不是有效的JSON文件")
        sys.exit(1)
    except KeyError as e:
        print(f"错误: JSON文件缺少必要的字段 {e}")
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {e}")
        sys.exit(1)


