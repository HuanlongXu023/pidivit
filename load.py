import torch
import os

# 文件路径
file_path = "/root/data1/xfr/PiDiViT/animal_novel_10shot.vitl14.bbox.p10.sk.pkl"
#file_path = "/root/data1/xfr/PiDiViT/animal_base_train.vitl14.bbox.p10.sk.pkl"


print("=" * 80)
print(f"检查文件: {file_path}")
print("=" * 80)

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"✗ 文件不存在: {file_path}")
    exit(1)

file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
print(f"✓ 文件存在，大小: {file_size:.2f} MB\n")

# 加载文件
try:
    data = torch.load(file_path, map_location='cpu')
    print(f"✓ 文件加载成功")
    print(f"数据类型: {type(data)}\n")
except Exception as e:
    print(f"✗ 加载失败: {e}")
    exit(1)

# 如果是字典，显示详细信息
if isinstance(data, dict):
    print("=" * 80)
    print("字典键和内容概览:")
    print("=" * 80)

    for key in data.keys():
        value = data[key]
        print(f"\n键: '{key}'")
        print(f"  类型: {type(value)}")

        if isinstance(value, list):
            print(f"  长度: {len(value)}")
            if len(value) > 0:
                print(f"  第一个元素类型: {type(value[0])}")
                if isinstance(value[0], torch.Tensor):
                    print(f"  第一个元素shape: {value[0].shape}")
                    print(f"  第一个元素示例: {value[0][:5] if value[0].numel() > 5 else value[0]}")
                else:
                    print(f"  前几个元素: {value[:5]}")

        elif isinstance(value, torch.Tensor):
            print(f"  Shape: {value.shape}")
            print(f"  Dtype: {value.dtype}")
            print(f"  Device: {value.device}")
            print(f"  示例值: {value.flatten()[:10]}")

        elif isinstance(value, (int, float, str)):
            print(f"  值: {value}")

        else:
            print(f"  内容: {value}")

    print("\n" + "=" * 80)
    print("关键检查:")
    print("=" * 80)

    # 检查是否有 prototypes 键
    if 'prototypes' in data:
        print("✓ 找到 'prototypes' 键")
        print(f"  Shape: {data['prototypes'].shape}")
    else:
        print("✗ 没有找到 'prototypes' 键")
        print("  可用的键:", list(data.keys()))

    # 检查 labels
    if 'labels' in data:
        labels = data['labels']
        print(f"\n✓ 找到 'labels' 键")
        print(f"  数量: {len(labels)}")
        if len(labels) > 0:
            print(f"  唯一类别数: {len(set(labels))}")
            print(f"  类别范围: {min(labels)} ~ {max(labels)}")
            print(f"  前10个标签: {labels[:10]}")

    # 检查 avg_patch_tokens
    if 'avg_patch_tokens' in data:
        tokens = data['avg_patch_tokens']
        print(f"\n✓ 找到 'avg_patch_tokens' 键")
        print(f"  数量: {len(tokens)}")
        if len(tokens) > 0:
            print(f"  第一个token shape: {tokens[0].shape}")
            print(f"  可以转换为 prototypes: torch.stack(avg_patch_tokens)")

    # 检查 patch_tokens (for stuff/background)
    if 'patch_tokens' in data:
        tokens = data['patch_tokens']
        print(f"\n✓ 找到 'patch_tokens' 键")
        print(f"  数量: {len(tokens)}")
        if len(tokens) > 0:
            print(f"  第一个token shape: {tokens[0].shape}")

else:
    print(f"数据不是字典类型，而是: {type(data)}")
    print(f"内容: {data}")

print("\n" + "=" * 80)
print("检查完成")
print("=" * 80)