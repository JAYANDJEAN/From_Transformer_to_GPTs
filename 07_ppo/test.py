import yaml

# 读取 YAML 文件
with open('../00_assets/yml/config.yml', 'r') as file:
    config = yaml.safe_load(file)

# 获取科学计数法表示的数值
small_number = config['settings']['small_number']
large_number = float(config['settings']['large_number'])
very_large_number = config['settings']['very_large_number']
floating_point_number = config['settings']['floating_point_number']
another_large_number = config['settings']['another_large_number']

# 确保数值是 float 类型
print(f"Small number: {small_number} (type: {type(small_number)})")
print(f"Large number: {large_number} (type: {type(large_number)})")
print(f"Very large number: {very_large_number} (type: {type(very_large_number)})")
print(f"Floating point number: {floating_point_number} (type: {type(floating_point_number)})")
print(f"Another large number: {another_large_number} (type: {type(another_large_number)})")