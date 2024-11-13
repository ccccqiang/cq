from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

CONFIG_FILE = 'configs.txt'

# 读取配置文件
def load_config():
    configs = {}
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.strip()
                if '=' in line and not line.startswith('#'):  # 只处理有等号且不以#开头的行
                    key, value = line.split('=', 1)  # 仅分割第一个等号
                    configs[key.strip()] = value.split('#')[0].strip()  # 去掉注释及空格
                elif line.startswith('#'):
                    # 保留注释并处理注释行
                    configs[f"comment_{i}"] = line.strip()  # 用一个独特的键来存储注释
    except FileNotFoundError:
        print(f"配置文件 {CONFIG_FILE} 未找到，使用默认配置。")
    except Exception as e:
        print(f"读取配置文件时发生错误: {e}")
    return configs

# 执行文件写入
def write_config(configs):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            index = 1
            for key, value in configs.items():
                if key.startswith('comment_'):
                    # 写入注释
                    f.write(f"{value}\n")
                else:
                    # 写入配置项
                    f.write(f"{key} = {value}               # {configs.get(f'comment_{index}', '')}\n")
                    index += 1
            f.write("\n")  # 写入空行，保持文件格式整洁
    except Exception as e:
        print(f"写入配置文件时发生错误: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取表单数据并进行简单校验
        configs = {key: request.form[key] for key in request.form if key and request.form[key]}
        if configs:  # 如果表单数据有效
            write_config(configs)  # 写入到配置文件
            return redirect(url_for('index'))  # 重定向到主页以显示更新后的数据
        else:
            return "无效的配置数据", 400

    configs = load_config()  # 读取配置文件
    return render_template('index.html', configs=configs)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
