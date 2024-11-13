from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

CONFIG_FILE = 'configs.txt'

# 读取配置文件
def load_config():
    configs = {}
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if '=' in line and not line.startswith('#'):  # 只处理有等号且不以#开头的行
                key, value = line.split('=', 1)  # 仅分割第一个等号
                configs[key.strip()] = value.split('#')[0].strip()  # 去掉注释及空格
    return configs

# 执行文件写入
def write_config(configs):
    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        for key, value in configs.items():
            f.write(f"{key} = {value}\n")  # 写入配置文件
        f.write("\n")  # 写入空行，保持文件格式整洁

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取表单数据
        configs = {key: request.form[key] for key in request.form}
        write_config(configs)  # 写入到配置文件
        return redirect(url_for('index'))  # 重定向到主页以显示更新后的数据

    configs = load_config()  # 读取配置文件
    return render_template('index.html', configs=configs)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
