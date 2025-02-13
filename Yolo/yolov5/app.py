import socket
from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

CONFIG_FILE = 'config_double.txt'

# 读取配置文件
def load_config():
    configs = []
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                line = line.rstrip()  # 去掉行尾的换行符，但保留其他空白字符
                if '=' in line and not line.startswith('#'):  # 处理有等号且不以#开头的行
                    key, value = line.split('=', 1)  # 仅分割第一个等号
                    configs.append((key.strip(), value.split('#')[0].strip()))  # 去掉注释及空格
                else:
                    # 保留注释和空行
                    configs.append((f"line_{i}", line))
    except FileNotFoundError:
        print(f"配置文件 {CONFIG_FILE} 未找到，使用默认配置。")
    except Exception as e:
        print(f"读取配置文件时发生错误: {e}")
    return configs

# 执行文件写入
def write_config(configs):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            for key, value in configs:
                if key.startswith('line_'):
                    # 写入注释或空行
                    f.write(f"{value}\n")
                else:
                    # 写入配置项
                    f.write(f"{key} = {value}\n")
    except Exception as e:
        print(f"写入配置文件时发生错误: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 获取表单数据并进行简单校验
        new_configs = {key: request.form[key] for key in request.form if key and request.form[key]}
        if new_configs:  # 如果表单数据有效
            # 读取当前配置文件内容
            current_configs = load_config()
            # 更新配置文件内容
            updated_configs = []
            for key, value in current_configs:
                if key in new_configs:
                    updated_configs.append((key, new_configs.pop(key)))
                else:
                    updated_configs.append((key, value))
            # 添加新的配置项
            for key, value in new_configs.items():
                updated_configs.append((key, value))
            write_config(updated_configs)  # 写入到配置文件
            return redirect(url_for('index'))  # 重定向到主页以显示更新后的数据
        else:
            return "无效的配置数据", 400

    configs = load_config()  # 读取配置文件
    return render_template('index.html', configs=configs)

def get_local_ip():
    # 获取本机的局域网 IP 地址
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # 尝试连接到外部服务器（如 Google DNS），然后获取本机 IP 地址
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
    except Exception:
        local_ip = '127.0.0.1'  # 如果无法获取外部 IP，则默认使用本地回环地址
    finally:
        s.close()
    return local_ip

if __name__ == "__main__":
    # 获取局域网 IP 地址
    local_ip = get_local_ip()
    print(f"正在使用本地 IP 地址 {local_ip} 进行服务")
    app.run(host=local_ip, port=5000)
