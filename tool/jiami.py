import os

def encrypt(content, password):
    half_length = (len(content) + 1) // 2
    first_half = content[:half_length]
    second_half = content[half_length:]

    # 使用密码进行加密
    encrypted_first_half = ''.join(chr(ord(c) ^ ord(password[i % len(password)])) for i, c in enumerate(first_half))

    return encrypted_first_half + second_half

# 输入 Lua 文件路径
lua_file_path = r'C:\Users\home123\cq\tool\cq.lua'  # 替换为你的 Lua 文件路径
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'encrypted_file.lua')

# 读取 Lua 文件内容
with open(lua_file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# 加密一半内容
password = '123'
encrypted_content = encrypt(content, password)

# 输出到桌面
with open(desktop_path, 'w', encoding='utf-8') as file:
    file.write(encrypted_content)

print(f"加密后的文件已保存到桌面: {desktop_path}")
