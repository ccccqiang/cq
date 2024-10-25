import os
def decrypt(content, password):
    half_length = (len(content) + 1) // 2
    first_half = content[:half_length]
    second_half = content[half_length:]

    # 使用密码进行解密
    decrypted_first_half = ''.join(chr(ord(c) ^ ord(password[i % len(password)])) for i, c in enumerate(first_half))

    return decrypted_first_half + second_half

# 输入加密的 Lua 文件路径
encrypted_file_path = r"C:\Users\home123\cq\tool\encrypted_file.lua"  # 替换为你的加密文件路径
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'decrypted_file.lua')

# 读取加密的 Lua 文件内容
with open(encrypted_file_path, 'r', encoding='utf-8') as file:
    content = file.read()

# 解密一半内容
password = '123'
decrypted_content = decrypt(content, password)

# 输出到桌面
with open(desktop_path, 'w', encoding='utf-8') as file:
    file.write(decrypted_content)

print(f"解密后的文件已保存到桌面: {desktop_path}")
