import os

def encrypt(content, password):
    half_length = (len(content) + 1) // 2
    first_half = content[:half_length]
    second_half = content[half_length:]


    encrypted_first_half = ''.join(chr(ord(c) ^ ord(password[i % len(password)])) for i, c in enumerate(first_half))

    return encrypted_first_half + second_half


lua_file_path = r'C:\Users\home123\cq\tool\cq.lua'
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'encrypted_file.lua')


with open(lua_file_path, 'r', encoding='utf-8') as file:
    content = file.read()


password = '123'
encrypted_content = encrypt(content, password)


with open(desktop_path, 'w', encoding='utf-8') as file:
    file.write(encrypted_content)

print(f"desktop_path: {desktop_path}")
