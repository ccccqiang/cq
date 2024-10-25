import os
def decrypt(content, password):
    half_length = (len(content) + 1) // 2
    first_half = content[:half_length]
    second_half = content[half_length:]


    decrypted_first_half = ''.join(chr(ord(c) ^ ord(password[i % len(password)])) for i, c in enumerate(first_half))

    return decrypted_first_half + second_half


encrypted_file_path = r"C:\Users\home123\cq\tool\encrypted_file.lua"
desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'decrypted_file.lua')


with open(encrypted_file_path, 'r', encoding='utf-8') as file:
    content = file.read()


password = '123'
decrypted_content = decrypt(content, password)


with open(desktop_path, 'w', encoding='utf-8') as file:
    file.write(decrypted_content)

print(f"desktop_path: {desktop_path}")
