import os

def save_code_to_file(filename=None):
    if filename is None:
        # 현재 스크립트의 파일명을 가져와서 확장자를 txt로 변경
        filename = os.path.splitext(os.path.basename(__file__))[0] + ".txt"
    else:
        filename = filename + ".txt"
    with open(__file__, "r", encoding="utf-8") as file:
        code = file.read()
    
    with open(filename, "w", encoding="utf-8") as file:
        file.write(code)


print("잘된다")

save_code_to_file()