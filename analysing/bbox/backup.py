def rename_files(directory):
    temp_suffix = ".temp"  # 임시 이름에 붙일 접미사

    # 1단계: 임시 이름으로 변경
    for file_name in os.listdir(directory):
        if file_name.endswith(".txt"):
            try:
                base_name, ext = os.path.splitext(file_name)
                temp_name = f"{base_name}{temp_suffix}{ext}"  # 임시 이름 생성
                os.rename(
                    os.path.join(directory, file_name),
                    os.path.join(directory, temp_name)
                )
            except ValueError:
                print(f"Skipping file with non-numeric name: {file_name}")