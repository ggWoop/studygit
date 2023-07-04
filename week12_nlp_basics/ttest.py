import csv
import os

def merge_temp_files(temp_files, output_file):
    with open(output_file, 'w', encoding='utf-8-sig',newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["url", "datetime_str", "title", "content", "title_tokens", "content_tokens"])

        for temp_file in temp_files:
            with open(temp_file, 'r', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                next(reader)  # 헤더 라인은 건너뜁니다.
                for row in reader:
                    writer.writerow(row)

            # 임시 파일 삭제
           # os.remove(temp_file)

# 사용 예시
temp_files = ['./data/tmp_2644.csv', './data/tmp_6668.csv', './data/tmp_18360.csv', './data/tmp_25876.csv']
output_file = './data/baseball_tokenized_1.csv'

merge_temp_files(temp_files, output_file)