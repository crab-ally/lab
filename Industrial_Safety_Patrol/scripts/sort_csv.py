import pandas as pd
from charset_normalizer import from_path
from pathlib import Path

file_path = Path("../docs/Accident_20241231.csv")
output_path = Path("../docs/check.csv")

# 파일 인코딩 방식 확인
encoding = from_path(file_path).best()
print(f"enconding: {encoding.encoding}")
print()

if encoding.encoding=="cp949":
    df = pd.read_csv(file_path, encoding="cp949")
else:
    df = pd.read_csv(file_path)

# 특정 행 필터 (인덱스 초기화)
filtered = df.iloc[31:140].copy().reset_index(drop=True)

# 확인용 csv 저장
text_cols = filtered.select_dtypes(include="object").columns
num_cols = filtered.select_dtypes(exclude="object").columns

col_sum = filtered[num_cols].sum()

new_row = {}
new_row[text_cols[0]] = "TOTAL"   # 첫 번째 문자열 컬럼만 TOTAL
new_row.update(col_sum.to_dict()) # 숫자 컬럼은 합계

filtered.loc[len(filtered)] = new_row

try:
    if encoding.encoding=="cp949":
        filtered.to_csv(output_path, index=False, encoding="cp949")
    else:
        filtered.to_csv(output_path, index=False)
    print("CSV 저장 성공")
    print()

except Exception as e:
    print("CSV 저장 실패:", e)
    print()

# 내림차순 정렬
result = col_sum.sort_values(ascending=False)

print(result)