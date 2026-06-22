from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd


# --------------------------------------------------
# 1. 파일 위치 설정
# --------------------------------------------------
DATA_PATH = Path("data/raw/subway_2025.csv")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. CSV 파일 인코딩 확인 후 불러오기
# --------------------------------------------------
def read_csv_with_encoding(path: Path) -> tuple[pd.DataFrame, str]:
    """
    한국 공공데이터에서 자주 사용되는 인코딩을 순서대로 시도한다.
    읽기에 성공한 DataFrame과 인코딩 이름을 반환한다.
    """
    encodings = ["utf-8-sig", "cp949", "euc-kr"]

    last_error = None

    for encoding in encodings:
        try:
            dataframe = pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
            )
            return dataframe, encoding

        except UnicodeDecodeError as error:
            last_error = error

    raise RuntimeError(
        f"CSV 인코딩을 확인할 수 없습니다. 마지막 오류: {last_error}"
    )


# --------------------------------------------------
# 3. 컬럼 이름을 자동으로 찾는 함수
# --------------------------------------------------
def find_column(
    columns: list[str],
    keywords: list[str],
) -> str | None:
    """
    컬럼 이름에 특정 단어가 들어 있는지 확인한다.
    가장 먼저 발견된 컬럼 이름을 반환한다.
    """
    for column in columns:
        for keyword in keywords:
            if keyword in column:
                return column

    return None


# --------------------------------------------------
# 4. 입력 파일 존재 여부 확인
# --------------------------------------------------
if not DATA_PATH.exists():
    raise FileNotFoundError(
        "\n데이터 파일을 찾을 수 없습니다.\n"
        "다음 위치에 파일이 있는지 확인하세요.\n"
        f"{DATA_PATH.resolve()}\n"
    )


# --------------------------------------------------
# 5. 데이터 불러오기
# --------------------------------------------------
df, detected_encoding = read_csv_with_encoding(DATA_PATH)

# 컬럼명 앞뒤 공백 제거
df.columns = [str(column).strip() for column in df.columns]

print("=" * 70)
print("1. 파일 불러오기 성공")
print("=" * 70)
print(f"파일 위치: {DATA_PATH.resolve()}")
print(f"사용한 인코딩: {detected_encoding}")
print(f"행 개수: {len(df):,}")
print(f"열 개수: {len(df.columns):,}")


# --------------------------------------------------
# 6. 컬럼 확인
# --------------------------------------------------
print("\n" + "=" * 70)
print("2. 컬럼 목록")
print("=" * 70)

for index, column in enumerate(df.columns, start=1):
    print(f"{index:02d}. {column}")


# --------------------------------------------------
# 7. 처음 5개 행 확인
# --------------------------------------------------
print("\n" + "=" * 70)
print("3. 처음 5개 행")
print("=" * 70)
print(df.head())


# --------------------------------------------------
# 8. 주요 컬럼 자동 탐색
# --------------------------------------------------
columns = df.columns.tolist()

date_column = find_column(
    columns,
    ["날짜", "수송일자", "사용일자", "일자"],
)

line_column = find_column(
    columns,
    ["호선"],
)

station_column = find_column(
    columns,
    ["역명", "역사명"],
)

direction_column = find_column(
    columns,
    ["승하차구분", "승하차", "구분"],
)

print("\n" + "=" * 70)
print("4. 자동 탐색한 주요 컬럼")
print("=" * 70)
print(f"날짜 컬럼: {date_column}")
print(f"호선 컬럼: {line_column}")
print(f"역명 컬럼: {station_column}")
print(f"승하차 구분 컬럼: {direction_column}")


if date_column is None:
    raise ValueError(
        "날짜 컬럼을 자동으로 찾지 못했습니다. "
        "위에 출력된 컬럼 목록을 확인하세요."
    )


# --------------------------------------------------
# 9. 시간대 컬럼 찾기
# --------------------------------------------------
time_columns = []

for column in columns:
    column_text = str(column)

    # 예: 06시 이전, 06-07시간대, 23시-24시, 24시 이후
    has_hour_text = bool(re.search(r"\d{1,2}\s*시", column_text))
    has_time_range = bool(
        re.search(r"\d{1,2}\s*[-~]\s*\d{1,2}", column_text)
    )

    if has_hour_text or has_time_range:
        time_columns.append(column)


print("\n" + "=" * 70)
print("5. 시간대 컬럼")
print("=" * 70)
print(f"시간대 컬럼 개수: {len(time_columns)}")

for column in time_columns:
    print(f"- {column}")


if not time_columns:
    raise ValueError(
        "시간대별 인원 컬럼을 찾지 못했습니다. "
        "출력된 전체 컬럼 목록을 확인하세요."
    )


# --------------------------------------------------
# 10. 시간대별 인원 값을 숫자로 변환
# --------------------------------------------------
for column in time_columns:
    df[column] = (
        df[column]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    ).fillna(0)


# 각 행의 시간대별 인원 합계
df["row_total"] = df[time_columns].sum(axis=1)


# --------------------------------------------------
# 11. 날짜 데이터 변환
# --------------------------------------------------
date_text = (
    df[date_column]
    .astype(str)
    .str.replace(r"\.0$", "", regex=True)
    .str.replace(r"[^0-9]", "", regex=True)
)

df["date"] = pd.to_datetime(
    date_text,
    format="%Y%m%d",
    errors="coerce",
)


invalid_date_count = df["date"].isna().sum()

print("\n" + "=" * 70)
print("6. 데이터 기본 점검")
print("=" * 70)
print(f"날짜 변환 실패 행 수: {invalid_date_count:,}")
print(f"결측값 전체 개수: {df.isna().sum().sum():,}")
print(f"완전 중복 행 수: {df.duplicated().sum():,}")


if station_column is not None:
    print(
        f"고유 역명 개수: "
        f"{df[station_column].nunique(dropna=True):,}"
    )

if line_column is not None:
    print(
        f"고유 호선 개수: "
        f"{df[line_column].nunique(dropna=True):,}"
    )

if direction_column is not None:
    print("승하차 구분 값:")
    print(df[direction_column].value_counts(dropna=False))


# --------------------------------------------------
# 12. 날짜별 전체 승하차량 계산
# --------------------------------------------------
valid_df = df.dropna(subset=["date"]).copy()

daily_total = (
    valid_df
    .groupby("date", as_index=False)["row_total"]
    .sum()
    .sort_values("date")
)


print("\n" + "=" * 70)
print("7. 날짜 범위")
print("=" * 70)
print(f"시작일: {daily_total['date'].min()}")
print(f"종료일: {daily_total['date'].max()}")

print("\n날짜별 전체 인원 처음 5개:")
print(daily_total.head())


# --------------------------------------------------
# 13. 분석 결과 CSV 저장
# --------------------------------------------------
daily_output_path = OUTPUT_DIR / "01_daily_total.csv"

daily_total.to_csv(
    daily_output_path,
    index=False,
    encoding="utf-8-sig",
)


# --------------------------------------------------
# 14. 날짜별 전체 승하차량 그래프 저장
# --------------------------------------------------
plt.figure(figsize=(14, 6))

plt.plot(
    daily_total["date"],
    daily_total["row_total"],
    linewidth=1,
)

plt.title("Daily Seoul Metro Passenger Volume")
plt.xlabel("Date")
plt.ylabel("Passenger Count")
plt.grid(alpha=0.3)
plt.tight_layout()

graph_output_path = OUTPUT_DIR / "01_daily_total.png"

plt.savefig(
    graph_output_path,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# --------------------------------------------------
# 15. 실행 결과 안내
# --------------------------------------------------
print("\n" + "=" * 70)
print("8. 작업 완료")
print("=" * 70)
print(f"일별 집계 CSV: {daily_output_path.resolve()}")
print(f"일별 그래프: {graph_output_path.resolve()}")
print("\n첫 번째 데이터 점검이 완료되었습니다.")