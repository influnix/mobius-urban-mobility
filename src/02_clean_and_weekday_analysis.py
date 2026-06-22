from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. 파일과 폴더 위치 설정
# ============================================================

RAW_DATA_PATH = Path("data/raw/subway_2025.csv")
PROCESSED_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 이번 데이터에서 사용할 컬럼 정의
# ============================================================

ID_COLUMNS = [
    "수송일자",
    "호선",
    "역번호",
    "역명",
    "승하차구분",
]

TIME_COLUMNS = [
    "06시이전",
    "06-07시간대",
    "07-08시간대",
    "08-09시간대",
    "09-10시간대",
    "10-11시간대",
    "11-12시간대",
    "12-13시간대",
    "13-14시간대",
    "14-15시간대",
    "15-16시간대",
    "16-17시간대",
    "17-18시간대",
    "18-19시간대",
    "19-20시간대",
    "20-21시간대",
    "21-22시간대",
    "22-23시간대",
    "23-24시간대",
    "24시이후",
]

# 시간대의 순서를 숫자로 표현한다.
# 이후 그래프를 그릴 때 06시 이전부터 24시 이후까지
# 올바른 순서로 정렬하기 위해 사용한다.
TIME_ORDER = {
    time_column: order
    for order, time_column in enumerate(TIME_COLUMNS)
}


# ============================================================
# 3. 원본 파일 존재 여부 확인
# ============================================================

if not RAW_DATA_PATH.exists():
    raise FileNotFoundError(
        "\n원본 데이터 파일을 찾지 못했습니다.\n"
        f"확인할 위치: {RAW_DATA_PATH.resolve()}\n"
    )


# ============================================================
# 4. 원본 CSV 불러오기
# ============================================================

df = pd.read_csv(
    RAW_DATA_PATH,
    encoding="cp949",
    low_memory=False,
)

# 컬럼 이름 앞뒤에 불필요한 공백이 있을 경우 제거
df.columns = df.columns.str.strip()

print("=" * 70)
print("1. 원본 데이터")
print("=" * 70)
print(f"원본 행 수: {len(df):,}")
print(f"원본 열 수: {len(df.columns):,}")


# ============================================================
# 5. 필요한 컬럼이 모두 있는지 확인
# ============================================================

expected_columns = ID_COLUMNS + TIME_COLUMNS

missing_columns = [
    column
    for column in expected_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "\n필요한 컬럼이 없습니다.\n"
        f"누락된 컬럼: {missing_columns}\n"
    )

print("\n필요한 컬럼이 모두 존재합니다.")


# ============================================================
# 6. 날짜 컬럼 변환
# ============================================================

# 원본 날짜를 문자열로 변환하고 앞뒤 공백 제거
date_text = (
    df["수송일자"]
    .astype("string")
    .str.strip()
)

# 정상적인 날짜는 datetime 형태로 변환되고,
# 날짜로 바꿀 수 없는 값은 NaT가 된다.
df["date"] = pd.to_datetime(
    date_text,
    errors="coerce",
)


# ============================================================
# 7. 유효하지 않은 행 찾기
# ============================================================

# 날짜가 없거나, 노선·역·승하차 정보가 없거나,
# 승하차 값이 승차/하차가 아닌 행을 찾는다.
invalid_mask = (
    df["date"].isna()
    | df["호선"].isna()
    | df["역번호"].isna()
    | df["역명"].isna()
    | ~df["승하차구분"].isin(["승차", "하차"])
)

invalid_rows = df.loc[invalid_mask].copy()

INVALID_OUTPUT_PATH = OUTPUT_DIR / "02_invalid_rows.csv"

invalid_rows.to_csv(
    INVALID_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 70)
print("2. 유효하지 않은 행 확인")
print("=" * 70)
print(f"유효하지 않은 행 수: {len(invalid_rows):,}")
print(f"확인용 파일: {INVALID_OUTPUT_PATH.resolve()}")


# ============================================================
# 8. 유효한 행만 남기기
# ============================================================

clean_df = df.loc[~invalid_mask].copy()

print(f"유효한 행 수: {len(clean_df):,}")


# ============================================================
# 9. 완전히 동일한 중복 행 확인
# ============================================================

exact_duplicate_count = clean_df.duplicated().sum()

print("\n" + "=" * 70)
print("3. 중복 행 확인")
print("=" * 70)
print(f"정제 후 완전 중복 행 수: {exact_duplicate_count:,}")

# 완전히 동일한 행만 제거한다.
clean_df = clean_df.drop_duplicates().copy()


# ============================================================
# 10. 분석 단위 기준 중복 확인
# ============================================================

# 해당 데이터에서 하나의 행을 구분하는 핵심 키
BUSINESS_KEY = [
    "date",
    "호선",
    "역번호",
    "역명",
    "승하차구분",
]

business_duplicate_mask = clean_df.duplicated(
    subset=BUSINESS_KEY,
    keep=False,
)

business_duplicate_rows = clean_df.loc[
    business_duplicate_mask
].copy()

BUSINESS_DUPLICATE_PATH = (
    OUTPUT_DIR / "02_business_key_duplicates.csv"
)

business_duplicate_rows.to_csv(
    BUSINESS_DUPLICATE_PATH,
    index=False,
    encoding="utf-8-sig",
)

print(
    "분석 단위 기준 중복 행 수: "
    f"{len(business_duplicate_rows):,}"
)

# 분석 단위 중복은 의미 있는 중복일 가능성이 있으므로
# 자동으로 삭제하지 않고 실행을 중단한다.
if not business_duplicate_rows.empty:
    raise ValueError(
        "\n분석 단위 기준 중복 행이 발견되었습니다.\n"
        "outputs/02_business_key_duplicates.csv를 확인하세요.\n"
        "이 중복은 임의로 삭제하면 안 됩니다.\n"
    )


# ============================================================
# 11. 시간대별 인원을 숫자로 변환
# ============================================================

for column in TIME_COLUMNS:
    clean_df[column] = (
        clean_df[column]
        .astype("string")
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    clean_df[column] = pd.to_numeric(
        clean_df[column],
        errors="coerce",
    ).fillna(0)

    clean_df[column] = clean_df[column].astype("int32")


# ============================================================
# 12. Wide Format을 Long Format으로 변환
# ============================================================

# 변환 전 구조
#
# 날짜 | 역명 | 승하차구분 | 06시이전 | 06-07시간대 | ...
#
# 변환 후 구조
#
# 날짜 | 역명 | 승하차구분 | time_slot | passenger_count

long_df = clean_df.melt(
    id_vars=[
        "date",
        "호선",
        "역번호",
        "역명",
        "승하차구분",
    ],
    value_vars=TIME_COLUMNS,
    var_name="time_slot",
    value_name="passenger_count",
)

# 영문 컬럼명으로 변경
long_df = long_df.rename(
    columns={
        "호선": "line",
        "역번호": "station_id",
        "역명": "station_name",
        "승하차구분": "direction",
    }
)

# 시간대 정렬용 숫자 추가
long_df["time_order"] = (
    long_df["time_slot"]
    .map(TIME_ORDER)
    .astype("int8")
)

# 승객 수를 정수형으로 지정
long_df["passenger_count"] = (
    long_df["passenger_count"]
    .astype("int32")
)

# 보기 좋은 컬럼 순서로 재배치
long_df = long_df[
    [
        "date",
        "line",
        "station_id",
        "station_name",
        "direction",
        "time_slot",
        "time_order",
        "passenger_count",
    ]
]


# ============================================================
# 13. Long Format 데이터 검증
# ============================================================

expected_long_rows = len(clean_df) * len(TIME_COLUMNS)
actual_long_rows = len(long_df)

print("\n" + "=" * 70)
print("4. Wide → Long 변환")
print("=" * 70)
print(f"변환 전 행 수: {len(clean_df):,}")
print(f"시간대 컬럼 수: {len(TIME_COLUMNS):,}")
print(f"예상 Long 행 수: {expected_long_rows:,}")
print(f"실제 Long 행 수: {actual_long_rows:,}")

if expected_long_rows != actual_long_rows:
    raise ValueError(
        "Wide → Long 변환 과정에서 행 수가 일치하지 않습니다."
    )

print("\nLong 데이터 처음 10개 행:")
print(long_df.head(10))


# ============================================================
# 14. Long Format 데이터를 Parquet으로 저장
# ============================================================

LONG_OUTPUT_PATH = (
    PROCESSED_DIR / "subway_2025_long.parquet"
)

long_df.to_parquet(
    LONG_OUTPUT_PATH,
    index=False,
)

print(f"\nLong 데이터 저장: {LONG_OUTPUT_PATH.resolve()}")


# ============================================================
# 15. 날짜별 전체 이용량 계산
# ============================================================

daily_total = (
    long_df
    .groupby("date", as_index=False)["passenger_count"]
    .sum()
    .sort_values("date")
)

# 0: 월요일, 1: 화요일, ..., 6: 일요일
daily_total["weekday_num"] = (
    daily_total["date"].dt.dayofweek
)

WEEKDAY_KOR = {
    0: "월",
    1: "화",
    2: "수",
    3: "목",
    4: "금",
    5: "토",
    6: "일",
}

WEEKDAY_ENG = {
    0: "Mon",
    1: "Tue",
    2: "Wed",
    3: "Thu",
    4: "Fri",
    5: "Sat",
    6: "Sun",
}

daily_total["weekday"] = (
    daily_total["weekday_num"]
    .map(WEEKDAY_KOR)
)

daily_total["weekday_eng"] = (
    daily_total["weekday_num"]
    .map(WEEKDAY_ENG)
)

daily_total["is_weekend"] = (
    daily_total["weekday_num"] >= 5
)


# ============================================================
# 16. 요일별 통계 계산
# ============================================================

weekday_summary = (
    daily_total
    .groupby(
        ["weekday_num", "weekday", "weekday_eng"],
        as_index=False,
    )["passenger_count"]
    .agg(
        average="mean",
        median="median",
        minimum="min",
        maximum="max",
        number_of_days="count",
    )
    .sort_values("weekday_num")
)

# 소수점 제거
for column in [
    "average",
    "median",
    "minimum",
    "maximum",
]:
    weekday_summary[column] = (
        weekday_summary[column]
        .round()
        .astype("int64")
    )


# ============================================================
# 17. 평일과 주말 평균 비교
# ============================================================

weekday_average = (
    daily_total.loc[
        ~daily_total["is_weekend"],
        "passenger_count",
    ]
    .mean()
)

weekend_average = (
    daily_total.loc[
        daily_total["is_weekend"],
        "passenger_count",
    ]
    .mean()
)

weekend_decrease_rate = (
    1 - weekend_average / weekday_average
) * 100


# ============================================================
# 18. 가장 이용량이 낮은 날짜와 높은 날짜
# ============================================================

lowest_days = (
    daily_total
    .nsmallest(10, "passenger_count")
    .copy()
)

highest_days = (
    daily_total
    .nlargest(10, "passenger_count")
    .copy()
)

lowest_days.to_csv(
    OUTPUT_DIR / "02_lowest_days.csv",
    index=False,
    encoding="utf-8-sig",
)

highest_days.to_csv(
    OUTPUT_DIR / "02_highest_days.csv",
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 19. 요일별 통계 CSV 저장
# ============================================================

WEEKDAY_SUMMARY_PATH = (
    OUTPUT_DIR / "02_weekday_summary.csv"
)

weekday_summary.to_csv(
    WEEKDAY_SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 20. 요일별 평균 그래프 생성
# ============================================================

plot_values = (
    weekday_summary["average"] / 1_000_000
)

plt.figure(figsize=(10, 6))

bars = plt.bar(
    weekday_summary["weekday_eng"],
    plot_values,
)

plt.title(
    "Average Daily Seoul Metro Passenger Volume by Weekday"
)
plt.xlabel("Weekday")
plt.ylabel("Average Passengers (Million)")
plt.ylim(0, plot_values.max() * 1.15)
plt.grid(axis="y", alpha=0.3)

# 막대 위에 실제 평균값 표시
for bar, value in zip(bars, plot_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{value:.2f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()

WEEKDAY_GRAPH_PATH = (
    OUTPUT_DIR / "02_weekday_average.png"
)

plt.savefig(
    WEEKDAY_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 21. 결과 출력
# ============================================================

print("\n" + "=" * 70)
print("5. 요일별 평균 이용량")
print("=" * 70)

print(
    weekday_summary[
        [
            "weekday",
            "average",
            "median",
            "minimum",
            "maximum",
            "number_of_days",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("6. 평일과 주말 비교")
print("=" * 70)
print(f"평일 일평균: {weekday_average:,.0f}")
print(f"주말 일평균: {weekend_average:,.0f}")
print(
    "주말의 평일 대비 감소율: "
    f"{weekend_decrease_rate:.1f}%"
)

print("\n가장 이용량이 낮은 날짜 10개:")
print(
    lowest_days[
        [
            "date",
            "weekday",
            "passenger_count",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("7. 작업 완료")
print("=" * 70)
print(f"정제 확인 파일: {INVALID_OUTPUT_PATH.resolve()}")
print(f"Long 데이터: {LONG_OUTPUT_PATH.resolve()}")
print(f"요일별 통계: {WEEKDAY_SUMMARY_PATH.resolve()}")
print(f"요일별 그래프: {WEEKDAY_GRAPH_PATH.resolve()}")