from datetime import date
from pathlib import Path

import holidays
import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. 파일 경로 설정
# ============================================================

INPUT_PATH = Path("outputs/01_daily_total.csv")
OUTPUT_DIR = Path("outputs")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 입력 파일 존재 여부 확인
# ============================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        "\n일별 집계 파일을 찾지 못했습니다.\n"
        f"확인할 위치: {INPUT_PATH.resolve()}\n"
        "먼저 01_data_audit.py를 실행해야 합니다.\n"
    )


# ============================================================
# 3. 일별 전체 이용량 불러오기
# ============================================================

daily_df = pd.read_csv(
    INPUT_PATH,
    encoding="utf-8-sig",
)

print("=" * 70)
print("1. 일별 이용량 파일 불러오기")
print("=" * 70)
print(f"파일 위치: {INPUT_PATH.resolve()}")
print(f"행 수: {len(daily_df):,}")
print(f"열 수: {len(daily_df.columns):,}")
print(f"컬럼: {daily_df.columns.tolist()}")


# ============================================================
# 4. 필요한 컬럼 확인
# ============================================================

required_columns = [
    "date",
    "row_total",
]

missing_columns = [
    column
    for column in required_columns
    if column not in daily_df.columns
]

if missing_columns:
    raise ValueError(
        "\n필요한 컬럼이 존재하지 않습니다.\n"
        f"누락 컬럼: {missing_columns}\n"
    )


# ============================================================
# 5. 날짜와 이용량 데이터 타입 변환
# ============================================================

daily_df["date"] = pd.to_datetime(
    daily_df["date"],
    errors="coerce",
)

daily_df["passenger_count"] = pd.to_numeric(
    daily_df["row_total"],
    errors="coerce",
)

invalid_mask = (
    daily_df["date"].isna()
    | daily_df["passenger_count"].isna()
)

invalid_count = invalid_mask.sum()

print("\n" + "=" * 70)
print("2. 데이터 타입 확인")
print("=" * 70)
print(f"날짜 또는 이용량 변환 실패 행 수: {invalid_count:,}")

if invalid_count > 0:
    invalid_rows = daily_df.loc[invalid_mask].copy()

    invalid_rows.to_csv(
        OUTPUT_DIR / "03_invalid_daily_rows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    raise ValueError(
        "날짜 또는 이용량을 변환하지 못한 행이 있습니다."
    )

daily_df = daily_df[
    [
        "date",
        "passenger_count",
    ]
].copy()


# ============================================================
# 6. 요일과 주말 여부 생성
# ============================================================

# 월요일=0, 화요일=1, ..., 일요일=6
daily_df["weekday_num"] = (
    daily_df["date"].dt.dayofweek
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

daily_df["weekday"] = (
    daily_df["weekday_num"]
    .map(WEEKDAY_KOR)
)

daily_df["is_weekend"] = (
    daily_df["weekday_num"] >= 5
)


# ============================================================
# 7. 2025년 대한민국 공휴일 생성
# ============================================================

kr_holidays = holidays.country_holidays(
    country="KR",
    years=[2025],
    language="ko",
)

# 일회성 임시공휴일은 사용 중인 라이브러리 버전에 따라
# 반영되지 않았을 가능성이 있으므로 명시적으로 추가한다.
MANUAL_HOLIDAYS = {
    date(2025, 1, 27): "임시공휴일",
    date(2025, 6, 3): "제21대 대통령선거일",
}

for holiday_date, holiday_name in MANUAL_HOLIDAYS.items():
    if holiday_date not in kr_holidays:
        kr_holidays[holiday_date] = holiday_name


# ============================================================
# 8. 일별 데이터에 공휴일 정보 추가
# ============================================================

daily_df["holiday_name"] = (
    daily_df["date"]
    .dt.date
    .map(kr_holidays.get)
)

daily_df["is_holiday"] = (
    daily_df["holiday_name"].notna()
)


# ============================================================
# 9. 날짜 유형 분류
# ============================================================

def classify_day_type(row: pd.Series) -> str:
    """
    하루를 일반 평일, 일반 주말, 공휴일 중 하나로 분류한다.

    공휴일이 토요일 또는 일요일에 발생한 경우에도
    공휴일로 먼저 분류한다.
    """
    if row["is_holiday"]:
        return "Public holiday"

    if row["is_weekend"]:
        return "Ordinary weekend"

    return "Ordinary weekday"


daily_df["day_type"] = daily_df.apply(
    classify_day_type,
    axis=1,
)

DAY_TYPE_ORDER = {
    "Ordinary weekday": 0,
    "Ordinary weekend": 1,
    "Public holiday": 2,
}

daily_df["day_type_order"] = (
    daily_df["day_type"]
    .map(DAY_TYPE_ORDER)
)


# ============================================================
# 10. 공휴일 목록 확인
# ============================================================

holiday_dates = (
    daily_df.loc[
        daily_df["is_holiday"],
        [
            "date",
            "weekday",
            "holiday_name",
            "passenger_count",
        ],
    ]
    .sort_values("date")
    .copy()
)

HOLIDAY_DATES_PATH = (
    OUTPUT_DIR / "03_holiday_dates.csv"
)

holiday_dates.to_csv(
    HOLIDAY_DATES_PATH,
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 70)
print("3. 2025년 공휴일 목록")
print("=" * 70)
print(f"공휴일 수: {len(holiday_dates):,}")

print(
    holiday_dates.to_string(
        index=False,
    )
)


# ============================================================
# 11. 날짜 유형별 통계 계산
# ============================================================

day_type_summary = (
    daily_df
    .groupby(
        [
            "day_type_order",
            "day_type",
        ],
        as_index=False,
    )["passenger_count"]
    .agg(
        number_of_days="count",
        average="mean",
        median="median",
        minimum="min",
        maximum="max",
        standard_deviation="std",
    )
    .sort_values("day_type_order")
)

numeric_summary_columns = [
    "average",
    "median",
    "minimum",
    "maximum",
    "standard_deviation",
]

for column in numeric_summary_columns:
    day_type_summary[column] = (
        day_type_summary[column]
        .round()
        .astype("int64")
    )


# ============================================================
# 12. 일반 평일, 일반 주말, 공휴일 평일 평균 계산
# ============================================================

ordinary_weekday_mask = (
    ~daily_df["is_weekend"]
    & ~daily_df["is_holiday"]
)

ordinary_weekend_mask = (
    daily_df["is_weekend"]
    & ~daily_df["is_holiday"]
)

holiday_weekday_mask = (
    ~daily_df["is_weekend"]
    & daily_df["is_holiday"]
)

ordinary_weekday_average = (
    daily_df.loc[
        ordinary_weekday_mask,
        "passenger_count",
    ]
    .mean()
)

ordinary_weekend_average = (
    daily_df.loc[
        ordinary_weekend_mask,
        "passenger_count",
    ]
    .mean()
)

holiday_weekday_average = (
    daily_df.loc[
        holiday_weekday_mask,
        "passenger_count",
    ]
    .mean()
)

weekend_decrease_rate = (
    1
    - ordinary_weekend_average
    / ordinary_weekday_average
) * 100

holiday_decrease_rate = (
    1
    - holiday_weekday_average
    / ordinary_weekday_average
) * 100


# ============================================================
# 13. 공휴일을 제외한 요일별 통계
# ============================================================

ordinary_weekdays = daily_df.loc[
    ordinary_weekday_mask
].copy()

ordinary_weekday_summary = (
    ordinary_weekdays
    .groupby(
        [
            "weekday_num",
            "weekday",
        ],
        as_index=False,
    )["passenger_count"]
    .agg(
        number_of_days="count",
        average="mean",
        median="median",
        minimum="min",
        maximum="max",
    )
    .sort_values("weekday_num")
)

for column in [
    "average",
    "median",
    "minimum",
    "maximum",
]:
    ordinary_weekday_summary[column] = (
        ordinary_weekday_summary[column]
        .round()
        .astype("int64")
    )


# ============================================================
# 14. 분석 결과 CSV 저장
# ============================================================

ENRICHED_DAILY_PATH = (
    OUTPUT_DIR / "03_daily_with_holidays.csv"
)

DAY_TYPE_SUMMARY_PATH = (
    OUTPUT_DIR / "03_day_type_summary.csv"
)

ORDINARY_WEEKDAY_PATH = (
    OUTPUT_DIR / "03_ordinary_weekday_summary.csv"
)

daily_df.to_csv(
    ENRICHED_DAILY_PATH,
    index=False,
    encoding="utf-8-sig",
)

day_type_summary.to_csv(
    DAY_TYPE_SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig",
)

ordinary_weekday_summary.to_csv(
    ORDINARY_WEEKDAY_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 15. 날짜 유형별 평균 막대그래프
# ============================================================

plot_summary = day_type_summary.copy()

plot_summary["average_million"] = (
    plot_summary["average"] / 1_000_000
)

plt.figure(figsize=(10, 6))

bars = plt.bar(
    plot_summary["day_type"],
    plot_summary["average_million"],
)

plt.title(
    "Average Seoul Metro Passenger Volume by Day Type"
)
plt.xlabel("Day Type")
plt.ylabel("Average Passengers (Million)")
plt.ylim(
    0,
    plot_summary["average_million"].max() * 1.2,
)
plt.grid(
    axis="y",
    alpha=0.3,
)

for bar, value in zip(
    bars,
    plot_summary["average_million"],
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{value:.2f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()

DAY_TYPE_GRAPH_PATH = (
    OUTPUT_DIR / "03_day_type_average.png"
)

plt.savefig(
    DAY_TYPE_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 16. 일별 이용량과 공휴일 위치 그래프
# ============================================================

plt.figure(figsize=(15, 7))

plt.plot(
    daily_df["date"],
    daily_df["passenger_count"],
    linewidth=1,
    label="Daily passenger volume",
)

plt.scatter(
    holiday_dates["date"],
    holiday_dates["passenger_count"],
    s=35,
    label="Public holiday",
    zorder=3,
)

plt.title(
    "Daily Seoul Metro Passenger Volume and Public Holidays"
)
plt.xlabel("Date")
plt.ylabel("Passenger Count")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

HOLIDAY_GRAPH_PATH = (
    OUTPUT_DIR / "03_daily_volume_with_holidays.png"
)

plt.savefig(
    HOLIDAY_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 17. 결과 출력
# ============================================================

print("\n" + "=" * 70)
print("4. 날짜 유형별 통계")
print("=" * 70)

print(
    day_type_summary[
        [
            "day_type",
            "number_of_days",
            "average",
            "median",
            "minimum",
            "maximum",
            "standard_deviation",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("5. 일반 평일·주말·공휴일 평일 비교")
print("=" * 70)
print(
    "일반 평일 일평균: "
    f"{ordinary_weekday_average:,.0f}"
)
print(
    "일반 주말 일평균: "
    f"{ordinary_weekend_average:,.0f}"
)
print(
    "공휴일인 평일 일평균: "
    f"{holiday_weekday_average:,.0f}"
)
print(
    "일반 주말의 일반 평일 대비 감소율: "
    f"{weekend_decrease_rate:.1f}%"
)
print(
    "공휴일 평일의 일반 평일 대비 감소율: "
    f"{holiday_decrease_rate:.1f}%"
)

print("\n" + "=" * 70)
print("6. 공휴일을 제외한 평일 요일별 통계")
print("=" * 70)

print(
    ordinary_weekday_summary[
        [
            "weekday",
            "number_of_days",
            "average",
            "median",
            "minimum",
            "maximum",
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("7. 작업 완료")
print("=" * 70)
print(
    f"공휴일 결합 일별 데이터: "
    f"{ENRICHED_DAILY_PATH.resolve()}"
)
print(
    f"공휴일 목록: "
    f"{HOLIDAY_DATES_PATH.resolve()}"
)
print(
    f"날짜 유형별 통계: "
    f"{DAY_TYPE_SUMMARY_PATH.resolve()}"
)
print(
    f"일반 평일 요일 통계: "
    f"{ORDINARY_WEEKDAY_PATH.resolve()}"
)
print(
    f"날짜 유형별 그래프: "
    f"{DAY_TYPE_GRAPH_PATH.resolve()}"
)
print(
    f"공휴일 표시 그래프: "
    f"{HOLIDAY_GRAPH_PATH.resolve()}"
)