from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rcParams


# ============================================================
# 1. 그래프 한글 폰트 설정
# ============================================================

def configure_korean_font() -> None:
    """
    Windows 기본 글꼴인 맑은 고딕을 Matplotlib에 적용한다.
    """
    font_path = Path("C:/Windows/Fonts/malgun.ttf")

    if font_path.exists():
        font_name = font_manager.FontProperties(
            fname=str(font_path)
        ).get_name()

        rcParams["font.family"] = font_name
        rcParams["axes.unicode_minus"] = False

        print(f"그래프 한글 폰트: {font_name}")
    else:
        print(
            "경고: 맑은 고딕 폰트를 찾지 못했습니다. "
            "그래프의 한글이 깨질 수 있습니다."
        )


configure_korean_font()


# ============================================================
# 2. 파일 경로 설정
# ============================================================

LONG_DATA_PATH = Path(
    "data/processed/subway_2025_long.parquet"
)

CALENDAR_PATH = Path(
    "outputs/03_daily_with_holidays.csv"
)

STATION_PROFILE_PATH = Path(
    "outputs/04_station_daytype_profile.csv"
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. 분석 설정
# ============================================================

MORNING_PEAK_SLOTS = [
    "07-08시간대",
    "08-09시간대",
    "09-10시간대",
]

EVENING_PEAK_SLOTS = [
    "17-18시간대",
    "18-19시간대",
    "19-20시간대",
]

# 방향성 점수 분류 기준
ORIENTATION_THRESHOLD = 0.15

# 순위표에서 지나치게 이용량이 작은 역을 제외하기 위한 기준
# 분류 자체에는 적용하지 않고 상·하위 순위표에만 적용한다.
MIN_WEEKDAY_AVERAGE_VOLUME = 10_000

STATION_NAME_ALIASES = {
    "삼각지": "삼각지(전쟁기념관)",
    "당고개": "불암산",
}


# ============================================================
# 4. 입력 파일 확인
# ============================================================

required_files = [
    LONG_DATA_PATH,
    CALENDAR_PATH,
    STATION_PROFILE_PATH,
]

for file_path in required_files:
    if not file_path.exists():
        raise FileNotFoundError(
            "\n필요한 파일을 찾지 못했습니다.\n"
            f"확인할 위치: {file_path.resolve()}\n"
        )


# ============================================================
# 5. Long 데이터 불러오기
# ============================================================

long_df = pd.read_parquet(
    LONG_DATA_PATH,
    columns=[
        "date",
        "line",
        "station_name",
        "direction",
        "time_slot",
        "time_order",
        "passenger_count",
    ],
)

long_df["date"] = pd.to_datetime(
    long_df["date"],
    errors="coerce",
)

long_df["station_name"] = (
    long_df["station_name"]
    .replace(STATION_NAME_ALIASES)
)

print("=" * 70)
print("1. Long 데이터 불러오기")
print("=" * 70)
print(f"행 수: {len(long_df):,}")
print(
    "표준화 후 고유 역 수: "
    f"{long_df['station_name'].nunique():,}"
)
print(
    "날짜 범위: "
    f"{long_df['date'].min().date()} "
    f"~ {long_df['date'].max().date()}"
)


# ============================================================
# 6. 기본 데이터 검증
# ============================================================

required_columns = [
    "date",
    "station_name",
    "direction",
    "time_slot",
    "time_order",
    "passenger_count",
]

missing_value_count = (
    long_df[required_columns]
    .isna()
    .any(axis=1)
    .sum()
)

negative_value_count = (
    long_df["passenger_count"] < 0
).sum()

direction_values = set(
    long_df["direction"]
    .dropna()
    .unique()
)

expected_direction_values = {
    "승차",
    "하차",
}

print("\n" + "=" * 70)
print("2. 기본 데이터 검증")
print("=" * 70)
print(f"필수 값 결측 행 수: {missing_value_count:,}")
print(f"음수 이용량 행 수: {negative_value_count:,}")
print(f"승하차 구분 값: {sorted(direction_values)}")

if missing_value_count > 0:
    raise ValueError(
        "필수 컬럼에 결측값이 있습니다."
    )

if negative_value_count > 0:
    raise ValueError(
        "승객 수에 음수가 존재합니다."
    )

if direction_values != expected_direction_values:
    raise ValueError(
        "\n승하차 구분 값이 예상과 다릅니다.\n"
        f"실제 값: {direction_values}\n"
    )


# ============================================================
# 7. 달력 데이터 불러오기
# ============================================================

calendar_df = pd.read_csv(
    CALENDAR_PATH,
    encoding="utf-8-sig",
)

calendar_df["date"] = pd.to_datetime(
    calendar_df["date"],
    errors="coerce",
)

required_calendar_columns = [
    "date",
    "day_type",
]

missing_calendar_columns = [
    column
    for column in required_calendar_columns
    if column not in calendar_df.columns
]

if missing_calendar_columns:
    raise ValueError(
        "\n달력 데이터에 필요한 컬럼이 없습니다.\n"
        f"누락 컬럼: {missing_calendar_columns}\n"
    )


# ============================================================
# 8. 일반 평일 날짜만 선택
# ============================================================

ordinary_weekday_dates = (
    calendar_df.loc[
        calendar_df["day_type"]
        == "Ordinary weekday",
        "date",
    ]
    .drop_duplicates()
    .sort_values()
)

number_of_ordinary_weekdays = (
    ordinary_weekday_dates.nunique()
)

weekday_df = long_df.loc[
    long_df["date"].isin(
        ordinary_weekday_dates
    )
].copy()

print("\n" + "=" * 70)
print("3. 일반 평일 데이터 선택")
print("=" * 70)
print(
    "일반 평일 날짜 수: "
    f"{number_of_ordinary_weekdays:,}"
)
print(
    "일반 평일 Long 행 수: "
    f"{len(weekday_df):,}"
)

if number_of_ordinary_weekdays != 244:
    print(
        "경고: 이전 분석에서 확인한 일반 평일 수 "
        "244일과 다릅니다."
    )


# ============================================================
# 9. 역·날짜·방향·시간대 단위 집계
# ============================================================

# 환승역의 여러 노선을 하나의 물리적 역으로 합산한다.
station_slot_daily = (
    weekday_df
    .groupby(
        [
            "date",
            "station_name",
            "direction",
            "time_slot",
            "time_order",
        ],
        as_index=False,
        observed=True,
    )["passenger_count"]
    .sum()
)

number_of_stations = (
    station_slot_daily["station_name"]
    .nunique()
)

number_of_time_slots = (
    station_slot_daily["time_slot"]
    .nunique()
)

number_of_directions = (
    station_slot_daily["direction"]
    .nunique()
)

expected_daily_rows = (
    number_of_stations
    * number_of_ordinary_weekdays
    * number_of_time_slots
    * number_of_directions
)

actual_daily_rows = len(station_slot_daily)

print("\n" + "=" * 70)
print("4. 역·날짜·방향·시간대 집계")
print("=" * 70)
print(f"고유 역 수: {number_of_stations:,}")
print(
    "고유 시간대 수: "
    f"{number_of_time_slots:,}"
)
print(
    "승하차 구분 수: "
    f"{number_of_directions:,}"
)
print(
    "예상 집계 행 수: "
    f"{expected_daily_rows:,}"
)
print(
    "실제 집계 행 수: "
    f"{actual_daily_rows:,}"
)

if expected_daily_rows != actual_daily_rows:
    raise ValueError(
        "\n예상 행 수와 실제 행 수가 다릅니다.\n"
        "일부 역·날짜·시간대·방향 조합이 "
        "누락되었을 가능성이 있습니다.\n"
    )


# ============================================================
# 10. 역별 일반 평일 날짜 커버리지 검증
# ============================================================

station_weekday_coverage = (
    station_slot_daily[
        [
            "station_name",
            "date",
        ]
    ]
    .drop_duplicates()
    .groupby(
        "station_name",
        as_index=False,
    )["date"]
    .nunique()
    .rename(
        columns={
            "date": "number_of_weekdays",
        }
    )
)

incomplete_station_coverage = (
    station_weekday_coverage.loc[
        station_weekday_coverage[
            "number_of_weekdays"
        ]
        != number_of_ordinary_weekdays
    ]
    .copy()
)

print("\n" + "=" * 70)
print("5. 일반 평일 날짜 커버리지")
print("=" * 70)
print(
    "역별 최소 일반 평일 수: "
    f"{station_weekday_coverage['number_of_weekdays'].min():,}"
)
print(
    "역별 최대 일반 평일 수: "
    f"{station_weekday_coverage['number_of_weekdays'].max():,}"
)
print(
    "일반 평일이 불완전한 역 수: "
    f"{len(incomplete_station_coverage):,}"
)

if not incomplete_station_coverage.empty:
    incomplete_station_coverage.to_csv(
        OUTPUT_DIR
        / "05_incomplete_weekday_station_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    raise ValueError(
        "일부 역의 일반 평일 데이터가 불완전합니다."
    )


# ============================================================
# 11. 역·방향·시간대별 일평균 계산
# ============================================================

station_slot_average = (
    station_slot_daily
    .groupby(
        [
            "station_name",
            "direction",
            "time_slot",
            "time_order",
        ],
        as_index=False,
        observed=True,
    )["passenger_count"]
    .agg(
        average_passengers="mean",
        median_passengers="median",
    )
)

# 승차와 하차를 각각 열로 변환한다.
station_slot_pivot = (
    station_slot_average
    .pivot(
        index=[
            "station_name",
            "time_slot",
            "time_order",
        ],
        columns="direction",
        values="average_passengers",
    )
    .reset_index()
)

station_slot_pivot.columns.name = None

for direction in [
    "승차",
    "하차",
]:
    if direction not in station_slot_pivot.columns:
        raise ValueError(
            f"{direction} 평균 컬럼이 생성되지 않았습니다."
        )

station_slot_pivot = (
    station_slot_pivot
    .rename(
        columns={
            "승차": "boarding_average",
            "하차": "alighting_average",
        }
    )
    .sort_values(
        [
            "station_name",
            "time_order",
        ]
    )
)


# ============================================================
# 12. 첨두시간별 파생 변수 생성
# ============================================================

is_morning_peak = (
    station_slot_pivot["time_slot"]
    .isin(MORNING_PEAK_SLOTS)
)

is_evening_peak = (
    station_slot_pivot["time_slot"]
    .isin(EVENING_PEAK_SLOTS)
)

station_slot_pivot["morning_boarding"] = (
    np.where(
        is_morning_peak,
        station_slot_pivot["boarding_average"],
        0,
    )
)

station_slot_pivot["morning_alighting"] = (
    np.where(
        is_morning_peak,
        station_slot_pivot["alighting_average"],
        0,
    )
)

station_slot_pivot["evening_boarding"] = (
    np.where(
        is_evening_peak,
        station_slot_pivot["boarding_average"],
        0,
    )
)

station_slot_pivot["evening_alighting"] = (
    np.where(
        is_evening_peak,
        station_slot_pivot["alighting_average"],
        0,
    )
)


# ============================================================
# 13. 역별 흐름 지표 계산
# ============================================================

station_flow_profile = (
    station_slot_pivot
    .groupby(
        "station_name",
        as_index=False,
    )
    .agg(
        all_day_boarding=(
            "boarding_average",
            "sum",
        ),
        all_day_alighting=(
            "alighting_average",
            "sum",
        ),
        morning_boarding=(
            "morning_boarding",
            "sum",
        ),
        morning_alighting=(
            "morning_alighting",
            "sum",
        ),
        evening_boarding=(
            "evening_boarding",
            "sum",
        ),
        evening_alighting=(
            "evening_alighting",
            "sum",
        ),
    )
)

station_flow_profile[
    "calculated_weekday_average"
] = (
    station_flow_profile["all_day_boarding"]
    + station_flow_profile["all_day_alighting"]
)

# 아침에 들어오는 순이동
station_flow_profile[
    "morning_net_inflow"
] = (
    station_flow_profile["morning_alighting"]
    - station_flow_profile["morning_boarding"]
)

# 저녁에 빠져나가는 순이동
station_flow_profile[
    "evening_net_outflow"
] = (
    station_flow_profile["evening_boarding"]
    - station_flow_profile["evening_alighting"]
)

# 업무 유입형 흐름
station_flow_profile[
    "employment_flow"
] = (
    station_flow_profile["morning_alighting"]
    + station_flow_profile["evening_boarding"]
)

# 주거 유출·귀가형 흐름
station_flow_profile[
    "residential_flow"
] = (
    station_flow_profile["morning_boarding"]
    + station_flow_profile["evening_alighting"]
)

station_flow_profile[
    "peak_direction_total"
] = (
    station_flow_profile["employment_flow"]
    + station_flow_profile["residential_flow"]
)

# 방향성 점수
station_flow_profile[
    "orientation_score"
] = np.where(
    station_flow_profile[
        "peak_direction_total"
    ] > 0,
    (
        station_flow_profile["employment_flow"]
        - station_flow_profile["residential_flow"]
    )
    / station_flow_profile[
        "peak_direction_total"
    ],
    np.nan,
)

# 하루 이용량 중 아침·저녁 첨두시간이 차지하는 비중
station_flow_profile[
    "peak_concentration"
] = np.where(
    station_flow_profile[
        "calculated_weekday_average"
    ] > 0,
    station_flow_profile[
        "peak_direction_total"
    ]
    / station_flow_profile[
        "calculated_weekday_average"
    ],
    np.nan,
)


# ============================================================
# 14. 4단계 역 프로파일 결합
# ============================================================

stage04_profile = pd.read_csv(
    STATION_PROFILE_PATH,
    encoding="utf-8-sig",
)

required_stage04_columns = [
    "station_name",
    "lines",
    "weekday_average",
    "weekend_average",
    "relative_weekend_index",
    "station_type",
]

missing_stage04_columns = [
    column
    for column in required_stage04_columns
    if column not in stage04_profile.columns
]

if missing_stage04_columns:
    raise ValueError(
        "\n4단계 프로파일에 필요한 컬럼이 없습니다.\n"
        f"누락 컬럼: {missing_stage04_columns}\n"
    )

stage04_profile = stage04_profile[
    required_stage04_columns
].copy()

station_flow_profile = (
    station_flow_profile
    .merge(
        stage04_profile,
        on="station_name",
        how="left",
        validate="one_to_one",
    )
)

stage04_merge_failure_count = (
    station_flow_profile["weekday_average"]
    .isna()
    .sum()
)

print("\n" + "=" * 70)
print("6. 4단계 역 프로파일 결합")
print("=" * 70)
print(
    "4단계 프로파일 결합 실패 역 수: "
    f"{stage04_merge_failure_count:,}"
)

if stage04_merge_failure_count > 0:
    raise ValueError(
        "4단계 역 프로파일 결합에 실패했습니다."
    )


# ============================================================
# 15. 모듈 간 이용량 검증
# ============================================================

station_flow_profile[
    "weekday_average_difference"
] = (
    station_flow_profile[
        "calculated_weekday_average"
    ]
    - station_flow_profile["weekday_average"]
)

maximum_volume_difference = (
    station_flow_profile[
        "weekday_average_difference"
    ]
    .abs()
    .max()
)

print(
    "4단계와 5단계 평일 평균의 최대 차이: "
    f"{maximum_volume_difference:.3f}"
)

# 4단계 값은 정수 반올림되었으므로 약간의 차이는 허용한다.
if maximum_volume_difference > 2:
    raise ValueError(
        "\n4단계와 5단계의 평일 평균 이용량이 "
        "일치하지 않습니다.\n"
        "집계 방식 또는 역사명 표준화를 확인하세요.\n"
    )


# ============================================================
# 16. 탐색적 흐름 유형 분류
# ============================================================

conditions = [
    station_flow_profile["orientation_score"]
    >= ORIENTATION_THRESHOLD,

    station_flow_profile["orientation_score"]
    <= -ORIENTATION_THRESHOLD,
]

choices = [
    "업무 유입형",
    "주거 유출·귀가형",
]

station_flow_profile["commute_pattern"] = (
    np.select(
        conditions,
        choices,
        default="혼합형",
    )
)


# ============================================================
# 17. 순위와 출력용 숫자 정리
# ============================================================

station_flow_profile[
    "orientation_rank"
] = (
    station_flow_profile[
        "orientation_score"
    ]
    .rank(
        method="min",
        ascending=False,
    )
    .astype("int64")
)

station_flow_profile[
    "weekday_volume_rank"
] = (
    station_flow_profile[
        "weekday_average"
    ]
    .rank(
        method="min",
        ascending=False,
    )
    .astype("int64")
)

# 저이용량 역은 작은 변화로 비율이 크게 흔들릴 수 있으므로
# 상·하위 순위표에서는 일정 규모 이상만 사용한다.
ranking_candidates = (
    station_flow_profile.loc[
        station_flow_profile["weekday_average"]
        >= MIN_WEEKDAY_AVERAGE_VOLUME
    ]
    .copy()
)

employment_inflow_top = (
    ranking_candidates
    .nlargest(
        20,
        "orientation_score",
    )
    .copy()
)

residential_flow_top = (
    ranking_candidates
    .nsmallest(
        20,
        "orientation_score",
    )
    .copy()
)

mixed_top = (
    ranking_candidates
    .assign(
        absolute_orientation_score=lambda data: (
            data["orientation_score"].abs()
        )
    )
    .nsmallest(
        20,
        "absolute_orientation_score",
    )
    .copy()
)


# ============================================================
# 18. 흐름 유형별 요약
# ============================================================

flow_type_summary = (
    station_flow_profile
    .groupby(
        "commute_pattern",
        as_index=False,
    )
    .agg(
        number_of_stations=(
            "station_name",
            "count",
        ),
        average_weekday_volume=(
            "weekday_average",
            "mean",
        ),
        average_orientation_score=(
            "orientation_score",
            "mean",
        ),
        average_peak_concentration=(
            "peak_concentration",
            "mean",
        ),
    )
)

flow_type_summary[
    "average_weekday_volume"
] = (
    flow_type_summary[
        "average_weekday_volume"
    ]
    .round()
    .astype("int64")
)

for column in [
    "average_orientation_score",
    "average_peak_concentration",
]:
    flow_type_summary[column] = (
        flow_type_summary[column]
        .round(3)
    )


# ============================================================
# 19. 시간대 프로파일에 흐름 유형 추가
# ============================================================

station_total_average = (
    station_slot_pivot
    .assign(
        total_slot_average=lambda data: (
            data["boarding_average"]
            + data["alighting_average"]
        )
    )
    .groupby(
        "station_name",
        as_index=False,
    )["total_slot_average"]
    .sum()
    .rename(
        columns={
            "total_slot_average":
            "station_total_average",
        }
    )
)

station_slot_profile = (
    station_slot_pivot
    .merge(
        station_flow_profile[
            [
                "station_name",
                "commute_pattern",
            ]
        ],
        on="station_name",
        how="left",
        validate="many_to_one",
    )
    .merge(
        station_total_average,
        on="station_name",
        how="left",
        validate="many_to_one",
    )
)

station_slot_profile[
    "boarding_share"
] = (
    station_slot_profile["boarding_average"]
    / station_slot_profile[
        "station_total_average"
    ]
)

station_slot_profile[
    "alighting_share"
] = (
    station_slot_profile["alighting_average"]
    / station_slot_profile[
        "station_total_average"
    ]
)

flow_type_time_profile = (
    station_slot_profile
    .groupby(
        [
            "commute_pattern",
            "time_slot",
            "time_order",
        ],
        as_index=False,
    )
    .agg(
        average_boarding_share=(
            "boarding_share",
            "mean",
        ),
        average_alighting_share=(
            "alighting_share",
            "mean",
        ),
    )
    .sort_values(
        [
            "commute_pattern",
            "time_order",
        ]
    )
)


# ============================================================
# 20. 결과 저장 전 숫자 반올림
# ============================================================

count_columns = [
    "all_day_boarding",
    "all_day_alighting",
    "calculated_weekday_average",
    "morning_boarding",
    "morning_alighting",
    "evening_boarding",
    "evening_alighting",
    "morning_net_inflow",
    "evening_net_outflow",
    "employment_flow",
    "residential_flow",
    "peak_direction_total",
]

for column in count_columns:
    station_flow_profile[column] = (
        station_flow_profile[column]
        .round()
        .astype("int64")
    )

for column in [
    "orientation_score",
    "peak_concentration",
    "weekday_average_difference",
]:
    station_flow_profile[column] = (
        station_flow_profile[column]
        .round(3)
    )

station_flow_profile = (
    station_flow_profile
    .sort_values(
        "weekday_volume_rank"
    )
    .reset_index(drop=True)
)


# ============================================================
# 21. 결과 파일 저장
# ============================================================

PROFILE_OUTPUT_PATH = (
    OUTPUT_DIR
    / "05_station_time_direction_profile.csv"
)

SLOT_OUTPUT_PATH = (
    OUTPUT_DIR
    / "05_station_time_direction_average.csv"
)

TYPE_TIME_OUTPUT_PATH = (
    OUTPUT_DIR
    / "05_flow_type_time_profile.csv"
)

TYPE_SUMMARY_OUTPUT_PATH = (
    OUTPUT_DIR
    / "05_flow_type_summary.csv"
)

EMPLOYMENT_OUTPUT_PATH = (
    OUTPUT_DIR
    / "05_employment_inflow_stations.csv"
)

RESIDENTIAL_OUTPUT_PATH = (
    OUTPUT_DIR
    / "05_residential_flow_stations.csv"
)

MIXED_OUTPUT_PATH = (
    OUTPUT_DIR
    / "05_mixed_flow_stations.csv"
)

station_flow_profile.to_csv(
    PROFILE_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

station_slot_profile.to_csv(
    SLOT_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

flow_type_time_profile.to_csv(
    TYPE_TIME_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

flow_type_summary.to_csv(
    TYPE_SUMMARY_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

employment_inflow_top.to_csv(
    EMPLOYMENT_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

residential_flow_top.to_csv(
    RESIDENTIAL_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)

mixed_top.to_csv(
    MIXED_OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 22. 아침 순유입·저녁 순유출 산점도
# ============================================================

plt.figure(figsize=(11, 8))

for pattern_name, group in (
    station_flow_profile
    .groupby("commute_pattern")
):
    plt.scatter(
        group["morning_net_inflow"] / 1_000,
        group["evening_net_outflow"] / 1_000,
        alpha=0.7,
        label=(
            f"{pattern_name} "
            f"(n={len(group)})"
        ),
    )

plt.axhline(
    0,
    linewidth=1,
    linestyle="--",
)

plt.axvline(
    0,
    linewidth=1,
    linestyle="--",
)

# 업무 유입형 상위 5개와 주거 유출·귀가형 상위 5개 표시
label_stations = pd.concat(
    [
        employment_inflow_top.head(5),
        residential_flow_top.head(5),
    ],
    ignore_index=True,
).drop_duplicates(
    subset=["station_name"]
)

for _, row in label_stations.iterrows():
    plt.annotate(
        row["station_name"],
        (
            row["morning_net_inflow"] / 1_000,
            row["evening_net_outflow"] / 1_000,
        ),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
    )

plt.title(
    "일반 평일 역별 아침 순유입과 저녁 순유출"
)
plt.xlabel(
    "아침 순유입: 하차 - 승차 (천 명)"
)
plt.ylabel(
    "저녁 순유출: 승차 - 하차 (천 명)"
)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

SCATTER_OUTPUT_PATH = (
    OUTPUT_DIR
    / "05_morning_inflow_evening_outflow_scatter.png"
)

plt.savefig(
    SCATTER_OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 23. 유형별 시간대 승차 비중 그래프
# ============================================================

plt.figure(figsize=(14, 7))

for pattern_name, group in (
    flow_type_time_profile
    .groupby("commute_pattern")
):
    group = group.sort_values("time_order")

    plt.plot(
        group["time_order"],
        group["average_boarding_share"] * 100,
        marker="o",
        label=pattern_name,
    )

time_order_table = (
    flow_type_time_profile[
        [
            "time_slot",
            "time_order",
        ]
    ]
    .drop_duplicates()
    .sort_values("time_order")
)

plt.xticks(
    time_order_table["time_order"],
    time_order_table["time_slot"],
    rotation=45,
    ha="right",
)

plt.title(
    "탐색적 흐름 유형별 일반 평일 시간대 승차 비중"
)
plt.xlabel("시간대")
plt.ylabel("일평균 전체 이용량 대비 승차 비중 (%)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

BOARDING_GRAPH_PATH = (
    OUTPUT_DIR
    / "05_flow_type_boarding_profile.png"
)

plt.savefig(
    BOARDING_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 24. 유형별 시간대 하차 비중 그래프
# ============================================================

plt.figure(figsize=(14, 7))

for pattern_name, group in (
    flow_type_time_profile
    .groupby("commute_pattern")
):
    group = group.sort_values("time_order")

    plt.plot(
        group["time_order"],
        group["average_alighting_share"] * 100,
        marker="o",
        label=pattern_name,
    )

plt.xticks(
    time_order_table["time_order"],
    time_order_table["time_slot"],
    rotation=45,
    ha="right",
)

plt.title(
    "탐색적 흐름 유형별 일반 평일 시간대 하차 비중"
)
plt.xlabel("시간대")
plt.ylabel("일평균 전체 이용량 대비 하차 비중 (%)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

ALIGHTING_GRAPH_PATH = (
    OUTPUT_DIR
    / "05_flow_type_alighting_profile.png"
)

plt.savefig(
    ALIGHTING_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 25. 터미널 출력
# ============================================================

DISPLAY_COLUMNS = [
    "station_name",
    "lines",
    "weekday_average",
    "morning_boarding",
    "morning_alighting",
    "evening_boarding",
    "evening_alighting",
    "orientation_score",
    "peak_concentration",
    "commute_pattern",
]

print("\n" + "=" * 70)
print("7. 탐색적 이동 흐름 유형 분포")
print("=" * 70)

print(
    flow_type_summary.to_string(
        index=False,
    )
)

print("\n" + "=" * 70)
print("8. 업무 유입 방향성 상위 20개 역")
print("=" * 70)

print(
    employment_inflow_top[
        DISPLAY_COLUMNS
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("9. 주거 유출·귀가 방향성 상위 20개 역")
print("=" * 70)

print(
    residential_flow_top[
        DISPLAY_COLUMNS
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("10. 혼합 흐름 상위 20개 역")
print("=" * 70)

print(
    mixed_top[
        DISPLAY_COLUMNS
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("11. 작업 완료")
print("=" * 70)
print(f"역별 흐름 프로파일: {PROFILE_OUTPUT_PATH.resolve()}")
print(f"시간대별 평균: {SLOT_OUTPUT_PATH.resolve()}")
print(f"유형별 시간 프로파일: {TYPE_TIME_OUTPUT_PATH.resolve()}")
print(f"유형별 요약: {TYPE_SUMMARY_OUTPUT_PATH.resolve()}")
print(f"업무 유입형 상위 역: {EMPLOYMENT_OUTPUT_PATH.resolve()}")
print(f"주거 흐름형 상위 역: {RESIDENTIAL_OUTPUT_PATH.resolve()}")
print(f"혼합형 상위 역: {MIXED_OUTPUT_PATH.resolve()}")
print(f"흐름 산점도: {SCATTER_OUTPUT_PATH.resolve()}")
print(f"승차 프로파일 그래프: {BOARDING_GRAPH_PATH.resolve()}")
print(f"하차 프로파일 그래프: {ALIGHTING_GRAPH_PATH.resolve()}")