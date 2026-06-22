from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import font_manager, rcParams

# =========================================
# Windows 한글 폰트 설정
# =========================================

def configure_korean_font() -> None:
    """
    Windows에 기본 설치된 맑은 고딕을 Matplotlib에 적용한다.
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
            "그래프의 한글이 정상적으로 표시되지 않을 수 있습니다."
        )
        
configure_korean_font()

# ============================================================
# 1. 파일 경로 설정
# ============================================================

LONG_DATA_PATH = Path(
    "data/processed/subway_2025_long.parquet"
)

CALENDAR_PATH = Path(
    "outputs/03_daily_with_holidays.csv"
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 입력 파일 존재 여부 확인
# ============================================================

for path in [
    LONG_DATA_PATH,
    CALENDAR_PATH,
]:
    if not path.exists():
        raise FileNotFoundError(
            "\n필요한 파일을 찾지 못했습니다.\n"
            f"파일 위치: {path.resolve()}\n"
        )


# ============================================================
# 3. Long 데이터 불러오기
# ============================================================

# 전체 컬럼을 읽지 않고 이번 분석에 필요한 컬럼만 읽는다.
# 메모리 사용량과 로딩 시간을 줄이기 위한 처리다.
long_df = pd.read_parquet(
    LONG_DATA_PATH,
    columns=[
        "date",
        "line",
        "station_name",
        "passenger_count",
    ],
)

long_df["date"] = pd.to_datetime(
    long_df["date"],
    errors="coerce",
)

# ===================================
# 역사명 표준화
# ===================================

# 같은 물리적 역이 기간에 따라 다른 이름으로 기록된 경우
# 하나의 대표 이름으로 통합한다.
STATION_NAME_ALIASES = {
    # 2025년 4월부터 병기명이 반영된 것으로 보이는 역
    "삼각지": "삼각지(전쟁기념관)",
    
    # 2025년 4월부터 역명이 변경되어 기록된 역
    "당고개": "불암산"
}

# 변경 전 이름을 감시 목적으로 보존한다.
long_df["station_name_original"] = (
    long_df["station_name"]
)

# 대표 이름으로 변환한다.
long_df["station_name"] = (
    long_df["station_name"]
    .replace(STATION_NAME_ALIASES)
)

alias_changed_rows = long_df.loc[
    long_df["station_name_original"]
    != long_df["station_name"]
].copy()

alias_audit = (
    alias_changed_rows
    .groupby(
        [
            "station_name_original",
            "station_name"
        ],
        as_index=False
    )
    .agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        number_of_rows=("date", "size")
    )
)

ALIAS_AUDIT_PATH = (
    OUTPUT_DIR / "04_station_name_alias_audit.csv"
)

alias_audit.to_csv(
    ALIAS_AUDIT_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\n역사명 표준화 내역:")
print(alias_audit.to_string(index=False))

print("=" * 70)
print("1. Long 데이터 불러오기")
print("=" * 70)
print(f"행 수: {len(long_df):,}")
print(f"열 수: {len(long_df.columns):,}")
print(
    "고유 역명 수: "
    f"{long_df['station_name'].nunique():,}"
)
print(
    "날짜 범위: "
    f"{long_df['date'].min().date()} "
    f"~ {long_df['date'].max().date()}"
)


# ============================================================
# 4. 기본 데이터 검증
# ============================================================

invalid_count = (
    long_df[
        [
            "date",
            "station_name",
            "passenger_count",
        ]
    ]
    .isna()
    .any(axis=1)
    .sum()
)

negative_count = (
    long_df["passenger_count"] < 0
).sum()

print("\n" + "=" * 70)
print("2. 기본 데이터 검증")
print("=" * 70)
print(f"필수 값 결측 행 수: {invalid_count:,}")
print(f"음수 이용량 행 수: {negative_count:,}")

if invalid_count > 0:
    raise ValueError(
        "필수 컬럼에 결측값이 있습니다."
    )

if negative_count > 0:
    raise ValueError(
        "이용량에 음수가 존재합니다."
    )


# ============================================================
# 5. 역별 노선 정보 만들기
# ============================================================

# 환승역은 여러 노선에 등장한다.
# 같은 역명의 노선을 하나의 문자열로 합친다.
station_lines = (
    long_df[
        [
            "station_name",
            "line",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "station_name",
            "line",
        ]
    )
    .groupby(
        "station_name",
        as_index=False,
    )["line"]
    .agg(
        lambda values: ", ".join(values)
    )
    .rename(
        columns={
            "line": "lines",
        }
    )
)


# ============================================================
# 6. 역·날짜별 전체 이용량 계산
# ============================================================

# 같은 역명의 여러 노선과 모든 시간대,
# 승차와 하차를 합산한다.
station_daily = (
    long_df
    .groupby(
        [
            "date",
            "station_name",
        ],
        as_index=False,
        observed=True,
    )["passenger_count"]
    .sum()
)

# ===============================================================
# 역별 날짜 커버리지 검증
# ===============================================================

# 전체 데이터에 존재하는 날짜 수
expected_number_of_days = (
    long_df["date"].nunique()
)

# 역마다 며칠의 데이터가 존재하는지 계산
station_date_coverage = (
    station_daily
    .groupby(
        "station_name",
        as_index=False
    )
    .agg(
        first_date=("date", "min"),
        last_date=("date", "max"),
        number_of_days=("date", "nunique")
    )
    .sort_values(
        [
            "number_of_days",
            "station_name"
        ]
    )
)

# 전체 날짜 수보다 적은 역 찾기
incomplete_stations = (
    station_date_coverage.loc[
        station_date_coverage["number_of_days"]
        != expected_number_of_days
    ]
    .copy()
)

STATION_COVERAGE_PATH = (
    OUTPUT_DIR / "04_station_date_coverage.csv"
)

INCOMPLETE_STATIONS_PATH = (
    OUTPUT_DIR / "04_incomplete_station_dates.csv"
)

station_date_coverage.to_csv(
    STATION_COVERAGE_PATH,
    index=False,
    encoding="utf-8-sig"
)

incomplete_stations.to_csv(
    INCOMPLETE_STATIONS_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 70)
print("역별 날짜 커버리지 검증")
print("=" * 70)
print(
    "전체 데이터 날짜 수: "
    f"{expected_number_of_days:,}"
)
print(
    "고유 표준 역 수: "
    f"{station_date_coverage['station_name'].nunique():,}"
)
print(
    "역별 최소 관측 일수: "
    f"{station_date_coverage['number_of_days'].min():,}"
)
print(
    "역별 최대 관측 일수: "
    f"{station_date_coverage['number_of_days'].max():,}"
)
print(
    "날짜가 불완전한 역 수: "
    f"{len(incomplete_stations):,}"
)

if not incomplete_stations.empty:
    print("\n날짜가 불완전한 역:")
    print(
        incomplete_stations.to_string(
            index=False,
        )
    )

    raise ValueError(
        "\n일부 역의 날짜 데이터가 불완전합니다.\n"
        "outputs/04_incomplete_station_dates.csv를 "
        "확인하세요.\n"
        "역사명 변경 또는 데이터 누락 가능성이 있습니다."
    )

print("\n" + "=" * 70)
print("3. 역·날짜 데이터")
print("=" * 70)
print(
    f"역·날짜 행 수: {len(station_daily):,}"
)
print(
    "고유 역명 수: "
    f"{station_daily['station_name'].nunique():,}"
)
print(
    "고유 날짜 수: "
    f"{station_daily['date'].nunique():,}"
)


# ============================================================
# 7. 공휴일·요일 정보 불러오기
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
    "weekday",
    "is_weekend",
    "is_holiday",
    "holiday_name",
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

calendar_df = calendar_df[
    required_calendar_columns
].copy()


# ============================================================
# 8. 역·날짜 데이터와 달력 결합
# ============================================================

station_daily = station_daily.merge(
    calendar_df,
    on="date",
    how="left",
    validate="many_to_one",
)

calendar_missing_count = (
    station_daily["day_type"].isna().sum()
)

print("\n" + "=" * 70)
print("4. 달력 데이터 결합")
print("=" * 70)
print(
    "달력 정보가 결합되지 않은 행 수: "
    f"{calendar_missing_count:,}"
)

if calendar_missing_count > 0:
    missing_dates = (
        station_daily.loc[
            station_daily["day_type"].isna(),
            "date",
        ]
        .drop_duplicates()
        .sort_values()
    )

    print("결합되지 않은 날짜:")
    print(missing_dates)

    raise ValueError(
        "달력 데이터 결합에 실패한 날짜가 있습니다."
    )


# ============================================================
# 9. 날짜 유형별 역 평균 계산
# ============================================================

station_daytype_average = (
    station_daily
    .groupby(
        [
            "station_name",
            "day_type",
        ],
        as_index=False,
    )["passenger_count"]
    .agg(
        average="mean",
        median="median",
        minimum="min",
        maximum="max",
        number_of_days="count",
    )
)

# 날짜 유형을 열로 변환한다.
average_pivot = (
    station_daytype_average
    .pivot(
        index="station_name",
        columns="day_type",
        values="average",
    )
    .reset_index()
)

average_pivot.columns.name = None

average_pivot = average_pivot.rename(
    columns={
        "Ordinary weekday": "weekday_average",
        "Ordinary weekend": "weekend_average",
        "Public holiday": "holiday_average",
    }
)


# ============================================================
# 10. 모든 역에 필요한 평균이 존재하는지 확인
# ============================================================

required_average_columns = [
    "weekday_average",
    "weekend_average",
    "holiday_average",
]

missing_average_columns = [
    column
    for column in required_average_columns
    if column not in average_pivot.columns
]

if missing_average_columns:
    raise ValueError(
        "\n날짜 유형별 평균 컬럼 생성에 실패했습니다.\n"
        f"누락 컬럼: {missing_average_columns}\n"
    )


# ============================================================
# 11. 전체 평균 이용량 추가
# ============================================================

overall_average = (
    station_daily
    .groupby(
        "station_name",
        as_index=False,
    )["passenger_count"]
    .mean()
    .rename(
        columns={
            "passenger_count": "overall_average",
        }
    )
)

station_profile = (
    average_pivot
    .merge(
        overall_average,
        on="station_name",
        how="left",
        validate="one_to_one",
    )
    .merge(
        station_lines,
        on="station_name",
        how="left",
        validate="one_to_one",
    )
)


# ============================================================
# 12. 네트워크 전체 주말 비율 계산
# ============================================================

network_daily = (
    station_daily
    .groupby(
        "date",
        as_index=False,
    )["passenger_count"]
    .sum()
    .merge(
        calendar_df[
            [
                "date",
                "day_type",
            ]
        ],
        on="date",
        how="left",
        validate="one_to_one",
    )
)

network_weekday_average = (
    network_daily.loc[
        network_daily["day_type"]
        == "Ordinary weekday",
        "passenger_count",
    ]
    .mean()
)

network_weekend_average = (
    network_daily.loc[
        network_daily["day_type"]
        == "Ordinary weekend",
        "passenger_count",
    ]
    .mean()
)

network_weekend_ratio = (
    network_weekend_average
    / network_weekday_average
)

print("\n" + "=" * 70)
print("5. 네트워크 기준")
print("=" * 70)
print(
    "네트워크 일반 평일 평균: "
    f"{network_weekday_average:,.0f}"
)
print(
    "네트워크 일반 주말 평균: "
    f"{network_weekend_average:,.0f}"
)
print(
    "네트워크 주말 비율: "
    f"{network_weekend_ratio:.3f}"
)


# ============================================================
# 13. 역별 지표 생성
# ============================================================

station_profile["weekend_ratio"] = (
    station_profile["weekend_average"]
    / station_profile["weekday_average"]
)

station_profile["holiday_ratio"] = (
    station_profile["holiday_average"]
    / station_profile["weekday_average"]
)

station_profile["relative_weekend_index"] = (
    station_profile["weekend_ratio"]
    / network_weekend_ratio
)

station_profile["weekend_change_rate"] = (
    (
        station_profile["weekend_average"]
        / station_profile["weekday_average"]
    )
    - 1
) * 100


# ============================================================
# 14. 탐색적 역사 유형 분류
# ============================================================

def classify_station(
    relative_weekend_index: float,
) -> str:
    """
    네트워크 전체 주말 패턴과 비교하여
    역의 탐색적 유형을 분류한다.

    이 기준은 확정된 정답이 아니라
    초기 탐색을 위한 휴리스틱 규칙이다.
    """
    if pd.isna(relative_weekend_index):
        return "Unclassified"

    if relative_weekend_index < 0.8:
        return "Weekday-oriented"

    if relative_weekend_index <= 1.2:
        return "Balanced"

    return "Weekend-relative-strong"


station_profile["station_type"] = (
    station_profile["relative_weekend_index"]
    .apply(classify_station)
)


# ============================================================
# 15. 이용량 순위 생성
# ============================================================

station_profile["overall_volume_rank"] = (
    station_profile["overall_average"]
    .rank(
        method="min",
        ascending=False,
    )
    .astype("int64")
)

station_profile["weekend_index_rank"] = (
    station_profile["relative_weekend_index"]
    .rank(
        method="min",
        ascending=False,
    )
    .astype("int64")
)


# ============================================================
# 16. 숫자 정리 및 정렬
# ============================================================

count_columns = [
    "weekday_average",
    "weekend_average",
    "holiday_average",
    "overall_average",
]

for column in count_columns:
    station_profile[column] = (
        station_profile[column]
        .round()
        .astype("int64")
    )

ratio_columns = [
    "weekend_ratio",
    "holiday_ratio",
    "relative_weekend_index",
    "weekend_change_rate",
]

for column in ratio_columns:
    station_profile[column] = (
        station_profile[column]
        .round(3)
    )

station_profile = station_profile.sort_values(
    "overall_volume_rank"
).reset_index(drop=True)


# ============================================================
# 17. 결과 테이블 생성
# ============================================================

top_volume = (
    station_profile
    .nsmallest(
        20,
        "overall_volume_rank",
    )
    .copy()
)

top_weekend_strength = (
    station_profile
    .nlargest(
        20,
        "relative_weekend_index",
    )
    .copy()
)

top_weekday_strength = (
    station_profile
    .nsmallest(
        20,
        "relative_weekend_index",
    )
    .copy()
)

station_type_summary = (
    station_profile
    .groupby(
        "station_type",
        as_index=False,
    )
    .agg(
        number_of_stations=(
            "station_name",
            "count",
        ),
        average_daily_volume=(
            "overall_average",
            "mean",
        ),
        average_relative_weekend_index=(
            "relative_weekend_index",
            "mean",
        ),
    )
)

station_type_summary[
    "average_daily_volume"
] = (
    station_type_summary[
        "average_daily_volume"
    ]
    .round()
    .astype("int64")
)

station_type_summary[
    "average_relative_weekend_index"
] = (
    station_type_summary[
        "average_relative_weekend_index"
    ]
    .round(3)
)


# ============================================================
# 18. CSV 저장
# ============================================================

PROFILE_PATH = (
    OUTPUT_DIR / "04_station_daytype_profile.csv"
)

TOP_VOLUME_PATH = (
    OUTPUT_DIR / "04_top_volume_stations.csv"
)

TOP_WEEKEND_PATH = (
    OUTPUT_DIR / "04_weekend_relative_strong_stations.csv"
)

TOP_WEEKDAY_PATH = (
    OUTPUT_DIR / "04_weekday_oriented_stations.csv"
)

TYPE_SUMMARY_PATH = (
    OUTPUT_DIR / "04_station_type_summary.csv"
)

STATION_DAILY_PATH = (
    OUTPUT_DIR / "04_station_daily.csv"
)

station_profile.to_csv(
    PROFILE_PATH,
    index=False,
    encoding="utf-8-sig",
)

top_volume.to_csv(
    TOP_VOLUME_PATH,
    index=False,
    encoding="utf-8-sig",
)

top_weekend_strength.to_csv(
    TOP_WEEKEND_PATH,
    index=False,
    encoding="utf-8-sig",
)

top_weekday_strength.to_csv(
    TOP_WEEKDAY_PATH,
    index=False,
    encoding="utf-8-sig",
)

station_type_summary.to_csv(
    TYPE_SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig",
)

station_daily.to_csv(
    STATION_DAILY_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 19. 평일·주말 평균 산점도
# ============================================================

plt.figure(figsize=(10, 8))

plt.scatter(
    station_profile["weekday_average"],
    station_profile["weekend_average"],
    alpha=0.7,
)

max_value = max(
    station_profile["weekday_average"].max(),
    station_profile["weekend_average"].max(),
)

reference_x = np.linspace(
    0,
    max_value,
    100,
)

# 네트워크 전체의 평일 대비 주말 비율을 기준선으로 표시
reference_y = (
    reference_x * network_weekend_ratio
)

plt.plot(
    reference_x,
    reference_y,
    linestyle="--",
    label=(
        "Network weekend ratio "
        f"({network_weekend_ratio:.3f})"
    ),
)

# 이용량이 많은 상위 10개 역 이름 표시
label_stations = station_profile.nsmallest(
    10,
    "overall_volume_rank",
)

for _, row in label_stations.iterrows():
    plt.annotate(
        row["station_name"],
        (
            row["weekday_average"],
            row["weekend_average"],
        ),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
    )

plt.title(
    "Station Weekday vs Weekend Passenger Volume"
)
plt.xlabel(
    "Average Passengers on Ordinary Weekdays"
)
plt.ylabel(
    "Average Passengers on Ordinary Weekends"
)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

SCATTER_PATH = (
    OUTPUT_DIR / "04_station_weekday_weekend_scatter.png"
)

plt.savefig(
    SCATTER_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 20. 유형별 역사 수 그래프
# ============================================================

type_plot = station_type_summary.sort_values(
    "number_of_stations",
    ascending=False,
)

plt.figure(figsize=(10, 6))

bars = plt.bar(
    type_plot["station_type"],
    type_plot["number_of_stations"],
)

plt.title(
    "Number of Stations by Exploratory Day-Type Profile"
)
plt.xlabel("Station Type")
plt.ylabel("Number of Stations")
plt.grid(
    axis="y",
    alpha=0.3,
)

for bar, value in zip(
    bars,
    type_plot["number_of_stations"],
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{int(value)}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()

TYPE_GRAPH_PATH = (
    OUTPUT_DIR / "04_station_type_counts.png"
)

plt.savefig(
    TYPE_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 21. 터미널 결과 출력
# ============================================================

DISPLAY_COLUMNS = [
    "station_name",
    "lines",
    "weekday_average",
    "weekend_average",
    "relative_weekend_index",
    "station_type",
]

print("\n" + "=" * 70)
print("6. 탐색적 역사 유형 분포")
print("=" * 70)
print(
    station_type_summary.to_string(
        index=False,
    )
)

print("\n" + "=" * 70)
print("7. 전체 일평균 이용량 상위 20개 역")
print("=" * 70)
print(
    top_volume[
        DISPLAY_COLUMNS
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("8. 주말 상대강세 지수 상위 20개 역")
print("=" * 70)
print(
    top_weekend_strength[
        DISPLAY_COLUMNS
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("9. 평일 집중도 상위 20개 역")
print("=" * 70)
print(
    top_weekday_strength[
        DISPLAY_COLUMNS
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("10. 작업 완료")
print("=" * 70)
print(f"역별 프로파일: {PROFILE_PATH.resolve()}")
print(f"이용량 상위 역: {TOP_VOLUME_PATH.resolve()}")
print(f"주말 상대강세 역: {TOP_WEEKEND_PATH.resolve()}")
print(f"평일 집중형 역: {TOP_WEEKDAY_PATH.resolve()}")
print(f"유형별 요약: {TYPE_SUMMARY_PATH.resolve()}")
print(f"역·날짜 데이터: {STATION_DAILY_PATH.resolve()}")
print(f"평일·주말 산점도: {SCATTER_PATH.resolve()}")
print(f"유형별 역사 수: {TYPE_GRAPH_PATH.resolve()}")