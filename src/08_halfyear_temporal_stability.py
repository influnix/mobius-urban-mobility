from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rcParams
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)


# ============================================================
# 1. 한글 폰트 설정
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
            "경고: 맑은 고딕을 찾지 못했습니다. "
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

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. 분석 설정
# ============================================================

PERIOD_ORDER = [
    "H1",
    "H2",
]

PERIOD_LABELS = {
    "H1": "2025 상반기",
    "H2": "2025 하반기",
}

EXPECTED_PERIOD_DAY_COUNTS = {
    "H1": 119,
    "H2": 125,
}

K_VALUES = [
    2,
    3,
]

RANDOM_STATE = 42
N_INIT = 20
MAX_ITER = 500

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

STATION_NAME_ALIASES = {
    "삼각지": "삼각지(전쟁기념관)",
    "당고개": "불암산",
}


# ============================================================
# 4. 입력 파일 확인
# ============================================================

for file_path in [
    LONG_DATA_PATH,
    CALENDAR_PATH,
]:
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
# 6. Long 데이터 기본 검증
# ============================================================

required_long_columns = [
    "date",
    "station_name",
    "direction",
    "time_slot",
    "time_order",
    "passenger_count",
]

missing_value_count = (
    long_df[required_long_columns]
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
print("2. Long 데이터 검증")
print("=" * 70)
print(f"필수 값 결측 행 수: {missing_value_count:,}")
print(f"음수 승객 수 행 수: {negative_value_count:,}")
print(f"승하차 구분 값: {sorted(direction_values)}")

if missing_value_count > 0:
    raise ValueError(
        "Long 데이터의 필수 컬럼에 결측값이 있습니다."
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

calendar_df = calendar_df[
    required_calendar_columns
].copy()

calendar_df["half_year"] = np.where(
    calendar_df["date"].dt.month <= 6,
    "H1",
    "H2",
)


# ============================================================
# 8. 일반 평일 날짜 선택
# ============================================================

ordinary_calendar = (
    calendar_df.loc[
        calendar_df["day_type"]
        == "Ordinary weekday"
    ]
    .copy()
)

period_day_counts = (
    ordinary_calendar
    .groupby(
        "half_year",
    )["date"]
    .nunique()
    .reindex(PERIOD_ORDER)
)

print("\n" + "=" * 70)
print("3. 상·하반기 일반 평일 날짜")
print("=" * 70)

for period in PERIOD_ORDER:
    actual_count = int(
        period_day_counts.loc[period]
    )

    expected_count = (
        EXPECTED_PERIOD_DAY_COUNTS[period]
    )

    print(
        f"{PERIOD_LABELS[period]} 일반 평일 수: "
        f"{actual_count:,}"
    )

    if actual_count != expected_count:
        raise ValueError(
            f"\n{period} 일반 평일 수가 예상과 다릅니다.\n"
            f"예상: {expected_count}\n"
            f"실제: {actual_count}\n"
        )

total_ordinary_weekdays = int(
    period_day_counts.sum()
)

print(
    "전체 일반 평일 수: "
    f"{total_ordinary_weekdays:,}"
)

if total_ordinary_weekdays != 244:
    raise ValueError(
        "전체 일반 평일 수가 244일이 아닙니다."
    )


# ============================================================
# 9. Long 데이터에 기간 정보 결합
# ============================================================

weekday_df = long_df.merge(
    ordinary_calendar[
        [
            "date",
            "half_year",
        ]
    ],
    on="date",
    how="inner",
    validate="many_to_one",
)

print("\n" + "=" * 70)
print("4. 일반 평일 데이터 선택")
print("=" * 70)
print(
    "일반 평일 Long 행 수: "
    f"{len(weekday_df):,}"
)

weekday_period_counts = (
    weekday_df
    .groupby("half_year")
    .size()
    .reindex(PERIOD_ORDER)
)

for period in PERIOD_ORDER:
    print(
        f"{PERIOD_LABELS[period]} Long 행 수: "
        f"{int(weekday_period_counts.loc[period]):,}"
    )


# ============================================================
# 10. 기간·날짜·역·방향·시간대 단위 집계
# ============================================================

# 같은 역명의 여러 노선을 물리적 역 하나로 합산한다.
station_slot_daily = (
    weekday_df
    .groupby(
        [
            "half_year",
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

print("\n" + "=" * 70)
print("5. 기간별 역·날짜·방향·시간대 집계")
print("=" * 70)
print(f"고유 역 수: {number_of_stations:,}")
print(f"고유 시간대 수: {number_of_time_slots:,}")
print(f"승하차 방향 수: {number_of_directions:,}")

if number_of_stations != 240:
    raise ValueError(
        "표준화된 물리적 역 수가 240개가 아닙니다."
    )

if number_of_time_slots != 20:
    raise ValueError(
        "시간대 수가 20개가 아닙니다."
    )

if number_of_directions != 2:
    raise ValueError(
        "승하차 방향 수가 2개가 아닙니다."
    )

for period in PERIOD_ORDER:
    number_of_days = int(
        period_day_counts.loc[period]
    )

    expected_rows = (
        number_of_stations
        * number_of_days
        * number_of_time_slots
        * number_of_directions
    )

    actual_rows = len(
        station_slot_daily.loc[
            station_slot_daily["half_year"]
            == period
        ]
    )

    print(
        f"{PERIOD_LABELS[period]} 예상 행 수: "
        f"{expected_rows:,}"
    )
    print(
        f"{PERIOD_LABELS[period]} 실제 행 수: "
        f"{actual_rows:,}"
    )

    if expected_rows != actual_rows:
        raise ValueError(
            f"\n{period} 예상 행 수와 실제 행 수가 다릅니다.\n"
            "역·날짜·시간대·방향 조합 누락 가능성이 있습니다.\n"
        )


# ============================================================
# 11. 역별 기간 날짜 커버리지 검증
# ============================================================

station_period_coverage = (
    station_slot_daily[
        [
            "half_year",
            "station_name",
            "date",
        ]
    ]
    .drop_duplicates()
    .groupby(
        [
            "half_year",
            "station_name",
        ],
        as_index=False,
    )["date"]
    .nunique()
    .rename(
        columns={
            "date": "number_of_days",
        }
    )
)

coverage_errors = []

for period in PERIOD_ORDER:
    expected_days = int(
        period_day_counts.loc[period]
    )

    period_coverage = (
        station_period_coverage.loc[
            station_period_coverage["half_year"]
            == period
        ]
    )

    incomplete = period_coverage.loc[
        period_coverage["number_of_days"]
        != expected_days
    ]

    print(
        f"{PERIOD_LABELS[period]} 역별 최소 관측일: "
        f"{period_coverage['number_of_days'].min():,}"
    )
    print(
        f"{PERIOD_LABELS[period]} 역별 최대 관측일: "
        f"{period_coverage['number_of_days'].max():,}"
    )
    print(
        f"{PERIOD_LABELS[period]} 불완전 역 수: "
        f"{len(incomplete):,}"
    )

    if not incomplete.empty:
        coverage_errors.append(
            incomplete
        )

if coverage_errors:
    coverage_error_df = pd.concat(
        coverage_errors,
        ignore_index=True,
    )

    coverage_error_df.to_csv(
        OUTPUT_DIR
        / "08_incomplete_halfyear_coverage.csv",
        index=False,
        encoding="utf-8-sig",
    )

    raise ValueError(
        "일부 역의 상·하반기 날짜 데이터가 불완전합니다."
    )


# ============================================================
# 12. 기간·역·방향·시간대별 일평균 계산
# ============================================================

station_slot_average = (
    station_slot_daily
    .groupby(
        [
            "half_year",
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

station_slot_pivot = (
    station_slot_average
    .pivot(
        index=[
            "half_year",
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

station_slot_pivot = (
    station_slot_pivot
    .rename(
        columns={
            "승차": "boarding_average",
            "하차": "alighting_average",
        }
    )
)

required_direction_columns = [
    "boarding_average",
    "alighting_average",
]

missing_direction_columns = [
    column
    for column in required_direction_columns
    if column not in station_slot_pivot.columns
]

if missing_direction_columns:
    raise ValueError(
        "\n승하차 평균 컬럼 생성에 실패했습니다.\n"
        f"누락 컬럼: {missing_direction_columns}\n"
    )


# ============================================================
# 13. 기간·역별 전체 평균과 비중 계산
# ============================================================

station_total_average = (
    station_slot_pivot
    .assign(
        slot_total=lambda data: (
            data["boarding_average"]
            + data["alighting_average"]
        )
    )
    .groupby(
        [
            "half_year",
            "station_name",
        ],
        as_index=False,
    )["slot_total"]
    .sum()
    .rename(
        columns={
            "slot_total": "weekday_average",
        }
    )
)

station_slot_profile = (
    station_slot_pivot
    .merge(
        station_total_average,
        on=[
            "half_year",
            "station_name",
        ],
        how="left",
        validate="many_to_one",
    )
)

station_slot_profile[
    "boarding_share"
] = (
    station_slot_profile["boarding_average"]
    / station_slot_profile["weekday_average"]
)

station_slot_profile[
    "alighting_share"
] = (
    station_slot_profile["alighting_average"]
    / station_slot_profile["weekday_average"]
)


# ============================================================
# 14. 역별 비중 합계 검증
# ============================================================

share_validation = (
    station_slot_profile
    .groupby(
        [
            "half_year",
            "station_name",
        ],
        as_index=False,
    )
    .agg(
        boarding_share_sum=(
            "boarding_share",
            "sum",
        ),
        alighting_share_sum=(
            "alighting_share",
            "sum",
        ),
    )
)

share_validation["total_share_sum"] = (
    share_validation["boarding_share_sum"]
    + share_validation["alighting_share_sum"]
)

maximum_share_error = (
    share_validation["total_share_sum"]
    .sub(1)
    .abs()
    .max()
)

print("\n" + "=" * 70)
print("6. 기간별 시간대 프로파일 검증")
print("=" * 70)
print(
    "기간·역·시간대 프로파일 행 수: "
    f"{len(station_slot_profile):,}"
)
print(
    "비중 합계 최대 오차: "
    f"{maximum_share_error:.10f}"
)

expected_profile_rows = (
    len(PERIOD_ORDER)
    * number_of_stations
    * number_of_time_slots
)

print(
    "예상 프로파일 행 수: "
    f"{expected_profile_rows:,}"
)

if len(station_slot_profile) != expected_profile_rows:
    raise ValueError(
        "기간별 시간대 프로파일 행 수가 예상과 다릅니다."
    )

if maximum_share_error > 1e-6:
    raise ValueError(
        "기간·역별 승하차 비중 합계가 1이 아닙니다."
    )


# ============================================================
# 15. 기간별 방향성 점수 계산
# ============================================================

is_morning_peak = (
    station_slot_profile["time_slot"]
    .isin(MORNING_PEAK_SLOTS)
)

is_evening_peak = (
    station_slot_profile["time_slot"]
    .isin(EVENING_PEAK_SLOTS)
)

station_slot_profile["morning_boarding"] = np.where(
    is_morning_peak,
    station_slot_profile["boarding_average"],
    0,
)

station_slot_profile["morning_alighting"] = np.where(
    is_morning_peak,
    station_slot_profile["alighting_average"],
    0,
)

station_slot_profile["evening_boarding"] = np.where(
    is_evening_peak,
    station_slot_profile["boarding_average"],
    0,
)

station_slot_profile["evening_alighting"] = np.where(
    is_evening_peak,
    station_slot_profile["alighting_average"],
    0,
)

period_station_flow = (
    station_slot_profile
    .groupby(
        [
            "half_year",
            "station_name",
        ],
        as_index=False,
    )
    .agg(
        weekday_average=(
            "weekday_average",
            "first",
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

period_station_flow[
    "employment_flow"
] = (
    period_station_flow["morning_alighting"]
    + period_station_flow["evening_boarding"]
)

period_station_flow[
    "residential_flow"
] = (
    period_station_flow["morning_boarding"]
    + period_station_flow["evening_alighting"]
)

period_station_flow[
    "peak_direction_total"
] = (
    period_station_flow["employment_flow"]
    + period_station_flow["residential_flow"]
)

period_station_flow[
    "orientation_score"
] = np.where(
    period_station_flow[
        "peak_direction_total"
    ] > 0,
    (
        period_station_flow["employment_flow"]
        - period_station_flow["residential_flow"]
    )
    / period_station_flow["peak_direction_total"],
    np.nan,
)

period_station_flow[
    "peak_concentration"
] = np.where(
    period_station_flow["weekday_average"] > 0,
    period_station_flow["peak_direction_total"]
    / period_station_flow["weekday_average"],
    np.nan,
)


# ============================================================
# 16. 시간대 순서와 군집 입력 행렬 생성 함수
# ============================================================

time_order_table = (
    station_slot_profile[
        [
            "time_order",
            "time_slot",
        ]
    ]
    .drop_duplicates()
    .sort_values("time_order")
)

time_orders = (
    time_order_table["time_order"]
    .tolist()
)


def build_feature_matrix(
    dataframe: pd.DataFrame,
    period: str,
) -> pd.DataFrame:
    """
    특정 기간의 역 × 40개 패턴 변수 행렬을 만든다.
    """
    period_df = dataframe.loc[
        dataframe["half_year"] == period
    ].copy()

    boarding_features = (
        period_df
        .pivot(
            index="station_name",
            columns="time_order",
            values="boarding_share",
        )
        .reindex(columns=time_orders)
    )

    alighting_features = (
        period_df
        .pivot(
            index="station_name",
            columns="time_order",
            values="alighting_share",
        )
        .reindex(columns=time_orders)
    )

    boarding_features.columns = [
        f"boarding_{int(order):02d}"
        for order in boarding_features.columns
    ]

    alighting_features.columns = [
        f"alighting_{int(order):02d}"
        for order in alighting_features.columns
    ]

    feature_df = pd.concat(
        [
            boarding_features,
            alighting_features,
        ],
        axis=1,
    )

    feature_df = feature_df.sort_index()

    if feature_df.shape != (
        number_of_stations,
        number_of_time_slots * 2,
    ):
        raise ValueError(
            f"{period} 입력 행렬 크기가 예상과 다릅니다."
        )

    if feature_df.isna().sum().sum() > 0:
        raise ValueError(
            f"{period} 입력 행렬에 결측값이 있습니다."
        )

    return feature_df


feature_frames = {
    period: build_feature_matrix(
        station_slot_profile,
        period,
    )
    for period in PERIOD_ORDER
}

if not feature_frames["H1"].index.equals(
    feature_frames["H2"].index
):
    raise ValueError(
        "상반기와 하반기의 역 목록 또는 정렬이 다릅니다."
    )

station_names = (
    feature_frames["H1"].index
)

print("\n" + "=" * 70)
print("7. 기간별 군집화 입력 행렬")
print("=" * 70)

for period in PERIOD_ORDER:
    print(
        f"{PERIOD_LABELS[period]} 입력 크기: "
        f"{feature_frames[period].shape[0]:,}행 "
        f"× {feature_frames[period].shape[1]:,}열"
    )


# ============================================================
# 17. 군집 이름 생성 함수
# ============================================================

def create_cluster_name_map(
    label_df: pd.DataFrame,
    number_of_clusters: int,
) -> dict:
    """
    군집의 평균 방향성 점수를 기준으로
    군집 ID에 탐색적 이름을 부여한다.
    """
    orientation_summary = (
        label_df
        .groupby(
            "cluster_id",
            as_index=False,
        )["orientation_score"]
        .mean()
        .sort_values("orientation_score")
    )

    ordered_cluster_ids = (
        orientation_summary[
            "cluster_id"
        ]
        .tolist()
    )

    if number_of_clusters == 2:
        return {
            ordered_cluster_ids[0]:
                "주거 유출·귀가 중심",
            ordered_cluster_ids[1]:
                "업무 유입 중심",
        }

    if number_of_clusters == 3:
        return {
            ordered_cluster_ids[0]:
                "주거 유출·귀가 중심",
            ordered_cluster_ids[1]:
                "혼합 흐름 중심",
            ordered_cluster_ids[2]:
                "업무 유입 중심",
        }

    raise ValueError(
        "이 함수는 k=2와 k=3만 지원합니다."
    )


# ============================================================
# 18. 기간별 K-Means 학습 함수
# ============================================================

def fit_period_model(
    period: str,
    number_of_clusters: int,
) -> dict:
    feature_df = feature_frames[period]

    X = feature_df.to_numpy(
        dtype=float,
    )

    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
        max_iter=MAX_ITER,
    )

    labels = model.fit_predict(X)

    distances = model.transform(X)

    assigned_distances = distances[
        np.arange(len(labels)),
        labels,
    ]

    sample_silhouettes = silhouette_samples(
        X,
        labels,
    )

    flow_df = (
        period_station_flow.loc[
            period_station_flow["half_year"]
            == period,
            [
                "station_name",
                "weekday_average",
                "orientation_score",
                "peak_concentration",
            ],
        ]
        .copy()
    )

    label_df = pd.DataFrame(
        {
            "station_name": feature_df.index,
            "cluster_id": labels,
            "distance_to_centroid": (
                assigned_distances
            ),
            "silhouette_sample": (
                sample_silhouettes
            ),
        }
    )

    label_df = label_df.merge(
        flow_df,
        on="station_name",
        how="left",
        validate="one_to_one",
    )

    cluster_name_map = (
        create_cluster_name_map(
            label_df=label_df,
            number_of_clusters=number_of_clusters,
        )
    )

    label_df["cluster_name"] = (
        label_df["cluster_id"]
        .map(cluster_name_map)
    )

    cluster_sizes = np.bincount(
        labels
    )

    metrics = {
        "half_year": period,
        "k": number_of_clusters,
        "silhouette_score": silhouette_score(
            X,
            labels,
        ),
        "inertia": model.inertia_,
        "calinski_harabasz_score": (
            calinski_harabasz_score(
                X,
                labels,
            )
        ),
        "davies_bouldin_score": (
            davies_bouldin_score(
                X,
                labels,
            )
        ),
        "minimum_cluster_size": (
            cluster_sizes.min()
        ),
        "maximum_cluster_size": (
            cluster_sizes.max()
        ),
    }

    return {
        "model": model,
        "labels": labels,
        "label_df": label_df,
        "metrics": metrics,
    }


# ============================================================
# 19. 상·하반기 k=2, k=3 학습
# ============================================================

model_results = {}
model_metric_records = []

for period in PERIOD_ORDER:
    for k in K_VALUES:
        result = fit_period_model(
            period=period,
            number_of_clusters=k,
        )

        model_results[
            (
                period,
                k,
            )
        ] = result

        model_metric_records.append(
            result["metrics"]
        )

model_metrics_df = pd.DataFrame(
    model_metric_records
)


# ============================================================
# 20. 역별 기간 결과 결합
# ============================================================

station_comparison = pd.DataFrame(
    {
        "station_name": station_names,
    }
)

for period in PERIOD_ORDER:
    prefix = period.lower()

    flow_df = (
        period_station_flow.loc[
            period_station_flow["half_year"]
            == period,
            [
                "station_name",
                "weekday_average",
                "orientation_score",
                "peak_concentration",
            ],
        ]
        .rename(
            columns={
                "weekday_average":
                    f"{prefix}_weekday_average",
                "orientation_score":
                    f"{prefix}_orientation_score",
                "peak_concentration":
                    f"{prefix}_peak_concentration",
            }
        )
    )

    station_comparison = (
        station_comparison
        .merge(
            flow_df,
            on="station_name",
            how="left",
            validate="one_to_one",
        )
    )

    for k in K_VALUES:
        label_df = (
            model_results[
                (
                    period,
                    k,
                )
            ]["label_df"]
            [
                [
                    "station_name",
                    "cluster_id",
                    "cluster_name",
                    "distance_to_centroid",
                    "silhouette_sample",
                ]
            ]
            .rename(
                columns={
                    "cluster_id":
                        f"{prefix}_k{k}_cluster_id",
                    "cluster_name":
                        f"{prefix}_k{k}_cluster_name",
                    "distance_to_centroid":
                        f"{prefix}_k{k}_distance",
                    "silhouette_sample":
                        f"{prefix}_k{k}_silhouette",
                }
            )
        )

        station_comparison = (
            station_comparison
            .merge(
                label_df,
                on="station_name",
                how="left",
                validate="one_to_one",
            )
        )


# ============================================================
# 21. 프로파일 변화량 계산
# ============================================================

X_H1 = feature_frames["H1"].to_numpy(
    dtype=float,
)

X_H2 = feature_frames["H2"].to_numpy(
    dtype=float,
)

profile_difference = (
    X_H2 - X_H1
)

profile_shift_l2 = np.linalg.norm(
    profile_difference,
    axis=1,
)

# 두 확률형 프로파일 간 전체 변화량
total_variation_distance = (
    0.5
    * np.abs(profile_difference).sum(axis=1)
)

h1_norm = np.linalg.norm(
    X_H1,
    axis=1,
)

h2_norm = np.linalg.norm(
    X_H2,
    axis=1,
)

cosine_similarity = (
    np.sum(
        X_H1 * X_H2,
        axis=1,
    )
    / (
        h1_norm
        * h2_norm
    )
)

station_comparison[
    "profile_shift_l2"
] = profile_shift_l2

station_comparison[
    "total_variation_distance"
] = total_variation_distance

station_comparison[
    "profile_cosine_similarity"
] = cosine_similarity

station_comparison[
    "orientation_score_change"
] = (
    station_comparison["h2_orientation_score"]
    - station_comparison["h1_orientation_score"]
)

for k in K_VALUES:
    station_comparison[
        f"k{k}_cluster_changed"
    ] = (
        station_comparison[
            f"h1_k{k}_cluster_name"
        ]
        != station_comparison[
            f"h2_k{k}_cluster_name"
        ]
    )


# ============================================================
# 22. 기간 간 안정성 지표 계산
# ============================================================

stability_records = []
transition_matrices = {}

CLUSTER_NAME_ORDERS = {
    2: [
        "주거 유출·귀가 중심",
        "업무 유입 중심",
    ],
    3: [
        "주거 유출·귀가 중심",
        "혼합 흐름 중심",
        "업무 유입 중심",
    ],
}

for k in K_VALUES:
    h1_result = model_results[
        (
            "H1",
            k,
        )
    ]

    h2_result = model_results[
        (
            "H2",
            k,
        )
    ]

    ari = adjusted_rand_score(
        h1_result["labels"],
        h2_result["labels"],
    )

    nmi = normalized_mutual_info_score(
        h1_result["labels"],
        h2_result["labels"],
    )

    h1_names = station_comparison[
        f"h1_k{k}_cluster_name"
    ]

    h2_names = station_comparison[
        f"h2_k{k}_cluster_name"
    ]

    retention_rate = (
        h1_names == h2_names
    ).mean()

    changed_count = (
        h1_names != h2_names
    ).sum()

    stability_records.append(
        {
            "k": k,
            "adjusted_rand_index": ari,
            "normalized_mutual_info": nmi,
            "semantic_retention_rate": (
                retention_rate
            ),
            "changed_station_count": (
                int(changed_count)
            ),
            "unchanged_station_count": (
                int(
                    number_of_stations
                    - changed_count
                )
            ),
        }
    )

    transition_matrix = pd.crosstab(
        h1_names,
        h2_names,
    )

    transition_matrix = (
        transition_matrix
        .reindex(
            index=CLUSTER_NAME_ORDERS[k],
            columns=CLUSTER_NAME_ORDERS[k],
            fill_value=0,
        )
    )

    transition_matrices[k] = (
        transition_matrix
    )

stability_df = pd.DataFrame(
    stability_records
)


# ============================================================
# 23. 군집별 요약 생성
# ============================================================

cluster_summary_records = []

for period in PERIOD_ORDER:
    for k in K_VALUES:
        label_df = model_results[
            (
                period,
                k,
            )
        ]["label_df"]

        summary = (
            label_df
            .groupby(
                "cluster_name",
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
                average_silhouette=(
                    "silhouette_sample",
                    "mean",
                ),
            )
        )

        summary["half_year"] = period
        summary["k"] = k

        cluster_summary_records.append(
            summary
        )

cluster_summary_df = pd.concat(
    cluster_summary_records,
    ignore_index=True,
)

cluster_summary_df = cluster_summary_df[
    [
        "half_year",
        "k",
        "cluster_name",
        "number_of_stations",
        "average_weekday_volume",
        "average_orientation_score",
        "average_peak_concentration",
        "average_silhouette",
    ]
]


# ============================================================
# 24. 변화량 순위 생성
# ============================================================

profile_shift_ranking = (
    station_comparison
    .sort_values(
        "total_variation_distance",
        ascending=False,
    )
    .reset_index(drop=True)
)

profile_shift_ranking[
    "profile_shift_rank"
] = (
    np.arange(
        1,
        len(profile_shift_ranking) + 1,
    )
)

profile_shift_ranking = (
    profile_shift_ranking[
        [
            "profile_shift_rank",
            "station_name",
            "profile_shift_l2",
            "total_variation_distance",
            "profile_cosine_similarity",
            "h1_weekday_average",
            "h2_weekday_average",
            "h1_orientation_score",
            "h2_orientation_score",
            "orientation_score_change",
            "h1_k2_cluster_name",
            "h2_k2_cluster_name",
            "k2_cluster_changed",
            "h1_k3_cluster_name",
            "h2_k3_cluster_name",
            "k3_cluster_changed",
        ]
    ]
)


# ============================================================
# 25. 결과 파일 저장
# ============================================================

MODEL_METRICS_PATH = (
    OUTPUT_DIR
    / "08_halfyear_model_metrics.csv"
)

STABILITY_PATH = (
    OUTPUT_DIR
    / "08_halfyear_cluster_metrics.csv"
)

STATION_COMPARISON_PATH = (
    OUTPUT_DIR
    / "08_station_halfyear_clusters.csv"
)

K2_TRANSITION_PATH = (
    OUTPUT_DIR
    / "08_k2_transition_matrix.csv"
)

K3_TRANSITION_PATH = (
    OUTPUT_DIR
    / "08_k3_transition_matrix.csv"
)

PROFILE_SHIFT_PATH = (
    OUTPUT_DIR
    / "08_profile_shift_ranking.csv"
)

CLUSTER_SUMMARY_PATH = (
    OUTPUT_DIR
    / "08_halfyear_cluster_summary.csv"
)

TIME_PROFILE_PATH = (
    OUTPUT_DIR
    / "08_halfyear_time_profiles.csv"
)

model_metrics_df.to_csv(
    MODEL_METRICS_PATH,
    index=False,
    encoding="utf-8-sig",
)

stability_df.to_csv(
    STABILITY_PATH,
    index=False,
    encoding="utf-8-sig",
)

station_comparison.to_csv(
    STATION_COMPARISON_PATH,
    index=False,
    encoding="utf-8-sig",
)

transition_matrices[2].to_csv(
    K2_TRANSITION_PATH,
    encoding="utf-8-sig",
)

transition_matrices[3].to_csv(
    K3_TRANSITION_PATH,
    encoding="utf-8-sig",
)

profile_shift_ranking.to_csv(
    PROFILE_SHIFT_PATH,
    index=False,
    encoding="utf-8-sig",
)

cluster_summary_df.to_csv(
    CLUSTER_SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig",
)

station_slot_profile.to_csv(
    TIME_PROFILE_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 26. 안정성 비교 그래프
# ============================================================

plot_df = stability_df.sort_values(
    "k"
)

x_positions = np.arange(
    len(plot_df)
)

bar_width = 0.35

plt.figure(figsize=(9, 6))

plt.bar(
    x_positions - bar_width / 2,
    plot_df["adjusted_rand_index"],
    width=bar_width,
    label="Adjusted Rand Index",
)

plt.bar(
    x_positions + bar_width / 2,
    plot_df["semantic_retention_rate"],
    width=bar_width,
    label="군집 유지율",
)

plt.xticks(
    x_positions,
    [
        f"k={k}"
        for k in plot_df["k"]
    ],
)

plt.ylim(0, 1.05)
plt.title(
    "상·하반기 시간대 패턴 군집 안정성"
)
plt.xlabel("군집 수")
plt.ylabel("안정성 지표")
plt.grid(
    axis="y",
    alpha=0.3,
)
plt.legend()
plt.tight_layout()

STABILITY_GRAPH_PATH = (
    OUTPUT_DIR
    / "08_halfyear_cluster_stability.png"
)

plt.savefig(
    STABILITY_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 27. 전환 행렬 그래프 함수
# ============================================================

def save_transition_heatmap(
    transition_matrix: pd.DataFrame,
    title: str,
    output_path: Path,
) -> None:
    values = transition_matrix.to_numpy()

    plt.figure(figsize=(9, 7))

    image = plt.imshow(
        values,
        aspect="auto",
    )

    plt.colorbar(
        image,
        label="역 수",
    )

    plt.xticks(
        np.arange(
            len(transition_matrix.columns)
        ),
        transition_matrix.columns,
        rotation=25,
        ha="right",
    )

    plt.yticks(
        np.arange(
            len(transition_matrix.index)
        ),
        transition_matrix.index,
    )

    plt.xlabel("하반기 군집")
    plt.ylabel("상반기 군집")
    plt.title(title)

    maximum_value = (
        values.max()
        if values.size > 0
        else 0
    )

    for row_index in range(
        values.shape[0]
    ):
        for column_index in range(
            values.shape[1]
        ):
            value = int(
                values[
                    row_index,
                    column_index,
                ]
            )

            text_color = (
                "white"
                if maximum_value > 0
                and value > maximum_value / 2
                else "black"
            )

            plt.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color=text_color,
                fontsize=12,
            )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


K2_HEATMAP_PATH = (
    OUTPUT_DIR
    / "08_k2_transition_heatmap.png"
)

K3_HEATMAP_PATH = (
    OUTPUT_DIR
    / "08_k3_transition_heatmap.png"
)

save_transition_heatmap(
    transition_matrix=transition_matrices[2],
    title="k=2 상반기 → 하반기 군집 전환",
    output_path=K2_HEATMAP_PATH,
)

save_transition_heatmap(
    transition_matrix=transition_matrices[3],
    title="k=3 상반기 → 하반기 군집 전환",
    output_path=K3_HEATMAP_PATH,
)


# ============================================================
# 28. 프로파일 변화량 상위 역 그래프
# ============================================================

top_shift_stations = (
    profile_shift_ranking
    .head(20)
    .sort_values(
        "total_variation_distance",
        ascending=True,
    )
)

plt.figure(figsize=(11, 8))

plt.barh(
    top_shift_stations["station_name"],
    top_shift_stations[
        "total_variation_distance"
    ],
)

plt.title(
    "상·하반기 시간대 승하차 프로파일 변화 상위 20개 역"
)
plt.xlabel(
    "Total Variation Distance"
)
plt.ylabel("역")
plt.grid(
    axis="x",
    alpha=0.3,
)
plt.tight_layout()

PROFILE_SHIFT_GRAPH_PATH = (
    OUTPUT_DIR
    / "08_profile_shift_stations.png"
)

plt.savefig(
    PROFILE_SHIFT_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 29. 방향성 점수 안정성 산점도
# ============================================================

plt.figure(figsize=(9, 8))

plt.scatter(
    station_comparison[
        "h1_orientation_score"
    ],
    station_comparison[
        "h2_orientation_score"
    ],
    alpha=0.7,
)

minimum_score = min(
    station_comparison[
        "h1_orientation_score"
    ].min(),
    station_comparison[
        "h2_orientation_score"
    ].min(),
)

maximum_score = max(
    station_comparison[
        "h1_orientation_score"
    ].max(),
    station_comparison[
        "h2_orientation_score"
    ].max(),
)

plt.plot(
    [
        minimum_score,
        maximum_score,
    ],
    [
        minimum_score,
        maximum_score,
    ],
    linestyle="--",
    label="상·하반기 동일",
)

label_stations = (
    profile_shift_ranking
    .head(10)
)

for _, row in label_stations.iterrows():
    plt.annotate(
        row["station_name"],
        (
            row["h1_orientation_score"],
            row["h2_orientation_score"],
        ),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
    )

plt.title(
    "역별 상·하반기 출퇴근 방향성 점수"
)
plt.xlabel("상반기 방향성 점수")
plt.ylabel("하반기 방향성 점수")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

ORIENTATION_GRAPH_PATH = (
    OUTPUT_DIR
    / "08_orientation_stability_scatter.png"
)

plt.savefig(
    ORIENTATION_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 30. 터미널 출력
# ============================================================

print("\n" + "=" * 70)
print("8. 기간별 군집 모델 평가")
print("=" * 70)

print(
    model_metrics_df
    .round(4)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("9. 상·하반기 군집 안정성")
print("=" * 70)

print(
    stability_df
    .round(4)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("10. k=2 군집 전환표")
print("=" * 70)

print(
    transition_matrices[2]
    .to_string()
)

print("\n" + "=" * 70)
print("11. k=3 군집 전환표")
print("=" * 70)

print(
    transition_matrices[3]
    .to_string()
)

print("\n" + "=" * 70)
print("12. 프로파일 변화 상위 20개 역")
print("=" * 70)

print(
    profile_shift_ranking[
        [
            "profile_shift_rank",
            "station_name",
            "total_variation_distance",
            "profile_cosine_similarity",
            "orientation_score_change",
            "h1_k2_cluster_name",
            "h2_k2_cluster_name",
            "k2_cluster_changed",
        ]
    ]
    .head(20)
    .round(4)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("13. 작업 완료")
print("=" * 70)
print(f"기간별 모델 평가: {MODEL_METRICS_PATH.resolve()}")
print(f"군집 안정성: {STABILITY_PATH.resolve()}")
print(f"역별 비교 결과: {STATION_COMPARISON_PATH.resolve()}")
print(f"k=2 전환표: {K2_TRANSITION_PATH.resolve()}")
print(f"k=3 전환표: {K3_TRANSITION_PATH.resolve()}")
print(f"프로파일 변화 순위: {PROFILE_SHIFT_PATH.resolve()}")
print(f"기간별 군집 요약: {CLUSTER_SUMMARY_PATH.resolve()}")
print(f"기간별 시간 프로파일: {TIME_PROFILE_PATH.resolve()}")
print(f"안정성 그래프: {STABILITY_GRAPH_PATH.resolve()}")
print(f"k=2 전환 그래프: {K2_HEATMAP_PATH.resolve()}")
print(f"k=3 전환 그래프: {K3_HEATMAP_PATH.resolve()}")
print(f"프로파일 변화 그래프: {PROFILE_SHIFT_GRAPH_PATH.resolve()}")
print(f"방향성 안정성 그래프: {ORIENTATION_GRAPH_PATH.resolve()}")