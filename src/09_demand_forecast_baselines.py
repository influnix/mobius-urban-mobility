from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rcParams


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
# 2. 파일 경로와 분석 설정
# ============================================================

STATION_DAILY_PATH = Path(
    "outputs/04_station_daily.csv"
)

STATION_PROFILE_PATH = Path(
    "outputs/04_station_daytype_profile.csv"
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_START_DATE = pd.Timestamp(
    "2025-10-01"
)

TEST_END_DATE = pd.Timestamp(
    "2025-12-31"
)

EXPECTED_NUMBER_OF_STATIONS = 240
EXPECTED_NUMBER_OF_DATES = 365
EXPECTED_NUMBER_OF_ROWS = 87_600
EXPECTED_NUMBER_OF_TEST_DATES = 92
EXPECTED_NUMBER_OF_TEST_ROWS = 22_080


# ============================================================
# 3. 입력 파일 확인
# ============================================================

for file_path in [
    STATION_DAILY_PATH,
    STATION_PROFILE_PATH,
]:
    if not file_path.exists():
        raise FileNotFoundError(
            "\n필요한 파일을 찾지 못했습니다.\n"
            f"확인할 위치: {file_path.resolve()}\n"
            "01~04단계 분석을 먼저 실행했는지 확인하세요."
        )


# ============================================================
# 4. 역·날짜 데이터 불러오기
# ============================================================

daily_df = pd.read_csv(
    STATION_DAILY_PATH,
    encoding="utf-8-sig",
)

required_daily_columns = [
    "date",
    "station_name",
    "passenger_count",
    "weekday",
    "is_weekend",
    "is_holiday",
    "holiday_name",
    "day_type",
]

missing_daily_columns = [
    column
    for column in required_daily_columns
    if column not in daily_df.columns
]

if missing_daily_columns:
    raise ValueError(
        "\n역·날짜 데이터에 필요한 컬럼이 없습니다.\n"
        f"누락 컬럼: {missing_daily_columns}\n"
    )

daily_df["date"] = pd.to_datetime(
    daily_df["date"],
    errors="coerce",
)

daily_df = (
    daily_df
    .sort_values(
        [
            "station_name",
            "date",
        ]
    )
    .reset_index(drop=True)
)

print("=" * 70)
print("1. 역·날짜 데이터 불러오기")
print("=" * 70)
print(f"행 수: {len(daily_df):,}")
print(
    "고유 역 수: "
    f"{daily_df['station_name'].nunique():,}"
)
print(
    "고유 날짜 수: "
    f"{daily_df['date'].nunique():,}"
)
print(
    "날짜 범위: "
    f"{daily_df['date'].min().date()} "
    f"~ {daily_df['date'].max().date()}"
)


# ============================================================
# 5. 기본 데이터 검증
# ============================================================

missing_value_count = (
    daily_df[
        [
            "date",
            "station_name",
            "passenger_count",
            "day_type",
        ]
    ]
    .isna()
    .any(axis=1)
    .sum()
)

negative_value_count = (
    daily_df["passenger_count"] < 0
).sum()

duplicate_count = (
    daily_df.duplicated(
        subset=[
            "station_name",
            "date",
        ]
    )
    .sum()
)

station_date_counts = (
    daily_df
    .groupby("station_name")["date"]
    .nunique()
)

incomplete_station_count = (
    station_date_counts
    != EXPECTED_NUMBER_OF_DATES
).sum()

print("\n" + "=" * 70)
print("2. 기본 데이터 검증")
print("=" * 70)
print(f"필수 값 결측 행 수: {missing_value_count:,}")
print(f"음수 이용량 행 수: {negative_value_count:,}")
print(f"역·날짜 중복 행 수: {duplicate_count:,}")
print(
    "역별 최소 관측일: "
    f"{station_date_counts.min():,}"
)
print(
    "역별 최대 관측일: "
    f"{station_date_counts.max():,}"
)
print(
    "관측일이 불완전한 역 수: "
    f"{incomplete_station_count:,}"
)

if len(daily_df) != EXPECTED_NUMBER_OF_ROWS:
    raise ValueError(
        "\n전체 행 수가 예상과 다릅니다.\n"
        f"예상: {EXPECTED_NUMBER_OF_ROWS:,}\n"
        f"실제: {len(daily_df):,}\n"
    )

if (
    daily_df["station_name"].nunique()
    != EXPECTED_NUMBER_OF_STATIONS
):
    raise ValueError(
        "고유 역 수가 240개가 아닙니다."
    )

if (
    daily_df["date"].nunique()
    != EXPECTED_NUMBER_OF_DATES
):
    raise ValueError(
        "고유 날짜 수가 365일이 아닙니다."
    )

if missing_value_count > 0:
    raise ValueError(
        "필수 값에 결측치가 있습니다."
    )

if negative_value_count > 0:
    raise ValueError(
        "이용량에 음수가 있습니다."
    )

if duplicate_count > 0:
    raise ValueError(
        "역·날짜 중복 행이 있습니다."
    )

if incomplete_station_count > 0:
    raise ValueError(
        "일부 역의 날짜 데이터가 불완전합니다."
    )


# ============================================================
# 6. 4단계 역 유형 정보 결합
# ============================================================

station_profile = pd.read_csv(
    STATION_PROFILE_PATH,
    encoding="utf-8-sig",
)

required_profile_columns = [
    "station_name",
    "station_type",
    "weekday_average",
    "weekend_average",
    "relative_weekend_index",
]

missing_profile_columns = [
    column
    for column in required_profile_columns
    if column not in station_profile.columns
]

if missing_profile_columns:
    raise ValueError(
        "\n4단계 역 프로파일에 필요한 컬럼이 없습니다.\n"
        f"누락 컬럼: {missing_profile_columns}\n"
    )

station_profile = station_profile[
    required_profile_columns
].copy()

profile_duplicate_count = (
    station_profile["station_name"]
    .duplicated()
    .sum()
)

if profile_duplicate_count > 0:
    raise ValueError(
        "4단계 역 프로파일에 중복 역이 있습니다."
    )

daily_df = daily_df.merge(
    station_profile,
    on="station_name",
    how="left",
    validate="many_to_one",
)

profile_merge_failure_count = (
    daily_df["station_type"]
    .isna()
    .sum()
)

print("\n" + "=" * 70)
print("3. 역 프로파일 결합")
print("=" * 70)
print(
    "역 프로파일 결합 실패 행 수: "
    f"{profile_merge_failure_count:,}"
)

if profile_merge_failure_count > 0:
    raise ValueError(
        "일부 역이 4단계 프로파일과 결합되지 않았습니다."
    )


# ============================================================
# 7. 누수 없는 과거 시차 변수 생성
# ============================================================

station_group = daily_df.groupby(
    "station_name",
    sort=False,
)

daily_df["lag_1"] = (
    station_group["passenger_count"]
    .shift(1)
)

daily_df["lag_7"] = (
    station_group["passenger_count"]
    .shift(7)
)

daily_df["lag_14"] = (
    station_group["passenger_count"]
    .shift(14)
)

daily_df["lag_21"] = (
    station_group["passenger_count"]
    .shift(21)
)

daily_df["lag_28"] = (
    station_group["passenger_count"]
    .shift(28)
)

four_week_lag_columns = [
    "lag_7",
    "lag_14",
    "lag_21",
    "lag_28",
]

daily_df[
    "same_weekday_4week_mean"
] = (
    daily_df[
        four_week_lag_columns
    ]
    .sum(
        axis=1,
        min_count=4,
    )
    / 4
)

baseline_column_map = {
    "previous_day": "lag_1",
    "same_weekday_last_week": "lag_7",
    "same_weekday_4week_mean":
        "same_weekday_4week_mean",
}

print("\n" + "=" * 70)
print("4. 기준모델 시차 변수 생성")
print("=" * 70)

for model_name, column_name in (
    baseline_column_map.items()
):
    available_count = (
        daily_df[column_name]
        .notna()
        .sum()
    )

    print(
        f"{model_name} 사용 가능 행 수: "
        f"{available_count:,}"
    )


# ============================================================
# 8. 백테스트 기간 선택
# ============================================================

test_df = daily_df.loc[
    daily_df["date"].between(
        TEST_START_DATE,
        TEST_END_DATE,
    )
].copy()

number_of_test_dates = (
    test_df["date"].nunique()
)

number_of_test_stations = (
    test_df["station_name"].nunique()
)

print("\n" + "=" * 70)
print("5. 백테스트 데이터")
print("=" * 70)
print(
    "백테스트 기간: "
    f"{TEST_START_DATE.date()} "
    f"~ {TEST_END_DATE.date()}"
)
print(
    "백테스트 날짜 수: "
    f"{number_of_test_dates:,}"
)
print(
    "백테스트 역 수: "
    f"{number_of_test_stations:,}"
)
print(
    "백테스트 행 수: "
    f"{len(test_df):,}"
)

if number_of_test_dates != EXPECTED_NUMBER_OF_TEST_DATES:
    raise ValueError(
        "\n백테스트 날짜 수가 예상과 다릅니다.\n"
        f"예상: {EXPECTED_NUMBER_OF_TEST_DATES}\n"
        f"실제: {number_of_test_dates}\n"
    )

if number_of_test_stations != EXPECTED_NUMBER_OF_STATIONS:
    raise ValueError(
        "백테스트 역 수가 240개가 아닙니다."
    )

if len(test_df) != EXPECTED_NUMBER_OF_TEST_ROWS:
    raise ValueError(
        "\n백테스트 행 수가 예상과 다릅니다.\n"
        f"예상: {EXPECTED_NUMBER_OF_TEST_ROWS:,}\n"
        f"실제: {len(test_df):,}\n"
    )


# ============================================================
# 9. 모델별 예측 결과를 Long 형태로 변환
# ============================================================

prediction_frames = []

base_prediction_columns = [
    "date",
    "station_name",
    "passenger_count",
    "weekday",
    "is_weekend",
    "is_holiday",
    "holiday_name",
    "day_type",
    "station_type",
    "weekday_average",
    "weekend_average",
    "relative_weekend_index",
]

for model_name, column_name in (
    baseline_column_map.items()
):
    model_predictions = test_df[
        base_prediction_columns
        + [
            column_name,
        ]
    ].copy()

    model_predictions = (
        model_predictions
        .rename(
            columns={
                "passenger_count": "actual",
                column_name: "prediction",
            }
        )
    )

    model_predictions[
        "model_name"
    ] = model_name

    prediction_frames.append(
        model_predictions
    )

predictions_df = pd.concat(
    prediction_frames,
    ignore_index=True,
)

missing_prediction_count = (
    predictions_df["prediction"]
    .isna()
    .sum()
)

print(
    "전체 모델 예측 행 수: "
    f"{len(predictions_df):,}"
)
print(
    "예측값 결측 행 수: "
    f"{missing_prediction_count:,}"
)

for model_name in baseline_column_map:
    model_row_count = len(
        predictions_df.loc[
            predictions_df["model_name"]
            == model_name
        ]
    )

    print(
        f"{model_name} 예측 행 수: "
        f"{model_row_count:,}"
    )

    if (
        model_row_count
        != EXPECTED_NUMBER_OF_TEST_ROWS
    ):
        raise ValueError(
            f"{model_name} 예측 행 수가 "
            "예상과 다릅니다."
        )

if missing_prediction_count > 0:
    raise ValueError(
        "백테스트 구간에 결측 예측값이 있습니다."
    )


# ============================================================
# 10. 오차 변수 생성
# ============================================================

predictions_df["error"] = (
    predictions_df["prediction"]
    - predictions_df["actual"]
)

predictions_df["residual"] = (
    predictions_df["actual"]
    - predictions_df["prediction"]
)

predictions_df["absolute_error"] = (
    predictions_df["error"].abs()
)

predictions_df["squared_error"] = (
    predictions_df["error"] ** 2
)

# 양수이면 실제 수요보다 낮게 예측한 경우
predictions_df["underprediction"] = (
    predictions_df["actual"]
    - predictions_df["prediction"]
)

predictions_df[
    "is_underprediction"
] = (
    predictions_df["prediction"]
    < predictions_df["actual"]
)


# ============================================================
# 11. 평가 지표 계산 함수
# ============================================================

def calculate_metrics(
    dataframe: pd.DataFrame,
) -> dict:
    """
    예측 결과 데이터프레임의 평가 지표를 계산한다.
    """
    actual = dataframe["actual"].to_numpy(
        dtype=float,
    )

    prediction = (
        dataframe["prediction"]
        .to_numpy(
            dtype=float,
        )
    )

    absolute_error = np.abs(
        prediction - actual
    )

    squared_error = (
        prediction - actual
    ) ** 2

    actual_sum = actual.sum()

    smape_denominator = (
        np.abs(actual)
        + np.abs(prediction)
    )

    smape_values = np.where(
        smape_denominator > 0,
        (
            2
            * absolute_error
            / smape_denominator
        ),
        0,
    )

    return {
        "number_of_predictions": len(dataframe),
        "actual_total": actual_sum,
        "prediction_total": prediction.sum(),
        "mae": absolute_error.mean(),
        "rmse": np.sqrt(
            squared_error.mean()
        ),
        "wape_pct": (
            absolute_error.sum()
            / actual_sum
            * 100
            if actual_sum > 0
            else np.nan
        ),
        "smape_pct": (
            smape_values.mean()
            * 100
        ),
        "bias_pct": (
            (
                prediction.sum()
                - actual_sum
            )
            / actual_sum
            * 100
            if actual_sum > 0
            else np.nan
        ),
        "underprediction_rate_pct": (
            (
                prediction
                < actual
            )
            .mean()
            * 100
        ),
    }


# ============================================================
# 12. 전체 모델 평가
# ============================================================

model_metric_records = []

for model_name, model_df in (
    predictions_df.groupby(
        "model_name",
        sort=True,
    )
):
    metrics = calculate_metrics(
        model_df
    )

    metrics["model_name"] = model_name

    model_metric_records.append(
        metrics
    )

model_metrics_df = pd.DataFrame(
    model_metric_records
)

model_metrics_df = model_metrics_df[
    [
        "model_name",
        "number_of_predictions",
        "actual_total",
        "prediction_total",
        "mae",
        "rmse",
        "wape_pct",
        "smape_pct",
        "bias_pct",
        "underprediction_rate_pct",
    ]
]

model_metrics_df = (
    model_metrics_df
    .sort_values(
        "wape_pct"
    )
    .reset_index(drop=True)
)

best_model_name = (
    model_metrics_df.iloc[0][
        "model_name"
    ]
)

print("\n" + "=" * 70)
print("6. 전체 기준모델 평가")
print("=" * 70)

print(
    model_metrics_df
    .round(3)
    .to_string(index=False)
)

print(
    "\nWAPE 기준 최적 기준모델: "
    f"{best_model_name}"
)


# ============================================================
# 13. 날짜 유형별 평가
# ============================================================

day_type_metric_records = []

for (
    model_name,
    day_type,
), segment_df in (
    predictions_df.groupby(
        [
            "model_name",
            "day_type",
        ],
        sort=True,
        observed=True,
    )
):
    metrics = calculate_metrics(
        segment_df
    )

    metrics["model_name"] = model_name
    metrics["day_type"] = day_type

    day_type_metric_records.append(
        metrics
    )

day_type_metrics_df = pd.DataFrame(
    day_type_metric_records
)

day_type_metrics_df = day_type_metrics_df[
    [
        "model_name",
        "day_type",
        "number_of_predictions",
        "mae",
        "rmse",
        "wape_pct",
        "smape_pct",
        "bias_pct",
        "underprediction_rate_pct",
    ]
]

day_type_metrics_df = (
    day_type_metrics_df
    .sort_values(
        [
            "day_type",
            "wape_pct",
        ]
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 70)
print("7. 날짜 유형별 평가")
print("=" * 70)

print(
    day_type_metrics_df
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 14. 역 유형별 평가
# ============================================================

station_type_metric_records = []

for (
    model_name,
    station_type,
), segment_df in (
    predictions_df.groupby(
        [
            "model_name",
            "station_type",
        ],
        sort=True,
        observed=True,
    )
):
    metrics = calculate_metrics(
        segment_df
    )

    metrics["model_name"] = model_name
    metrics["station_type"] = station_type

    station_type_metric_records.append(
        metrics
    )

station_type_metrics_df = pd.DataFrame(
    station_type_metric_records
)

station_type_metrics_df = (
    station_type_metrics_df[
        [
            "model_name",
            "station_type",
            "number_of_predictions",
            "mae",
            "rmse",
            "wape_pct",
            "smape_pct",
            "bias_pct",
            "underprediction_rate_pct",
        ]
    ]
    .sort_values(
        [
            "station_type",
            "wape_pct",
        ]
    )
    .reset_index(drop=True)
)

print("\n" + "=" * 70)
print("8. 역 유형별 평가")
print("=" * 70)

print(
    station_type_metrics_df
    .round(3)
    .to_string(index=False)
)


# ============================================================
# 15. 역별 평가
# ============================================================

station_metric_records = []

for (
    model_name,
    station_name,
), station_df in (
    predictions_df.groupby(
        [
            "model_name",
            "station_name",
        ],
        sort=True,
    )
):
    metrics = calculate_metrics(
        station_df
    )

    metrics["model_name"] = model_name
    metrics["station_name"] = station_name
    metrics["station_type"] = (
        station_df["station_type"]
        .iloc[0]
    )

    station_metric_records.append(
        metrics
    )

station_metrics_df = pd.DataFrame(
    station_metric_records
)

station_metrics_df = station_metrics_df[
    [
        "model_name",
        "station_name",
        "station_type",
        "number_of_predictions",
        "mae",
        "rmse",
        "wape_pct",
        "smape_pct",
        "bias_pct",
        "underprediction_rate_pct",
    ]
]

station_metrics_df = (
    station_metrics_df
    .sort_values(
        [
            "model_name",
            "wape_pct",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# 16. 최적 기준모델의 네트워크 일별 결과
# ============================================================

best_predictions = (
    predictions_df.loc[
        predictions_df["model_name"]
        == best_model_name
    ]
    .copy()
)

network_daily = (
    best_predictions
    .groupby(
        [
            "date",
            "day_type",
            "holiday_name",
        ],
        as_index=False,
        dropna=False,
    )
    .agg(
        actual_total=(
            "actual",
            "sum",
        ),
        prediction_total=(
            "prediction",
            "sum",
        ),
        absolute_error_total=(
            "absolute_error",
            "sum",
        ),
    )
)

network_daily["error"] = (
    network_daily["prediction_total"]
    - network_daily["actual_total"]
)

network_daily["absolute_error"] = (
    network_daily["error"].abs()
)

network_daily["absolute_error_pct"] = (
    network_daily["absolute_error"]
    / network_daily["actual_total"]
    * 100
)

network_daily["model_name"] = (
    best_model_name
)

worst_dates = (
    network_daily
    .sort_values(
        "absolute_error_pct",
        ascending=False,
    )
    .reset_index(drop=True)
)

worst_dates["error_rank"] = (
    np.arange(
        1,
        len(worst_dates) + 1,
    )
)


# ============================================================
# 17. 과소예측과 절대오차 상위 사례
# ============================================================

top_underpredictions = (
    best_predictions.loc[
        best_predictions[
            "underprediction"
        ] > 0
    ]
    .nlargest(
        100,
        "underprediction",
    )
    .copy()
)

top_absolute_errors = (
    best_predictions
    .nlargest(
        100,
        "absolute_error",
    )
    .copy()
)

print("\n" + "=" * 70)
print("9. 최적 기준모델의 오차 상위 날짜")
print("=" * 70)

worst_dates_display = (
    worst_dates[
        [
            "error_rank",
            "date",
            "day_type",
            "holiday_name",
            "actual_total",
            "prediction_total",
            "error",
            "absolute_error_pct",
        ]
    ]
    .head(20)
    .copy()
)

worst_date_numeric_columns = [
    "actual_total",
    "prediction_total",
    "error",
    "absolute_error_pct",
]

worst_dates_display[
    worst_date_numeric_columns
] = (
    worst_dates_display[
        worst_date_numeric_columns
    ]
    .round(2)
)

print(
    worst_dates_display.to_string(
        index=False,
    )
)

print("\n" + "=" * 70)
print("10. 최적 기준모델의 과소예측 상위 20건")
print("=" * 70)

underprediction_display = (
    top_underpredictions[
        [
            "date",
            "station_name",
            "day_type",
            "holiday_name",
            "actual",
            "prediction",
            "underprediction",
        ]
    ]
    .head(20)
    .copy()
)

underprediction_numeric_columns = [
    "actual",
    "prediction",
    "underprediction",
]

underprediction_display[
    underprediction_numeric_columns
] = (
    underprediction_display[
        underprediction_numeric_columns
    ]
    .round(2)
)

print(
    underprediction_display.to_string(
        index=False,
    )
)


# ============================================================
# 18. 결과 파일 저장
# ============================================================

MODEL_METRICS_PATH = (
    OUTPUT_DIR
    / "09_baseline_model_metrics.csv"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "09_baseline_predictions.csv"
)

DAY_TYPE_METRICS_PATH = (
    OUTPUT_DIR
    / "09_day_type_metrics.csv"
)

STATION_TYPE_METRICS_PATH = (
    OUTPUT_DIR
    / "09_station_type_metrics.csv"
)

STATION_METRICS_PATH = (
    OUTPUT_DIR
    / "09_station_metrics.csv"
)

NETWORK_DAILY_PATH = (
    OUTPUT_DIR
    / "09_network_daily_forecast.csv"
)

WORST_DATES_PATH = (
    OUTPUT_DIR
    / "09_worst_dates.csv"
)

UNDERPREDICTION_PATH = (
    OUTPUT_DIR
    / "09_top_underpredictions.csv"
)

ABSOLUTE_ERROR_PATH = (
    OUTPUT_DIR
    / "09_top_absolute_errors.csv"
)

model_metrics_df.to_csv(
    MODEL_METRICS_PATH,
    index=False,
    encoding="utf-8-sig",
)

predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False,
    encoding="utf-8-sig",
)

day_type_metrics_df.to_csv(
    DAY_TYPE_METRICS_PATH,
    index=False,
    encoding="utf-8-sig",
)

station_type_metrics_df.to_csv(
    STATION_TYPE_METRICS_PATH,
    index=False,
    encoding="utf-8-sig",
)

station_metrics_df.to_csv(
    STATION_METRICS_PATH,
    index=False,
    encoding="utf-8-sig",
)

network_daily.to_csv(
    NETWORK_DAILY_PATH,
    index=False,
    encoding="utf-8-sig",
)

worst_dates.to_csv(
    WORST_DATES_PATH,
    index=False,
    encoding="utf-8-sig",
)

top_underpredictions.to_csv(
    UNDERPREDICTION_PATH,
    index=False,
    encoding="utf-8-sig",
)

top_absolute_errors.to_csv(
    ABSOLUTE_ERROR_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 19. 기준모델 WAPE 비교 그래프
# ============================================================

plot_metrics = (
    model_metrics_df
    .sort_values(
        "wape_pct",
        ascending=False,
    )
)

plt.figure(figsize=(9, 6))

bars = plt.barh(
    plot_metrics["model_name"],
    plot_metrics["wape_pct"],
)

for bar, value in zip(
    bars,
    plot_metrics["wape_pct"],
):
    plt.text(
        value,
        bar.get_y()
        + bar.get_height() / 2,
        f" {value:.2f}%",
        va="center",
    )

plt.title(
    "역·날짜 수요 예측 기준모델 WAPE 비교"
)
plt.xlabel("WAPE (%)")
plt.ylabel("기준모델")
plt.grid(
    axis="x",
    alpha=0.3,
)
plt.tight_layout()

WAPE_GRAPH_PATH = (
    OUTPUT_DIR
    / "09_baseline_wape_comparison.png"
)

plt.savefig(
    WAPE_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 20. 네트워크 실제값과 예측값 그래프
# ============================================================

plt.figure(figsize=(15, 7))

plt.plot(
    network_daily["date"],
    network_daily["actual_total"],
    label="실제 이용량",
)

plt.plot(
    network_daily["date"],
    network_daily["prediction_total"],
    label=(
        f"예측 이용량: {best_model_name}"
    ),
)

plt.title(
    "2025년 4분기 네트워크 일일 이용량 백테스트"
)
plt.xlabel("날짜")
plt.ylabel("전체 이용량")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

NETWORK_GRAPH_PATH = (
    OUTPUT_DIR
    / "09_network_daily_forecast.png"
)

plt.savefig(
    NETWORK_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 21. 날짜 유형별 WAPE 그래프
# ============================================================

day_type_plot = (
    day_type_metrics_df
    .pivot(
        index="day_type",
        columns="model_name",
        values="wape_pct",
    )
)

day_type_plot.plot(
    kind="bar",
    figsize=(11, 7),
)

plt.title(
    "날짜 유형별 기준모델 WAPE"
)
plt.xlabel("날짜 유형")
plt.ylabel("WAPE (%)")
plt.xticks(
    rotation=0,
)
plt.grid(
    axis="y",
    alpha=0.3,
)
plt.legend(
    title="기준모델",
)
plt.tight_layout()

DAY_TYPE_GRAPH_PATH = (
    OUTPUT_DIR
    / "09_day_type_wape.png"
)

plt.savefig(
    DAY_TYPE_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 22. 오차 상위 날짜 그래프
# ============================================================

top_worst_dates = (
    worst_dates
    .head(20)
    .sort_values(
        "absolute_error_pct",
        ascending=True,
    )
    .copy()
)

top_worst_dates[
    "date_label"
] = (
    top_worst_dates["date"]
    .dt.strftime("%Y-%m-%d")
)

plt.figure(figsize=(11, 8))

plt.barh(
    top_worst_dates["date_label"],
    top_worst_dates[
        "absolute_error_pct"
    ],
)

plt.title(
    f"{best_model_name} 오차율 상위 20개 날짜"
)
plt.xlabel("네트워크 절대 오차율 (%)")
plt.ylabel("날짜")
plt.grid(
    axis="x",
    alpha=0.3,
)
plt.tight_layout()

WORST_DATE_GRAPH_PATH = (
    OUTPUT_DIR
    / "09_worst_dates.png"
)

plt.savefig(
    WORST_DATE_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 23. 작업 완료 출력
# ============================================================

print("\n" + "=" * 70)
print("11. 작업 완료")
print("=" * 70)
print(f"전체 모델 평가: {MODEL_METRICS_PATH.resolve()}")
print(f"전체 예측 결과: {PREDICTIONS_PATH.resolve()}")
print(f"날짜 유형별 평가: {DAY_TYPE_METRICS_PATH.resolve()}")
print(f"역 유형별 평가: {STATION_TYPE_METRICS_PATH.resolve()}")
print(f"역별 평가: {STATION_METRICS_PATH.resolve()}")
print(f"네트워크 일별 결과: {NETWORK_DAILY_PATH.resolve()}")
print(f"오차 상위 날짜: {WORST_DATES_PATH.resolve()}")
print(f"과소예측 사례: {UNDERPREDICTION_PATH.resolve()}")
print(f"절대오차 사례: {ABSOLUTE_ERROR_PATH.resolve()}")
print(f"WAPE 비교 그래프: {WAPE_GRAPH_PATH.resolve()}")
print(f"네트워크 예측 그래프: {NETWORK_GRAPH_PATH.resolve()}")
print(f"날짜 유형 그래프: {DAY_TYPE_GRAPH_PATH.resolve()}")
print(f"오차 상위 날짜 그래프: {WORST_DATE_GRAPH_PATH.resolve()}")