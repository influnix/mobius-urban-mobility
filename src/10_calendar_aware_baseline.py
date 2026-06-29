from pathlib import Path

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from matplotlib import font_manager
from matplotlib import rcParams


DATA_PATH = Path(
    "outputs/04_station_daily.csv"
)

EXPECTED_NUMBER_OF_ROWS = 87_600
EXPECTED_NUMBER_OF_STATIONS = 240
EXPECTED_NUMBER_OF_DATES = 365


def load_and_validate_data():
    """
    04_station_daily.csv를 읽고
    날짜·역·이용량·day_type 구조를 검증한다.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {DATA_PATH.resolve()}"
        )

    dataframe = pd.read_csv(
        DATA_PATH,
        encoding="utf-8-sig",
    )

    required_columns = [
        "date",
        "station_name",
        "passenger_count",
        "day_type",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"필수 컬럼이 없습니다: {missing_columns}"
        )

    dataframe["date"] = pd.to_datetime(
        dataframe["date"],
        errors="coerce",
    )

    dataframe = (
        dataframe
        .sort_values(
            [
                "station_name",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    missing_value_count = (
        dataframe[required_columns]
        .isna()
        .any(axis=1)
        .sum()
    )

    duplicate_count = (
        dataframe
        .duplicated(
            subset=[
                "station_name",
                "date",
            ]
        )
        .sum()
    )

    number_of_rows = len(dataframe)

    number_of_stations = (
        dataframe["station_name"]
        .nunique()
    )

    number_of_dates = (
        dataframe["date"]
        .nunique()
    )

    if missing_value_count > 0:
        raise ValueError(
            f"필수 값 결측 행 수: {missing_value_count}"
        )

    if duplicate_count > 0:
        raise ValueError(
            f"역·날짜 중복 행 수: {duplicate_count}"
        )

    if number_of_rows != EXPECTED_NUMBER_OF_ROWS:
        raise ValueError(
            f"행 수 오류: {number_of_rows:,}"
        )

    if number_of_stations != EXPECTED_NUMBER_OF_STATIONS:
        raise ValueError(
            f"역 수 오류: {number_of_stations:,}"
        )

    if number_of_dates != EXPECTED_NUMBER_OF_DATES:
        raise ValueError(
            f"날짜 수 오류: {number_of_dates:,}"
        )

    print("=" * 70)
    print("1. 데이터 검증")
    print("=" * 70)
    print(f"행 수: {number_of_rows:,}")
    print(f"고유 역 수: {number_of_stations:,}")
    print(f"고유 날짜 수: {number_of_dates:,}")
    print(f"필수 값 결측 행 수: {missing_value_count:,}")
    print(f"역·날짜 중복 행 수: {duplicate_count:,}")

    return dataframe


def add_lag_features(dataframe):
    """
    역별 lag_1, lag_7, lag_7_day_type을 생성한다.
    """

    result = dataframe.copy()

    result = (
        result
        .sort_values(
            [
                "station_name",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    station_group = result.groupby(
        "station_name",
        sort=False,
    )

    result["lag_1"] = (
        station_group["passenger_count"]
        .shift(1)
    )

    result["lag_7"] = (
        station_group["passenger_count"]
        .shift(7)
    )

    result["lag_7_day_type"] = (
        station_group["day_type"]
        .shift(7)
    )

    result[
        "day_type_matches_lag_7"
    ] = (
        result["lag_7_day_type"].notna()
        & result["day_type"].eq(
            result["lag_7_day_type"]
        )
    )

    return result


def add_historical_daytype_statistics(
    dataframe,
    minimum_history=3,
):
    """
    station_name × day_type 그룹에서
    현재 행 이전의 이용량 개수와 중앙값을 계산한다.
    """

    result = dataframe.copy()

    result = (
        result
        .sort_values(
            [
                "station_name",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    group_columns = [
        "station_name",
        "day_type",
    ]

    result[
        "historical_daytype_count"
    ] = (
        result
        .groupby(
            group_columns,
            sort=False,
        )
        .cumcount()
    )

    result[
        "historical_daytype_median"
    ] = (
        result
        .groupby(
            group_columns,
            sort=False,
        )["passenger_count"]
        .transform(
            lambda series: (
                series
                .shift(1)
                .expanding(
                    min_periods=1,
                )
                .median()
            )
        )
    )

    result[
        "has_sufficient_daytype_history"
    ] = (
        result[
            "historical_daytype_count"
        ]
        >= minimum_history
    )

    return result


def build_calendar_aware_prediction(
    dataframe,
    minimum_history=3,
):
    """
    날짜 유형 일치 여부와 과거 표본 수에 따라
    예측값과 prediction_source를 생성한다.
    """

    result = dataframe.copy()

    result["prediction"] = pd.NA
    result["prediction_source"] = pd.NA

    same_type_mask = (
        result[
            "day_type_matches_lag_7"
        ]
        & result["lag_7"].notna()
    )

    median_mask = (
        ~same_type_mask
        & (
            result[
                "historical_daytype_count"
            ]
            >= minimum_history
        )
        & result[
            "historical_daytype_median"
        ].notna()
    )

    fallback_mask = (
        ~same_type_mask
        & ~median_mask
        & result["lag_1"].notna()
    )

    result.loc[
        same_type_mask,
        "prediction",
    ] = result.loc[
        same_type_mask,
        "lag_7"
    ]

    result.loc[
        same_type_mask,
        "prediction_source",
    ] = "lag_7_same_type"

    result.loc[
        median_mask,
        "prediction",
    ] = result.loc[
        median_mask,
        "historical_daytype_median"
    ]

    result.loc[
        median_mask,
        "prediction_source",
    ] = "historical_daytype_median"

    result.loc[
        fallback_mask,
        "prediction",
    ] = result.loc[
        fallback_mask,
        "lag_1"
    ]

    result.loc[
        fallback_mask,
        "prediction_source",
    ] = "lag_1_fallback"

    result["prediction"] = pd.to_numeric(
        result["prediction"],
        errors="coerce",
    )

    return result


def calculate_metrics(
    dataframe,
    prediction_column,
):
    """
    지정한 예측 열에 대해
    MAE, RMSE, WAPE, sMAPE, Bias를 계산한다.
    """

    actual = dataframe[
        "passenger_count"
    ].to_numpy(dtype=float)

    prediction = dataframe[
        prediction_column
    ].to_numpy(dtype=float)

    absolute_error = np.abs(
        prediction - actual
    )

    squared_error = (
        prediction - actual
    ) ** 2

    actual_total = actual.sum()

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
        "mae": absolute_error.mean(),
        "rmse": np.sqrt(
            squared_error.mean()
        ),
        "wape_pct": (
            absolute_error.sum()
            / actual_total
            * 100
        ),
        "smape_pct": (
            smape_values.mean()
            * 100
        ),
        "bias_pct": (
            (
                prediction.sum()
                - actual_total
            )
            / actual_total
            * 100
        ),
    }
    
    
def configure_korean_font():
    """
    Windows 환경에서 맑은 고딕을 설정한다.
    """

    font_path = Path(
        "C:/Windows/Fonts/malgun.ttf"
    )

    if font_path.exists():
        font_name = (
            font_manager
            .FontProperties(
                fname=str(font_path)
            )
            .get_name()
        )

        rcParams["font.family"] = (
            font_name
        )

        rcParams[
            "axes.unicode_minus"
        ] = False
        
        
def save_model_wape_comparison(
    model_metrics_df,
    output_dir,
):
    """
    기존 lag_7과 달력 인식 모델의
    전체 WAPE를 비교하여 저장한다.
    """

    model_order = [
        "same_weekday_last_week",
        "calendar_aware_baseline",
    ]

    display_name_map = {
        "same_weekday_last_week":
            "기존 lag_7",

        "calendar_aware_baseline":
            "달력 인식 기준모델",
    }

    plot_df = (
        model_metrics_df
        .set_index("model_name")
        .loc[model_order]
        .reset_index()
    )

    plot_df["display_name"] = (
        plot_df["model_name"]
        .map(display_name_map)
    )

    figure, axis = plt.subplots(
        figsize=(8, 5),
    )

    bars = axis.bar(
        plot_df["display_name"],
        plot_df["wape_pct"],
    )

    axis.bar_label(
        bars,
        fmt="%.3f%%",
        padding=3,
    )

    axis.set_title(
        "전체 수요예측 WAPE 비교"
    )

    axis.set_xlabel(
        "예측 모델"
    )

    axis.set_ylabel(
        "WAPE (%)"
    )

    axis.set_ylim(
        0,
        plot_df["wape_pct"].max()
        * 1.2,
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    output_path = (
        output_dir
        / "10_model_wape_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def save_day_type_wape_comparison(
    day_type_metrics_df,
    output_dir,
):
    """
    날짜 유형별로 기존 lag_7과
    달력 인식 기준모델의 WAPE를 비교한다.
    """

    day_type_order = [
        "Ordinary weekday",
        "Ordinary weekend",
        "Public holiday",
    ]

    day_type_name_map = {
        "Ordinary weekday": "일반 평일",
        "Ordinary weekend": "일반 주말",
        "Public holiday": "공휴일",
    }

    model_name_map = {
        "same_weekday_last_week":
            "기존 lag_7",

        "calendar_aware_baseline":
            "달력 인식 기준모델",
    }

    plot_df = (
        day_type_metrics_df
        .pivot(
            index="day_type",
            columns="model_name",
            values="wape_pct",
        )
        .loc[day_type_order]
        .rename(
            index=day_type_name_map,
            columns=model_name_map,
        )
    )

    axis = plot_df.plot(
        kind="bar",
        figsize=(9, 6),
    )

    axis.set_title(
        "날짜 유형별 수요예측 WAPE 비교"
    )

    axis.set_xlabel(
        "날짜 유형"
    )

    axis.set_ylabel(
        "WAPE (%)"
    )

    axis.tick_params(
        axis="x",
        rotation=0,
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    for container in axis.containers:
        axis.bar_label(
            container,
            fmt="%.3f%%",
            padding=3,
        )

    axis.legend(
        title="예측 모델"
    )

    figure = axis.get_figure()

    figure.tight_layout()

    output_path = (
        output_dir
        / "10_day_type_wape_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def save_key_dates_error_comparison(
    key_date_result,
    output_dir,
):
    """
    주요 공휴일·연휴 전후 날짜의
    네트워크 절대오차율을 비교한다.
    """

    plot_df = (
        key_date_result
        .copy()
        .sort_values("date")
    )

    plot_df["date_label"] = (
        plot_df["date"]
        .dt.strftime("%m-%d")
    )

    plot_df = (
        plot_df
        .set_index("date_label")
        [
            [
                "baseline_ape_pct",
                "calendar_ape_pct",
            ]
        ]
        .rename(
            columns={
                "baseline_ape_pct": "기존 lag_7",
                "calendar_ape_pct": "달력 인식 기준모델",
            }
        )
    )

    axis = plot_df.plot(
        kind="bar",
        figsize=(11, 6),
    )

    axis.set_title(
        "핵심 날짜별 네트워크 절대오차율 비교"
    )

    axis.set_xlabel(
        "날짜"
    )

    axis.set_ylabel(
        "절대오차율 (%)"
    )

    axis.tick_params(
        axis="x",
        rotation=0,
    )

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    for container in axis.containers:
        axis.bar_label(
            container,
            fmt="%.1f%%",
            padding=3,
        )

    axis.legend(
        title="예측 모델"
    )

    figure = axis.get_figure()
    figure.tight_layout()

    output_path = (
        output_dir
        / "10_key_dates_error_comparison.png"
    )

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path
    


if __name__ == "__main__":
    configure_korean_font()
    
    TEST_START_DATE = pd.Timestamp(
        "2025-10-01"
    )
    
    TEST_END_DATE = pd.Timestamp(
        "2025-12-31"
    )
    
    EXPECTED_TEST_ROWS = 22_080
    
    daily_df = load_and_validate_data()
    daily_df = add_lag_features(
        daily_df,
    )
    daily_df = add_historical_daytype_statistics(
        daily_df,
        minimum_history=3
    )
    daily_df = build_calendar_aware_prediction(
        daily_df,
        minimum_history=3
    )

    check_dates = pd.to_datetime(
        [
            "2025-10-06",
            "2025-10-13",
            "2025-12-25",
            "2025-12-26",
        ]
    )

    check_result = (
        daily_df.loc[
            (
                daily_df["station_name"]
                == "강남"
            )
            & (
                daily_df["date"]
                .isin(check_dates)
            ),
            [
                "date",
                "station_name",
                "passenger_count",
                "day_type",
                "lag_7",
                "lag_7_day_type",
                "day_type_matches_lag_7",
                "historical_daytype_count",
                "historical_daytype_median",
                "has_sufficient_daytype_history",
                "prediction",
                "prediction_source"
            ],
        ]
        .sort_values("date")
    )

    print("\n" + "=" * 70)
    print("2. 강남 날짜 유형 시차 검사")
    print("=" * 70)
    print(
        check_result.to_string(
            index=False,
        )
    )
    
    target_date = pd.Timestamp(
        "2025-10-13"
    )

    target_station = "강남"

    target_day_type = (
        daily_df.loc[
            (
                daily_df["station_name"]
                == target_station
            )
            & (
                daily_df["date"]
                == target_date
            ),
            "day_type",
        ]
        .iloc[0]
    )

    manual_history = (
        daily_df.loc[
            (
                daily_df["station_name"]
                == target_station
            )
            & (
                daily_df["day_type"]
                == target_day_type
            )
            & (
                daily_df["date"]
                < target_date
            ),
            "passenger_count",
        ]
    )

    calculated_row = (
        daily_df.loc[
            (
                daily_df["station_name"]
                == target_station
            )
            & (
                daily_df["date"]
                == target_date
            )
        ]
        .iloc[0]
    )

    print("\n" + "=" * 70)
    print("3. 과거 동일 날짜 유형 통계 수동 검증")
    print("=" * 70)

    print(
        "수동 과거 관측 수:",
        len(manual_history),
    )

    print(
        "자동 과거 관측 수:",
        int(
            calculated_row[
                "historical_daytype_count"
            ]
        ),
    )

    print(
        "수동 과거 중앙값:",
        manual_history.median(),
    )

    print(
        "자동 과거 중앙값:",
        calculated_row[
            "historical_daytype_median"
        ],
    )
    
    test_df = daily_df.loc[
        daily_df["date"].between(
            TEST_START_DATE,
            TEST_END_DATE,
        )
    ].copy()
    
    test_df[
        "baseline_prediction"
    ] = test_df["lag_7"]

    test_df[
        "calendar_prediction"
    ] = test_df["prediction"]
    
    baseline_missing_count = (
        test_df["baseline_prediction"]
        .isna()
        .sum()
    )

    if baseline_missing_count > 0:
        raise ValueError(
            "기존 lag_7 기준모델에 결측값이 있습니다."
        )

    missing_prediction_count = (
        test_df["prediction"]
        .isna()
        .sum()
    )

    negative_prediction_count = (
        test_df["prediction"] < 0
    ).sum()

    source_counts = (
        test_df["prediction_source"]
        .value_counts(
            dropna=False,
        )
    )

    print("\n" + "=" * 70)
    print("4. 백테스트 예측 구조 검증")
    print("=" * 70)
    print(f"백테스트 행 수: {len(test_df):,}")
    print(
        "결측 예측값 수:",
        missing_prediction_count,
    )
    print(
        "음수 예측값 수:",
        negative_prediction_count,
    )
    print("\n예측값 출처:")
    print(source_counts.to_string())

    if len(test_df) != EXPECTED_TEST_ROWS:
        raise ValueError(
            f"백테스트 행 수 오류: {len(test_df):,}"
        )

    if missing_prediction_count > 0:
        raise ValueError(
            "백테스트 구간에 결측 예측값이 있습니다."
        )

    if negative_prediction_count > 0:
        raise ValueError(
            "백테스트 구간에 음수 예측값이 있습니다."
        )

    if source_counts.sum() != EXPECTED_TEST_ROWS:
        raise ValueError(
            "prediction_source 합계가 백테스트 행 수와 다릅니다."
        )
        
    fallback_examples = (
        daily_df.loc[
            daily_df["prediction_source"]
            == "lag_1_fallback",
            [
                "date",
                "station_name",
                "day_type",
                "lag_1",
                "lag_7",
                "historical_daytype_count",
                "historical_daytype_median",
                "prediction",
                "prediction_source",
            ],
        ]
        .head(10)
    )            
        
    print("\n" + "=" * 70)
    print("5. lag_1 fallback 동작 확인")
    print("=" * 70)

    if fallback_examples.empty:
        print("lag_1 fallback 사례가 없습니다.")
    else:
        print(
            fallback_examples.to_string(
                index=False,
            )
        )
        
    model_columns = {
        "same_weekday_last_week":
            "baseline_prediction",
        "calendar_aware_baseline":
            "calendar_prediction",
    }

    model_metric_records = []

    for model_name, prediction_column in (
        model_columns.items()
    ):
        metrics = calculate_metrics(
            test_df,
            prediction_column,
        )

        metrics["model_name"] = model_name

        model_metric_records.append(
            metrics
        )

    model_metrics_df = pd.DataFrame(
        model_metric_records
    )

    model_metrics_df = (
        model_metrics_df
        .sort_values("wape_pct")
        .reset_index(drop=True)
    )

    print("\n" + "=" * 70)
    print("6. 전체 모델 성능 비교")
    print("=" * 70)
    print(
        model_metrics_df
        .round(3)
        .to_string(index=False)
    )
    
    day_type_metric_records = []

    for day_type, segment_df in (
        test_df.groupby(
            "day_type",
            sort=True,
        )
    ):
        for model_name, prediction_column in (
            model_columns.items()
        ):
            metrics = calculate_metrics(
                segment_df,
                prediction_column,
            )

            metrics["model_name"] = model_name
            metrics["day_type"] = day_type

            day_type_metric_records.append(
                metrics
            )

    day_type_metrics_df = pd.DataFrame(
        day_type_metric_records
    )

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
    print("7. 날짜 유형별 성능 비교")
    print("=" * 70)
    print(
        day_type_metrics_df
        .round(3)
        .to_string(index=False)
    )
    
    test_df[
        "baseline_absolute_error"
    ] = (
        test_df["baseline_prediction"]
        - test_df["passenger_count"]
    ).abs()

    test_df[
        "calendar_absolute_error"
    ] = (
        test_df["calendar_prediction"]
        - test_df["passenger_count"]
    ).abs()
    
    daily_comparison = (
        test_df
        .groupby(
            [
                "date",
                "day_type",
            ],
            as_index=False,
        )
        .agg(
            actual_total=(
                "passenger_count",
                "sum",
            ),
            baseline_absolute_error=(
                "baseline_absolute_error",
                "sum",
            ),
            calendar_absolute_error=(
                "calendar_absolute_error",
                "sum",
            ),
        )
    )
    
    daily_comparison[
        "baseline_ape_pct"
    ] = (
        daily_comparison[
            "baseline_absolute_error"
        ]
        / daily_comparison["actual_total"]
        * 100
    )

    daily_comparison[
        "calendar_ape_pct"
    ] = (
        daily_comparison[
            "calendar_absolute_error"
        ]
        / daily_comparison["actual_total"]
        * 100
    )
    
    daily_comparison[
        "calendar_is_better"
    ] = (
        daily_comparison[
            "calendar_ape_pct"
        ]
        < daily_comparison[
            "baseline_ape_pct"
        ]
    )
    
    improved_date_rate = (
        daily_comparison[
            "calendar_is_better"
        ]
        .mean()
        * 100
    )

    baseline_daily_median = (
        daily_comparison[
            "baseline_ape_pct"
        ]
        .median()
    )

    calendar_daily_median = (
        daily_comparison[
            "calendar_ape_pct"
        ]
        .median()
    )

    print("\n" + "=" * 70)
    print("8. 날짜별 개선 안정성")
    print("=" * 70)
    print(
        f"달력 인식 모델 개선 날짜 비율: "
        f"{improved_date_rate:.2f}%"
    )
    print(
        f"기존 모델 날짜별 오차율 중앙값: "
        f"{baseline_daily_median:.3f}%"
    )
    print(
        f"달력 인식 모델 날짜별 오차율 중앙값: "
        f"{calendar_daily_median:.3f}%"
    )
    
    key_dates = pd.to_datetime(
        [
            "2025-10-06",
            "2025-10-07",
            "2025-10-13",
            "2025-10-14",
            "2025-12-25",
            "2025-12-26",
        ]
    )

    key_date_result = (
        daily_comparison.loc[
            daily_comparison["date"]
            .isin(key_dates)
        ]
        .sort_values("date")
    )

    print("\n" + "=" * 70)
    print("9. 핵심 날짜 네트워크 오차 비교")
    print("=" * 70)
    key_date_display = (
        key_date_result.copy()
    )

    key_date_numeric_columns = [
        "actual_total",
        "baseline_absolute_error",
        "calendar_absolute_error",
        "baseline_ape_pct",
        "calendar_ape_pct",
    ]

    key_date_display[
        key_date_numeric_columns
    ] = (
        key_date_display[
            key_date_numeric_columns
        ]
        .round(3)
    )

    print(
        key_date_display.to_string(
            index=False,
        )
    )
    
    comparison_tolerance = 1e-9

    daily_comparison[
        "ape_difference"
    ] = (
        daily_comparison["calendar_ape_pct"]
        - daily_comparison["baseline_ape_pct"]
    )

    daily_comparison[
        "comparison_result"
    ] = np.select(
        [
            daily_comparison[
                "ape_difference"
            ] < -comparison_tolerance,

            daily_comparison[
                "ape_difference"
            ] > comparison_tolerance,
        ],
        [
            "improved",
            "worsened",
        ],
        default="unchanged",
    )
    

    comparison_counts = (
        daily_comparison[
            "comparison_result"
        ]
        .value_counts()
    )

    changed_date_mask = (
        daily_comparison[
            "comparison_result"
        ]
        != "unchanged"
    )

    changed_date_count = (
        changed_date_mask.sum()
    )

    improved_changed_date_rate = (
        (
            daily_comparison.loc[
                changed_date_mask,
                "comparison_result",
            ]
            == "improved"
        )
        .mean()
        * 100
    )

    non_worsened_date_rate = (
        (
            daily_comparison[
                "comparison_result"
            ]
            != "worsened"
        )
        .mean()
        * 100
    )

    print("\n날짜별 비교 결과:")
    print(comparison_counts.to_string())

    print(
        "예측값이 실제로 변경된 날짜 수:",
        changed_date_count,
    )

    print(
        "변경 날짜 중 개선 비율:",
        f"{improved_changed_date_rate:.2f}%",
    )

    print(
        "전체 날짜 중 비악화 비율:",
        f"{non_worsened_date_rate:.2f}%",
    )
        
    worsened_dates = (
        daily_comparison.loc[
            daily_comparison[
                "comparison_result"
            ]
            == "worsened",
            [
                "date",
                "day_type",
                "actual_total",
                "baseline_absolute_error",
                "calendar_absolute_error",
                "baseline_ape_pct",
                "calendar_ape_pct",
                "ape_difference",
            ],
        ]
        .copy()
    )

    worsened_numeric_columns = [
        "actual_total",
        "baseline_absolute_error",
        "calendar_absolute_error",
        "baseline_ape_pct",
        "calendar_ape_pct",
        "ape_difference",
    ]

    worsened_dates[
        worsened_numeric_columns
    ] = (
        worsened_dates[
            worsened_numeric_columns
        ]
        .round(3)
    )

    print("\n" + "=" * 70)
    print("10. 달력 인식 모델 악화 날짜")
    print("=" * 70)
    print(
        worsened_dates.to_string(
            index=False,
        )
    )
    
    if not worsened_dates.empty:
        worsened_date = (
            worsened_dates["date"]
            .iloc[0]
        )

        worsened_station_detail = (
            test_df.loc[
                test_df["date"]
                == worsened_date,
                [
                    "date",
                    "station_name",
                    "day_type",
                    "passenger_count",
                    "baseline_prediction",
                    "calendar_prediction",
                    "baseline_absolute_error",
                    "calendar_absolute_error",
                    "prediction_source",
                ],
            ]
            .copy()
        )

        worsened_station_detail[
            "error_increase"
        ] = (
            worsened_station_detail[
                "calendar_absolute_error"
            ]
            - worsened_station_detail[
                "baseline_absolute_error"
            ]
        )

        worsened_station_detail = (
            worsened_station_detail
            .sort_values(
                "error_increase",
                ascending=False,
            )
            .head(20)
        )

        print("\n악화 기여 상위 역:")
        print(
            worsened_station_detail
            .round(
                {
                    "passenger_count": 1,
                    "baseline_prediction": 1,
                    "calendar_prediction": 1,
                    "baseline_absolute_error": 1,
                    "calendar_absolute_error": 1,
                    "error_increase": 1,
                }
            )
            .to_string(index=False)
        )
        
    future_changed_source = (
        daily_df[
            [
                "date",
                "station_name",
                "passenger_count",
                "day_type",
            ]
        ]
        .copy()
    )

    future_change_mask = (
        future_changed_source["date"]
        >= pd.Timestamp("2025-12-01")
    )

    future_changed_source.loc[
        future_change_mask,
        "passenger_count",
    ] = (
        future_changed_source.loc[
            future_change_mask,
            "passenger_count",
        ]
        * 100
    )

    future_changed_df = add_lag_features(
        future_changed_source
    )

    future_changed_df = (
        add_historical_daytype_statistics(
            future_changed_df,
            minimum_history=3,
        )
    )

    future_changed_df = (
        build_calendar_aware_prediction(
            future_changed_df,
            minimum_history=3,
        )
    )

    october_mask = daily_df[
        "date"
    ].between(
        pd.Timestamp("2025-10-01"),
        pd.Timestamp("2025-10-31"),
    )

    original_october = (
        daily_df.loc[
            october_mask,
            [
                "date",
                "station_name",
                "prediction",
            ],
        ]
        .rename(
            columns={
                "prediction":
                    "original_prediction",
            }
        )
    )

    changed_october = (
        future_changed_df.loc[
            future_changed_df["date"]
            .between(
                pd.Timestamp("2025-10-01"),
                pd.Timestamp("2025-10-31"),
            ),
            [
                "date",
                "station_name",
                "prediction",
            ],
        ]
        .rename(
            columns={
                "prediction":
                    "future_changed_prediction",
            }
        )
    )

    leakage_comparison = (
        original_october
        .merge(
            changed_october,
            on=[
                "date",
                "station_name",
            ],
            how="inner",
            validate="one_to_one",
        )
    )

    maximum_prediction_difference = (
        (
            leakage_comparison[
                "original_prediction"
            ]
            - leakage_comparison[
                "future_changed_prediction"
            ]
        )
        .abs()
        .max()
    )

    print("\n" + "=" * 70)
    print("11. 미래 데이터 파괴 실험")
    print("=" * 70)
    print(
        "12월 실제값 100배 변경 후 "
        "10월 예측 최대 차이:",
        maximum_prediction_difference,
    )

    if not np.isclose(
        maximum_prediction_difference,
        0.0,
    ):
        raise ValueError(
            "미래 데이터가 과거 예측에 영향을 줬습니다."
        )
        

    OUTPUT_DIR = Path("outputs")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_columns = [
        "date",
        "station_name",
        "passenger_count",
        "day_type",
        "lag_1",
        "lag_7",
        "lag_7_day_type",
        "historical_daytype_count",
        "historical_daytype_median",
        "baseline_prediction",
        "calendar_prediction",
        "prediction_source",
        "baseline_absolute_error",
        "calendar_absolute_error",
    ]

    predictions_output = (
        test_df.loc[
            :,
            prediction_columns,
        ]
        .copy()
    )

    worsened_dates_output = (
        daily_comparison.loc[
            daily_comparison[
                "comparison_result"
            ]
            == "worsened"
        ]
        .copy()
    )

    output_tables = {
        "10_calendar_model_metrics.csv":
            model_metrics_df,

        "10_calendar_day_type_metrics.csv":
            day_type_metrics_df,

        "10_calendar_daily_comparison.csv":
            daily_comparison,

        "10_calendar_predictions.csv":
            predictions_output,

        "10_calendar_worsened_dates.csv":
            worsened_dates_output,
    }

    expected_output_rows = {
        "10_calendar_model_metrics.csv":
            2,

        "10_calendar_day_type_metrics.csv":
            6,

        "10_calendar_daily_comparison.csv":
            92,

        "10_calendar_predictions.csv":
            22_080,

        "10_calendar_worsened_dates.csv":
            1,
    }

    print("\n" + "=" * 70)
    print("12. CSV 결과 저장 및 검증")
    print("=" * 70)

    for filename, output_dataframe in (
        output_tables.items()
    ):
        output_path = (
            OUTPUT_DIR
            / filename
        )

        output_dataframe.to_csv(
            output_path,
            index=False,
            encoding="utf-8-sig",
        )

        actual_rows = len(
            output_dataframe
        )

        expected_rows = (
            expected_output_rows[
                filename
            ]
        )

        print(
            f"{filename}: "
            f"{actual_rows:,}행"
        )

        if actual_rows != expected_rows:
            raise ValueError(
                f"{filename} 행 수 오류: "
                f"예상 {expected_rows:,}, "
                f"실제 {actual_rows:,}"
            )

    prediction_missing_counts = (
        predictions_output[
            [
                "baseline_prediction",
                "calendar_prediction",
            ]
        ]
        .isna()
        .sum()
    )

    print("\n예측값 결측 수:")
    print(
        prediction_missing_counts
        .to_string()
    )

    if (
        prediction_missing_counts.sum()
        > 0
    ):
        raise ValueError(
            "저장할 예측 결과에 결측값이 있습니다."
        )
        
    model_wape_figure_path = (
        save_model_wape_comparison(
            model_metrics_df,
            OUTPUT_DIR
        )
    )
    
    print("\n" + "=" * 70)
    print("13. 전체 WAPE 비교 그래프 저장")
    print("=" * 70)
    print(
        model_wape_figure_path
    )
    
    day_type_wape_figure_path = (
        save_day_type_wape_comparison(
            day_type_metrics_df,
            OUTPUT_DIR,
        )
    )

    print("\n" + "=" * 70)
    print("14. 날짜 유형별 WAPE 비교 그래프 저장")
    print("=" * 70)
    print(
        day_type_wape_figure_path
    )
    
    key_dates_figure_path = (
        save_key_dates_error_comparison(
            key_date_result,
            OUTPUT_DIR,
        )
    )

    print("\n" + "=" * 70)
    print("15. 핵심 날짜 오차 비교 그래프 저장")
    print("=" * 70)
    print(key_dates_figure_path)