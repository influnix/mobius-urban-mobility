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
    silhouette_samples,
    silhouette_score,
)


# ============================================================
# 1. 한글 폰트 설정
# ============================================================

def configure_korean_font() -> None:
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
# 2. 파일 경로와 설정
# ============================================================

TIME_PROFILE_PATH = Path(
    "outputs/05_station_time_direction_average.csv"
)

STATION_PROFILE_PATH = Path(
    "outputs/05_station_time_direction_profile.csv"
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_RANDOM_STATE = 42
N_INIT = 20
MAX_ITER = 500

TEST_SEEDS = [
    0,
    1,
    2,
    3,
    10,
    20,
    42,
    99,
]


# ============================================================
# 3. 입력 파일 확인
# ============================================================

for file_path in [
    TIME_PROFILE_PATH,
    STATION_PROFILE_PATH,
]:
    if not file_path.exists():
        raise FileNotFoundError(
            "\n필요한 파일을 찾지 못했습니다.\n"
            f"파일 위치: {file_path.resolve()}\n"
        )


# ============================================================
# 4. 시간대 프로파일 불러오기
# ============================================================

time_df = pd.read_csv(
    TIME_PROFILE_PATH,
    encoding="utf-8-sig",
)

required_time_columns = [
    "station_name",
    "time_slot",
    "time_order",
    "boarding_share",
    "alighting_share",
]

missing_time_columns = [
    column
    for column in required_time_columns
    if column not in time_df.columns
]

if missing_time_columns:
    raise ValueError(
        "\n시간대 프로파일에 필요한 컬럼이 없습니다.\n"
        f"누락 컬럼: {missing_time_columns}\n"
    )


# ============================================================
# 5. 데이터 구조 검증
# ============================================================

number_of_stations = (
    time_df["station_name"].nunique()
)

number_of_time_slots = (
    time_df["time_slot"].nunique()
)

expected_rows = (
    number_of_stations
    * number_of_time_slots
)

duplicate_count = (
    time_df.duplicated(
        subset=[
            "station_name",
            "time_slot",
        ]
    )
    .sum()
)

missing_count = (
    time_df[required_time_columns]
    .isna()
    .any(axis=1)
    .sum()
)

station_slot_counts = (
    time_df
    .groupby("station_name")["time_slot"]
    .nunique()
)

incomplete_station_count = (
    station_slot_counts
    != number_of_time_slots
).sum()

print("=" * 70)
print("1. 입력 데이터 검증")
print("=" * 70)
print(f"고유 역 수: {number_of_stations:,}")
print(f"고유 시간대 수: {number_of_time_slots:,}")
print(f"예상 행 수: {expected_rows:,}")
print(f"실제 행 수: {len(time_df):,}")
print(f"역·시간대 중복 행 수: {duplicate_count:,}")
print(f"필수 값 결측 행 수: {missing_count:,}")
print(
    "시간대가 불완전한 역 수: "
    f"{incomplete_station_count:,}"
)

if len(time_df) != expected_rows:
    raise ValueError(
        "예상 행 수와 실제 행 수가 다릅니다."
    )

if duplicate_count > 0:
    raise ValueError(
        "역·시간대 중복 행이 존재합니다."
    )

if missing_count > 0:
    raise ValueError(
        "입력 데이터에 결측값이 존재합니다."
    )

if incomplete_station_count > 0:
    raise ValueError(
        "일부 역에 20개 시간대가 모두 존재하지 않습니다."
    )


# ============================================================
# 6. 역 × 40개 입력 행렬 생성
# ============================================================

time_order_table = (
    time_df[
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

boarding_features = (
    time_df
    .pivot(
        index="station_name",
        columns="time_order",
        values="boarding_share",
    )
    .reindex(columns=time_orders)
)

alighting_features = (
    time_df
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
).sort_index()

X = feature_df.to_numpy(
    dtype=float,
)

print("\n" + "=" * 70)
print("2. 군집화 입력 행렬")
print("=" * 70)
print(
    f"입력 크기: "
    f"{feature_df.shape[0]:,}행 "
    f"× {feature_df.shape[1]:,}열"
)

if feature_df.shape != (
    240,
    40,
):
    raise ValueError(
        "입력 행렬 크기가 240 × 40이 아닙니다."
    )


# ============================================================
# 7. 기존 5단계 프로파일 불러오기
# ============================================================

profile_df = pd.read_csv(
    STATION_PROFILE_PATH,
    encoding="utf-8-sig",
)

required_profile_columns = [
    "station_name",
    "lines",
    "weekday_average",
    "relative_weekend_index",
    "orientation_score",
    "peak_concentration",
    "commute_pattern",
]

missing_profile_columns = [
    column
    for column in required_profile_columns
    if column not in profile_df.columns
]

if missing_profile_columns:
    raise ValueError(
        "\n5단계 프로파일에 필요한 컬럼이 없습니다.\n"
        f"누락 컬럼: {missing_profile_columns}\n"
    )

profile_df = profile_df[
    required_profile_columns
].copy()


# ============================================================
# 8. 모델 학습 함수
# ============================================================

def fit_kmeans(
    number_of_clusters: int,
    random_state: int,
) -> dict:
    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=random_state,
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

    return {
        "model": model,
        "labels": labels,
        "assigned_distances": assigned_distances,
        "sample_silhouettes": sample_silhouettes,
        "silhouette": silhouette_score(
            X,
            labels,
        ),
        "inertia": model.inertia_,
        "calinski_harabasz": (
            calinski_harabasz_score(
                X,
                labels,
            )
        ),
        "davies_bouldin": (
            davies_bouldin_score(
                X,
                labels,
            )
        ),
        "cluster_sizes": np.bincount(
            labels
        ),
    }


# ============================================================
# 9. k=2와 k=3 기준 모델 학습
# ============================================================

reference_results = {}

for k in [
    2,
    3,
]:
    reference_results[k] = fit_kmeans(
        number_of_clusters=k,
        random_state=REFERENCE_RANDOM_STATE,
    )


# ============================================================
# 10. 역별 기준 모델 결과 구성
# ============================================================

station_result = pd.DataFrame(
    {
        "station_name": feature_df.index,
        "k2_cluster_id": (
            reference_results[2]["labels"]
        ),
        "k3_cluster_id": (
            reference_results[3]["labels"]
        ),
        "k2_distance_to_centroid": (
            reference_results[2][
                "assigned_distances"
            ]
        ),
        "k3_distance_to_centroid": (
            reference_results[3][
                "assigned_distances"
            ]
        ),
        "k2_silhouette_sample": (
            reference_results[2][
                "sample_silhouettes"
            ]
        ),
        "k3_silhouette_sample": (
            reference_results[3][
                "sample_silhouettes"
            ]
        ),
    }
)

station_result = station_result.merge(
    profile_df,
    on="station_name",
    how="left",
    validate="one_to_one",
)

merge_failure_count = (
    station_result["commute_pattern"]
    .isna()
    .sum()
)

print("\n" + "=" * 70)
print("3. 5단계 프로파일 결합")
print("=" * 70)
print(
    "결합 실패 역 수: "
    f"{merge_failure_count:,}"
)

if merge_failure_count > 0:
    raise ValueError(
        "5단계 프로파일 결합에 실패했습니다."
    )


# ============================================================
# 11. 군집 ID에 의미 있는 이름 부여
# ============================================================

def create_cluster_name_map(
    dataframe: pd.DataFrame,
    cluster_column: str,
    number_of_clusters: int,
) -> dict:
    orientation_summary = (
        dataframe
        .groupby(
            cluster_column,
            as_index=False,
        )["orientation_score"]
        .mean()
        .sort_values("orientation_score")
    )

    ordered_cluster_ids = (
        orientation_summary[
            cluster_column
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
        "이 함수는 k=2 또는 k=3만 지원합니다."
    )


k2_name_map = create_cluster_name_map(
    dataframe=station_result,
    cluster_column="k2_cluster_id",
    number_of_clusters=2,
)

k3_name_map = create_cluster_name_map(
    dataframe=station_result,
    cluster_column="k3_cluster_id",
    number_of_clusters=3,
)

station_result["k2_cluster_name"] = (
    station_result["k2_cluster_id"]
    .map(k2_name_map)
)

station_result["k3_cluster_name"] = (
    station_result["k3_cluster_id"]
    .map(k3_name_map)
)


# ============================================================
# 12. k=2와 k=3 평가 결과 비교
# ============================================================

comparison_records = []

for k in [
    2,
    3,
]:
    result = reference_results[k]

    comparison_records.append(
        {
            "k": k,
            "silhouette_score": (
                result["silhouette"]
            ),
            "inertia": result["inertia"],
            "calinski_harabasz_score": (
                result["calinski_harabasz"]
            ),
            "davies_bouldin_score": (
                result["davies_bouldin"]
            ),
            "minimum_cluster_size": (
                result["cluster_sizes"].min()
            ),
            "maximum_cluster_size": (
                result["cluster_sizes"].max()
            ),
        }
    )

comparison_df = pd.DataFrame(
    comparison_records
)


# ============================================================
# 13. 여러 random seed 안정성 실험
# ============================================================

stability_records = []

for k in [
    2,
    3,
]:
    reference_labels = (
        reference_results[k]["labels"]
    )

    for seed in TEST_SEEDS:
        result = fit_kmeans(
            number_of_clusters=k,
            random_state=seed,
        )

        ari = adjusted_rand_score(
            reference_labels,
            result["labels"],
        )

        stability_records.append(
            {
                "k": k,
                "random_state": seed,
                "adjusted_rand_index": ari,
                "silhouette_score": (
                    result["silhouette"]
                ),
                "inertia": result["inertia"],
                "minimum_cluster_size": (
                    result[
                        "cluster_sizes"
                    ].min()
                ),
                "maximum_cluster_size": (
                    result[
                        "cluster_sizes"
                    ].max()
                ),
            }
        )

stability_df = pd.DataFrame(
    stability_records
)

stability_summary = (
    stability_df
    .groupby(
        "k",
        as_index=False,
    )
    .agg(
        mean_ari=(
            "adjusted_rand_index",
            "mean",
        ),
        minimum_ari=(
            "adjusted_rand_index",
            "min",
        ),
        maximum_ari=(
            "adjusted_rand_index",
            "max",
        ),
        mean_silhouette=(
            "silhouette_score",
            "mean",
        ),
        minimum_silhouette=(
            "silhouette_score",
            "min",
        ),
        maximum_silhouette=(
            "silhouette_score",
            "max",
        ),
    )
)


# ============================================================
# 14. k=2와 k=3 구조 비교
# ============================================================

k2_k3_crosstab = pd.crosstab(
    station_result["k2_cluster_name"],
    station_result["k3_cluster_name"],
)

k3_rule_crosstab = pd.crosstab(
    station_result["k3_cluster_name"],
    station_result["commute_pattern"],
)


# ============================================================
# 15. k=3 군집별 요약
# ============================================================

k3_summary = (
    station_result
    .groupby(
        [
            "k3_cluster_id",
            "k3_cluster_name",
        ],
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
        average_relative_weekend_index=(
            "relative_weekend_index",
            "mean",
        ),
        average_silhouette=(
            "k3_silhouette_sample",
            "mean",
        ),
    )
    .sort_values(
        "average_orientation_score",
        ascending=False,
    )
)

for column in [
    "average_weekday_volume",
]:
    k3_summary[column] = (
        k3_summary[column]
        .round()
        .astype("int64")
    )

for column in [
    "average_orientation_score",
    "average_peak_concentration",
    "average_relative_weekend_index",
    "average_silhouette",
]:
    k3_summary[column] = (
        k3_summary[column]
        .round(3)
    )


# ============================================================
# 16. k=3 대표 역 추출
# ============================================================

k3_representative_stations = (
    station_result
    .sort_values(
        [
            "k3_cluster_id",
            "k3_distance_to_centroid",
        ]
    )
    .groupby(
        "k3_cluster_id",
        as_index=False,
    )
    .head(10)
    .copy()
)


# ============================================================
# 17. 결과 파일 저장
# ============================================================

COMPARISON_PATH = (
    OUTPUT_DIR
    / "07_k2_k3_model_comparison.csv"
)

STABILITY_PATH = (
    OUTPUT_DIR
    / "07_seed_stability_results.csv"
)

STABILITY_SUMMARY_PATH = (
    OUTPUT_DIR
    / "07_seed_stability_summary.csv"
)

STATION_RESULT_PATH = (
    OUTPUT_DIR
    / "07_station_k2_k3_labels.csv"
)

K2_K3_CROSSTAB_PATH = (
    OUTPUT_DIR
    / "07_k2_k3_crosstab.csv"
)

K3_RULE_CROSSTAB_PATH = (
    OUTPUT_DIR
    / "07_k3_rule_crosstab.csv"
)

K3_SUMMARY_PATH = (
    OUTPUT_DIR
    / "07_k3_cluster_summary.csv"
)

K3_REPRESENTATIVE_PATH = (
    OUTPUT_DIR
    / "07_k3_representative_stations.csv"
)

comparison_df.to_csv(
    COMPARISON_PATH,
    index=False,
    encoding="utf-8-sig",
)

stability_df.to_csv(
    STABILITY_PATH,
    index=False,
    encoding="utf-8-sig",
)

stability_summary.to_csv(
    STABILITY_SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig",
)

station_result.to_csv(
    STATION_RESULT_PATH,
    index=False,
    encoding="utf-8-sig",
)

k2_k3_crosstab.to_csv(
    K2_K3_CROSSTAB_PATH,
    encoding="utf-8-sig",
)

k3_rule_crosstab.to_csv(
    K3_RULE_CROSSTAB_PATH,
    encoding="utf-8-sig",
)

k3_summary.to_csv(
    K3_SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig",
)

k3_representative_stations.to_csv(
    K3_REPRESENTATIVE_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 18. k=2와 k=3 Silhouette 비교 그래프
# ============================================================

plt.figure(figsize=(8, 6))

bars = plt.bar(
    comparison_df["k"].astype(str),
    comparison_df["silhouette_score"],
)

plt.title(
    "k=2와 k=3 Silhouette Score 비교"
)
plt.xlabel("군집 수 k")
plt.ylabel("Silhouette Score")
plt.ylim(
    0,
    comparison_df[
        "silhouette_score"
    ].max() * 1.2,
)
plt.grid(
    axis="y",
    alpha=0.3,
)

for bar, value in zip(
    bars,
    comparison_df["silhouette_score"],
):
    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{value:.4f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()

COMPARISON_GRAPH_PATH = (
    OUTPUT_DIR
    / "07_k2_k3_silhouette_comparison.png"
)

plt.savefig(
    COMPARISON_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 19. random seed별 ARI 그래프
# ============================================================

plt.figure(figsize=(10, 6))

for k, group in stability_df.groupby("k"):
    group = group.sort_values(
        "random_state"
    )

    plt.plot(
        group["random_state"],
        group["adjusted_rand_index"],
        marker="o",
        label=f"k={k}",
    )

plt.axhline(
    1,
    linestyle="--",
    linewidth=1,
)

plt.title(
    "Random Seed에 따른 군집 안정성"
)
plt.xlabel("Random State")
plt.ylabel(
    "기준 모델 대비 Adjusted Rand Index"
)
plt.ylim(
    min(
        0,
        stability_df[
            "adjusted_rand_index"
        ].min() - 0.05,
    ),
    1.05,
)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

STABILITY_GRAPH_PATH = (
    OUTPUT_DIR
    / "07_seed_stability_ari.png"
)

plt.savefig(
    STABILITY_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 20. k=3과 규칙 유형 비교 그래프
# ============================================================

plot_crosstab = (
    k3_rule_crosstab
    .reindex(
        columns=[
            "업무 유입형",
            "혼합형",
            "주거 유출·귀가형",
        ],
        fill_value=0,
    )
)

plot_crosstab.plot(
    kind="bar",
    stacked=True,
    figsize=(11, 7),
)

plt.title(
    "k=3 군집과 기존 규칙 기반 유형 비교"
)
plt.xlabel("k=3 군집")
plt.ylabel("역 수")
plt.xticks(
    rotation=0,
)
plt.grid(
    axis="y",
    alpha=0.3,
)
plt.legend(
    title="기존 유형",
)
plt.tight_layout()

K3_RULE_GRAPH_PATH = (
    OUTPUT_DIR
    / "07_k3_rule_comparison.png"
)

plt.savefig(
    K3_RULE_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 21. 터미널 출력
# ============================================================

print("\n" + "=" * 70)
print("4. k=2와 k=3 모델 비교")
print("=" * 70)

print(
    comparison_df
    .round(4)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("5. Random Seed 안정성 요약")
print("=" * 70)

print(
    stability_summary
    .round(4)
    .to_string(index=False)
)

print("\n세부 결과:")

print(
    stability_df
    .round(4)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("6. k=2에서 k=3으로의 분리")
print("=" * 70)

print(
    k2_k3_crosstab.to_string()
)

print("\n" + "=" * 70)
print("7. k=3과 기존 규칙 유형 비교")
print("=" * 70)

print(
    k3_rule_crosstab.to_string()
)

print("\n" + "=" * 70)
print("8. k=3 군집별 요약")
print("=" * 70)

print(
    k3_summary.to_string(
        index=False,
    )
)

print("\n" + "=" * 70)
print("9. k=3 군집별 대표 역")
print("=" * 70)

print(
    k3_representative_stations[
        [
            "k3_cluster_name",
            "station_name",
            "k3_distance_to_centroid",
            "orientation_score",
            "commute_pattern",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("10. 작업 완료")
print("=" * 70)
print(f"모델 비교: {COMPARISON_PATH.resolve()}")
print(f"Seed별 결과: {STABILITY_PATH.resolve()}")
print(
    f"Seed 안정성 요약: "
    f"{STABILITY_SUMMARY_PATH.resolve()}"
)
print(f"역별 k2·k3 결과: {STATION_RESULT_PATH.resolve()}")
print(f"k2·k3 교차표: {K2_K3_CROSSTAB_PATH.resolve()}")
print(
    f"k3·규칙 유형 교차표: "
    f"{K3_RULE_CROSSTAB_PATH.resolve()}"
)
print(f"k3 군집 요약: {K3_SUMMARY_PATH.resolve()}")
print(
    f"k3 대표 역: "
    f"{K3_REPRESENTATIVE_PATH.resolve()}"
)
print(
    f"Silhouette 비교 그래프: "
    f"{COMPARISON_GRAPH_PATH.resolve()}"
)
print(
    f"Seed 안정성 그래프: "
    f"{STABILITY_GRAPH_PATH.resolve()}"
)
print(
    f"k3 유형 비교 그래프: "
    f"{K3_RULE_GRAPH_PATH.resolve()}"
)