from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, rcParams
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)


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
            "경고: 맑은 고딕을 찾지 못했습니다. "
            "그래프의 한글이 깨질 수 있습니다."
        )


configure_korean_font()


# ============================================================
# 2. 파일 경로와 분석 설정
# ============================================================

TIME_PROFILE_PATH = Path(
    "outputs/05_station_time_direction_average.csv"
)

STATION_PROFILE_PATH = Path(
    "outputs/05_station_time_direction_profile.csv"
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MIN_K = 2
MAX_K = 8

RANDOM_STATE = 42
N_INIT = 20

# None이면 Silhouette가 가장 높은 k를 자동 선택한다.
# 특정 k를 별도로 실험하려면 3과 같이 숫자를 입력한다.
FORCED_K = None


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
            f"확인할 위치: {file_path.resolve()}\n"
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
        "\n시간대 데이터에 필요한 컬럼이 없습니다.\n"
        f"누락 컬럼: {missing_time_columns}\n"
    )

print("=" * 70)
print("1. 시간대 프로파일 불러오기")
print("=" * 70)
print(f"행 수: {len(time_df):,}")
print(f"열 수: {len(time_df.columns):,}")


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

station_slot_counts = (
    time_df
    .groupby("station_name")["time_slot"]
    .nunique()
)

incomplete_station_count = (
    station_slot_counts
    != number_of_time_slots
).sum()

missing_value_count = (
    time_df[required_time_columns]
    .isna()
    .any(axis=1)
    .sum()
)

negative_share_count = (
    (
        time_df["boarding_share"] < 0
    )
    | (
        time_df["alighting_share"] < 0
    )
).sum()

station_share_sums = (
    time_df
    .groupby(
        "station_name",
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

station_share_sums["total_share_sum"] = (
    station_share_sums["boarding_share_sum"]
    + station_share_sums["alighting_share_sum"]
)

maximum_share_error = (
    station_share_sums["total_share_sum"]
    .sub(1)
    .abs()
    .max()
)

print("\n" + "=" * 70)
print("2. 데이터 구조 검증")
print("=" * 70)
print(f"고유 역 수: {number_of_stations:,}")
print(f"고유 시간대 수: {number_of_time_slots:,}")
print(f"예상 행 수: {expected_rows:,}")
print(f"실제 행 수: {len(time_df):,}")
print(f"역·시간대 중복 행 수: {duplicate_count:,}")
print(
    "시간대가 불완전한 역 수: "
    f"{incomplete_station_count:,}"
)
print(f"필수 값 결측 행 수: {missing_value_count:,}")
print(f"음수 비중 행 수: {negative_share_count:,}")
print(
    "역별 전체 비중 합계의 최대 오차: "
    f"{maximum_share_error:.10f}"
)

if len(time_df) != expected_rows:
    raise ValueError(
        "예상 행 수와 실제 행 수가 다릅니다."
    )

if duplicate_count > 0:
    raise ValueError(
        "동일한 역·시간대 조합이 중복되어 있습니다."
    )

if incomplete_station_count > 0:
    raise ValueError(
        "일부 역에 20개 시간대가 모두 존재하지 않습니다."
    )

if missing_value_count > 0:
    raise ValueError(
        "군집화 입력 데이터에 결측값이 있습니다."
    )

if negative_share_count > 0:
    raise ValueError(
        "승차 또는 하차 비중에 음수가 있습니다."
    )

if maximum_share_error > 1e-6:
    raise ValueError(
        "역별 승하차 비중 합계가 1이 아닙니다."
    )


# ============================================================
# 6. 시간대 순서 확인
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

time_slot_map = dict(
    zip(
        time_order_table["time_order"],
        time_order_table["time_slot"],
    )
)

print("\n시간대 순서:")

print(
    time_order_table.to_string(
        index=False,
    )
)


# ============================================================
# 7. 역 × 40개 변수 형태로 변환
# ============================================================

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
)

feature_df = feature_df.sort_index()

feature_missing_count = (
    feature_df.isna().sum().sum()
)

print("\n" + "=" * 70)
print("3. 군집화 입력 행렬")
print("=" * 70)
print(
    "입력 행렬 크기: "
    f"{feature_df.shape[0]:,}행 "
    f"× {feature_df.shape[1]:,}열"
)
print(
    "입력 행렬 결측값 수: "
    f"{feature_missing_count:,}"
)

if feature_df.shape != (
    number_of_stations,
    number_of_time_slots * 2,
):
    raise ValueError(
        "군집화 입력 행렬의 크기가 예상과 다릅니다."
    )

if feature_missing_count > 0:
    raise ValueError(
        "군집화 입력 행렬에 결측값이 있습니다."
    )

# 모든 변수가 동일하게 0~1 사이의 이용량 비중이므로
# 이번 기본 분석에서는 별도의 표준화를 하지 않는다.
X = feature_df.to_numpy(
    dtype=float,
)


# ============================================================
# 8. k=2~8 후보 모델 평가
# ============================================================

metric_records = []

for k in range(
    MIN_K,
    MAX_K + 1,
):
    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
        max_iter=500,
    )

    labels = model.fit_predict(X)

    cluster_sizes = np.bincount(labels)

    metric_records.append(
        {
            "k": k,
            "inertia": model.inertia_,
            "silhouette_score": (
                silhouette_score(
                    X,
                    labels,
                )
            ),
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
    )

metrics_df = pd.DataFrame(
    metric_records
)

best_k = int(
    metrics_df.loc[
        metrics_df[
            "silhouette_score"
        ].idxmax(),
        "k",
    ]
)

if FORCED_K is None:
    selected_k = best_k
else:
    selected_k = int(FORCED_K)

if not (
    MIN_K
    <= selected_k
    <= MAX_K
):
    raise ValueError(
        f"선택한 k={selected_k}가 "
        f"{MIN_K}~{MAX_K} 범위를 벗어났습니다."
    )

metrics_df["is_best_silhouette"] = (
    metrics_df["k"] == best_k
)

metrics_df["is_selected"] = (
    metrics_df["k"] == selected_k
)

METRICS_PATH = (
    OUTPUT_DIR
    / "06_k_selection_metrics.csv"
)

metrics_df.to_csv(
    METRICS_PATH,
    index=False,
    encoding="utf-8-sig",
)

print("\n" + "=" * 70)
print("4. 군집 수 후보 평가")
print("=" * 70)

print(
    metrics_df.round(4).to_string(
        index=False,
    )
)

print(
    f"\nSilhouette 기준 최적 k: {best_k}"
)
print(
    f"이번 분석에서 선택한 k: {selected_k}"
)


# ============================================================
# 9. 최종 K-Means 모델 학습
# ============================================================

final_model = KMeans(
    n_clusters=selected_k,
    random_state=RANDOM_STATE,
    n_init=N_INIT,
    max_iter=500,
)

final_labels = final_model.fit_predict(X)

all_distances = final_model.transform(X)

assigned_distances = all_distances[
    np.arange(len(final_labels)),
    final_labels,
]

sample_silhouettes = silhouette_samples(
    X,
    final_labels,
)

station_cluster_df = pd.DataFrame(
    {
        "station_name": feature_df.index,
        "cluster_id": final_labels,
        "distance_to_centroid": (
            assigned_distances
        ),
        "silhouette_sample": (
            sample_silhouettes
        ),
    }
)


# ============================================================
# 10. PCA 2차원 시각화 좌표 생성
# ============================================================

pca = PCA(
    n_components=2,
)

pca_coordinates = pca.fit_transform(X)

station_cluster_df["pc1"] = (
    pca_coordinates[:, 0]
)

station_cluster_df["pc2"] = (
    pca_coordinates[:, 1]
)

pc1_variance = (
    pca.explained_variance_ratio_[0]
)

pc2_variance = (
    pca.explained_variance_ratio_[1]
)

total_pca_variance = (
    pc1_variance
    + pc2_variance
)


# ============================================================
# 11. 5단계 프로파일 결합
# ============================================================

stage05_df = pd.read_csv(
    STATION_PROFILE_PATH,
    encoding="utf-8-sig",
)

required_profile_columns = [
    "station_name",
    "lines",
    "weekday_average",
    "relative_weekend_index",
    "station_type",
    "orientation_score",
    "peak_concentration",
    "commute_pattern",
]

missing_profile_columns = [
    column
    for column in required_profile_columns
    if column not in stage05_df.columns
]

if missing_profile_columns:
    raise ValueError(
        "\n5단계 프로파일에 필요한 컬럼이 없습니다.\n"
        f"누락 컬럼: {missing_profile_columns}\n"
    )

station_cluster_df = (
    station_cluster_df
    .merge(
        stage05_df[
            required_profile_columns
        ],
        on="station_name",
        how="left",
        validate="one_to_one",
    )
)

profile_merge_failure_count = (
    station_cluster_df[
        "commute_pattern"
    ]
    .isna()
    .sum()
)

print("\n" + "=" * 70)
print("5. 5단계 프로파일 결합")
print("=" * 70)
print(
    "5단계 프로파일 결합 실패 역 수: "
    f"{profile_merge_failure_count:,}"
)

if profile_merge_failure_count > 0:
    raise ValueError(
        "5단계 프로파일 결합에 실패했습니다."
    )


# ============================================================
# 12. 군집별 요약
# ============================================================

cluster_summary = (
    station_cluster_df
    .groupby(
        "cluster_id",
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
            "silhouette_sample",
            "mean",
        ),
    )
)

commute_crosstab = pd.crosstab(
    station_cluster_df["cluster_id"],
    station_cluster_df["commute_pattern"],
)

dominant_pattern = (
    commute_crosstab.idxmax(axis=1)
    .rename("dominant_commute_pattern")
    .reset_index()
)

cluster_summary = cluster_summary.merge(
    dominant_pattern,
    on="cluster_id",
    how="left",
    validate="one_to_one",
)


def interpret_cluster(
    average_orientation_score: float,
) -> str:
    """
    군집 자체는 40개 패턴 변수로 생성했다.
    이 함수는 생성된 군집의 평균 방향성을
    사후 해석하기 위한 용도다.
    """
    if average_orientation_score >= 0.15:
        return "업무 유입 중심"

    if average_orientation_score <= -0.15:
        return "주거 유출·귀가 중심"

    return "혼합 흐름 중심"


cluster_summary[
    "interpretation"
] = (
    cluster_summary[
        "average_orientation_score"
    ]
    .apply(interpret_cluster)
)

cluster_summary["cluster_name"] = (
    "군집 "
    + cluster_summary["cluster_id"].astype(str)
    + ": "
    + cluster_summary["interpretation"]
)

cluster_name_map = dict(
    zip(
        cluster_summary["cluster_id"],
        cluster_summary["cluster_name"],
    )
)

station_cluster_df["cluster_name"] = (
    station_cluster_df["cluster_id"]
    .map(cluster_name_map)
)


# ============================================================
# 13. 군집 중심에 가까운 대표 역 선정
# ============================================================

representative_stations = (
    station_cluster_df
    .sort_values(
        [
            "cluster_id",
            "distance_to_centroid",
        ]
    )
    .groupby(
        "cluster_id",
        as_index=False,
    )
    .head(10)
    .copy()
)


# ============================================================
# 14. 군집별 시간대 평균 프로파일
# ============================================================

time_with_clusters = time_df.merge(
    station_cluster_df[
        [
            "station_name",
            "cluster_id",
            "cluster_name",
        ]
    ],
    on="station_name",
    how="left",
    validate="many_to_one",
)

cluster_time_profile = (
    time_with_clusters
    .groupby(
        [
            "cluster_id",
            "cluster_name",
            "time_order",
            "time_slot",
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
            "cluster_id",
            "time_order",
        ]
    )
)


# ============================================================
# 15. 결과 파일 저장
# ============================================================

STATION_CLUSTER_PATH = (
    OUTPUT_DIR
    / "06_station_clusters.csv"
)

CLUSTER_SUMMARY_PATH = (
    OUTPUT_DIR
    / "06_cluster_summary.csv"
)

CROSSTAB_PATH = (
    OUTPUT_DIR
    / "06_cluster_commute_crosstab.csv"
)

REPRESENTATIVE_PATH = (
    OUTPUT_DIR
    / "06_representative_stations.csv"
)

CLUSTER_TIME_PATH = (
    OUTPUT_DIR
    / "06_cluster_time_profile.csv"
)

station_cluster_df.to_csv(
    STATION_CLUSTER_PATH,
    index=False,
    encoding="utf-8-sig",
)

cluster_summary.to_csv(
    CLUSTER_SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig",
)

commute_crosstab.to_csv(
    CROSSTAB_PATH,
    encoding="utf-8-sig",
)

representative_stations.to_csv(
    REPRESENTATIVE_PATH,
    index=False,
    encoding="utf-8-sig",
)

cluster_time_profile.to_csv(
    CLUSTER_TIME_PATH,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# 16. k별 Silhouette 그래프
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    metrics_df["k"],
    metrics_df["silhouette_score"],
    marker="o",
)

selected_metric = metrics_df.loc[
    metrics_df["k"] == selected_k
].iloc[0]

plt.scatter(
    [selected_k],
    [
        selected_metric[
            "silhouette_score"
        ]
    ],
    s=100,
    label=f"선택된 k={selected_k}",
)

plt.title(
    "군집 수에 따른 Silhouette Score"
)
plt.xlabel("군집 수 k")
plt.ylabel("Silhouette Score")
plt.xticks(
    metrics_df["k"]
)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

SILHOUETTE_GRAPH_PATH = (
    OUTPUT_DIR
    / "06_silhouette_by_k.png"
)

plt.savefig(
    SILHOUETTE_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 17. k별 Inertia 그래프
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    metrics_df["k"],
    metrics_df["inertia"],
    marker="o",
)

plt.title(
    "군집 수에 따른 K-Means Inertia"
)
plt.xlabel("군집 수 k")
plt.ylabel("Inertia")
plt.xticks(
    metrics_df["k"]
)
plt.grid(alpha=0.3)
plt.tight_layout()

INERTIA_GRAPH_PATH = (
    OUTPUT_DIR
    / "06_inertia_by_k.png"
)

plt.savefig(
    INERTIA_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 18. PCA 군집 산점도
# ============================================================

plt.figure(figsize=(11, 8))

for cluster_name, group in (
    station_cluster_df
    .groupby("cluster_name")
):
    plt.scatter(
        group["pc1"],
        group["pc2"],
        alpha=0.7,
        label=(
            f"{cluster_name} "
            f"(n={len(group)})"
        ),
    )

label_stations = (
    representative_stations
    .groupby(
        "cluster_id",
        as_index=False,
    )
    .head(3)
)

for _, row in label_stations.iterrows():
    plt.annotate(
        row["station_name"],
        (
            row["pc1"],
            row["pc2"],
        ),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8,
    )

plt.title(
    "시간대별 승하차 패턴 K-Means 군집"
)
plt.xlabel(
    f"PC1 ({pc1_variance * 100:.1f}% 설명)"
)
plt.ylabel(
    f"PC2 ({pc2_variance * 100:.1f}% 설명)"
)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

PCA_GRAPH_PATH = (
    OUTPUT_DIR
    / "06_cluster_pca_scatter.png"
)

plt.savefig(
    PCA_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 19. 군집별 승차 프로파일
# ============================================================

plt.figure(figsize=(14, 7))

for cluster_name, group in (
    cluster_time_profile
    .groupby("cluster_name")
):
    group = group.sort_values(
        "time_order"
    )

    plt.plot(
        group["time_order"],
        group["average_boarding_share"] * 100,
        marker="o",
        label=cluster_name,
    )

plt.xticks(
    time_order_table["time_order"],
    time_order_table["time_slot"],
    rotation=45,
    ha="right",
)

plt.title(
    "K-Means 군집별 일반 평일 시간대 승차 비중"
)
plt.xlabel("시간대")
plt.ylabel("일평균 전체 이용량 대비 승차 비중 (%)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

BOARDING_GRAPH_PATH = (
    OUTPUT_DIR
    / "06_cluster_boarding_profile.png"
)

plt.savefig(
    BOARDING_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 20. 군집별 하차 프로파일
# ============================================================

plt.figure(figsize=(14, 7))

for cluster_name, group in (
    cluster_time_profile
    .groupby("cluster_name")
):
    group = group.sort_values(
        "time_order"
    )

    plt.plot(
        group["time_order"],
        group["average_alighting_share"] * 100,
        marker="o",
        label=cluster_name,
    )

plt.xticks(
    time_order_table["time_order"],
    time_order_table["time_slot"],
    rotation=45,
    ha="right",
)

plt.title(
    "K-Means 군집별 일반 평일 시간대 하차 비중"
)
plt.xlabel("시간대")
plt.ylabel("일평균 전체 이용량 대비 하차 비중 (%)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

ALIGHTING_GRAPH_PATH = (
    OUTPUT_DIR
    / "06_cluster_alighting_profile.png"
)

plt.savefig(
    ALIGHTING_GRAPH_PATH,
    dpi=150,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# 21. 터미널 결과 출력
# ============================================================

print("\n" + "=" * 70)
print("6. 최종 군집 결과")
print("=" * 70)
print(f"선택된 군집 수: {selected_k}")
print(
    "최종 Silhouette Score: "
    f"{selected_metric['silhouette_score']:.4f}"
)
print(
    "PCA 두 축의 누적 설명 분산: "
    f"{total_pca_variance * 100:.1f}%"
)

print("\n군집별 요약:")

print(
    cluster_summary[
        [
            "cluster_name",
            "number_of_stations",
            "average_weekday_volume",
            "average_orientation_score",
            "average_peak_concentration",
            "average_relative_weekend_index",
            "average_silhouette",
            "dominant_commute_pattern",
        ]
    ]
    .round(3)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("7. 5단계 규칙 분류와 군집 결과 비교")
print("=" * 70)

print(
    commute_crosstab.to_string()
)

print("\n" + "=" * 70)
print("8. 군집별 대표 역")
print("=" * 70)

print(
    representative_stations[
        [
            "cluster_name",
            "station_name",
            "distance_to_centroid",
            "orientation_score",
            "commute_pattern",
        ]
    ]
    .round(4)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("9. 작업 완료")
print("=" * 70)
print(f"k 평가 결과: {METRICS_PATH.resolve()}")
print(f"역별 군집: {STATION_CLUSTER_PATH.resolve()}")
print(f"군집별 요약: {CLUSTER_SUMMARY_PATH.resolve()}")
print(f"기존 유형 비교: {CROSSTAB_PATH.resolve()}")
print(f"군집 대표 역: {REPRESENTATIVE_PATH.resolve()}")
print(f"군집 시간 프로파일: {CLUSTER_TIME_PATH.resolve()}")
print(f"Silhouette 그래프: {SILHOUETTE_GRAPH_PATH.resolve()}")
print(f"Inertia 그래프: {INERTIA_GRAPH_PATH.resolve()}")
print(f"PCA 산점도: {PCA_GRAPH_PATH.resolve()}")
print(f"승차 프로파일: {BOARDING_GRAPH_PATH.resolve()}")
print(f"하차 프로파일: {ALIGHTING_GRAPH_PATH.resolve()}")