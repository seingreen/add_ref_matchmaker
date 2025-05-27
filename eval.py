import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from matchmaker.features.audio import FRAME_RATE

TOLERANCES = [50, 100, 300, 500, 1000, 2000]

def transfer_positions(wp, ref_anns, frame_rate, reverse=False):
    if reverse:
        # 악보 시간 -> 참조 시간 (역방향)
        x, y = wp[1], wp[0]
        print("  Direction: score time -> reference time")
    else:
        # 참조 시간 -> 악보 시간 (정방향)
        x, y = wp[0], wp[1]
        print("  Direction: reference time -> score time")
    
    # 초 단위 어노테이션을 프레임 인덱스로 변환
    ref_anns_frame = np.round(ref_anns * frame_rate).astype(int)
    print(f"  annotations as frames: {ref_anns_frame[:5]}")

    predicted_targets = []
    for frame in ref_anns_frame:
        # 가장 가까운 프레임 찾기 (이전 방법: >= 조건)
        distances = np.abs(x - frame)
        closest_idx = np.argmin(distances)
        if closest_idx < len(y):
            predicted_targets.append(y[closest_idx])
        else:
            # 인덱스가 범위를 벗어나면 마지막 값 사용
            predicted_targets.append(y[-1])

    result = np.array(predicted_targets) / frame_rate
    print(f"  output annotations count: {len(result)}")
    print(f"  first few output annotations (seconds): {result[:5]}")
    
    return result

def transfer_from_score_to_predicted_perf(wp, score_annots, frame_rate):
    predicted_perf_idx = transfer_positions(wp, score_annots, frame_rate)
    return predicted_perf_idx


def transfer_from_perf_to_predicted_score(wp, perf_annots, frame_rate):
    predicted_score_idx = transfer_positions(wp, perf_annots, frame_rate, reverse=True)
    return predicted_score_idx

'''def get_evaluation_results(
    ref_annots,         # 참조 어노테이션 (그라운드 트루스)
    perf_annots,        # 실행 어노테이션
    total_length,       # 전체 길이
    warping_path=None,  # DTW 워핑 경로
    frame_rate=None,    # 프레임 레이트
    tolerances=TOLERANCES,
    in_seconds=True
):
    """
    DTW 워핑 경로를 사용하여 참조 어노테이션 지점에서 알고리즘의 정확도를 평가합니다.
    
    Parameters:
    -----------
    ref_annots : array-like
        참조 어노테이션 (그라운드 트루스, 초 단위)
    perf_annots : array-like
        실행 어노테이션 (초 단위)
    total_length : int
        전체 길이
    warping_path : tuple
        (참조 프레임, 실행 프레임) 형태의 워핑 경로
    frame_rate : int
        프레임 레이트
    tolerances : list
        허용 오차 임계값 목록 (밀리초)
    in_seconds : bool
        어노테이션이 초 단위인지 여부
    
    Returns:
    --------
    dict
        평가 지표를 포함하는 사전
    """
    print(f"\n=== {len(ref_annots)}개의 참조 어노테이션을 사용한 DTW 평가 ===")
    
    if warping_path is not None and frame_rate is not None:
        # 워핑 경로를 사용하여 참조 어노테이션 시간에 해당하는 실행 시간 계산
        ref_frames = np.round(np.array(ref_annots) * frame_rate).astype(int)
        predicted_perf_times = []
        
        ref_path_frames = warping_path[0]
        perf_path_frames = warping_path[1]
        
        for ref_frame in ref_frames:
            # 가장 가까운 참조 프레임 찾기
            closest_idx = np.argmin(np.abs(ref_path_frames - ref_frame))
            # 해당 인덱스의 실행 프레임
            matching_perf_frame = perf_path_frames[closest_idx]
            # 프레임을 시간으로 변환
            matching_perf_time = matching_perf_frame / frame_rate
            predicted_perf_times.append(matching_perf_time)
        
        predicted_perf_times = np.array(predicted_perf_times)
        
        # 예측된 실행 시간과 실제 실행 어노테이션 비교
        # 각 참조 어노테이션에 대해 가장 가까운 실행 어노테이션 찾기
        actual_perf_times = []
        for ref_time in ref_annots:
            # 가장 가까운 실행 어노테이션 찾기
            closest_idx = np.argmin(np.abs(np.array(perf_annots) - ref_time))
            actual_perf_times.append(perf_annots[closest_idx])
        
        actual_perf_times = np.array(actual_perf_times)
        
        # 예측된 시간과 실제 시간 간의 오차 계산
        if in_seconds:
            errors_in_delay = (predicted_perf_times - actual_perf_times) * 1000  # 밀리초 단위
        else:
            errors_in_delay = predicted_perf_times - actual_perf_times
    else:
        # 워핑 경로가 없으면 단순 비교
        min_length = min(len(ref_annots), len(perf_annots))
        ref_annots = ref_annots[:min_length]
        perf_annots = perf_annots[:min_length]
        
        if in_seconds:
            errors_in_delay = (np.array(ref_annots) - np.array(perf_annots)) * 1000  # 밀리초 단위
        else:
            errors_in_delay = np.array(ref_annots) - np.array(perf_annots)
    
    # 오차 통계 계산
    abs_errors = np.abs(errors_in_delay)
    filtered_errors = abs_errors[abs_errors <= tolerances[-1]]
    
    # 기본 통계
    results = {
        "mean": float(f"{np.nanmean(filtered_errors):.4f}"),
        "median": float(f"{np.nanmedian(filtered_errors):.4f}"),
        "std": float(f"{np.nanstd(filtered_errors):.4f}"),
    }
    
    # scipy 통계 함수 안전하게 처리
    try:
        results["skewness"] = float(f"{scipy.stats.skew(errors_in_delay):.4f}")
        results["kurtosis"] = float(f"{scipy.stats.kurtosis(errors_in_delay):.4f}")
    except:
        results["skewness"] = float("nan")
        results["kurtosis"] = float("nan")
    
    # 각 허용 오차 수준에서의 정확도
    for tau in tolerances:
        tau_str = f"{tau}ms" if in_seconds else f"{tau}"
        results[tau_str] = float(f"{np.sum(abs_errors <= tau) / len(abs_errors):.4f}")
    
    results["count"] = len(filtered_errors)
    results["total_count"] = len(abs_errors)
    results["pcr"] = results[f"{tolerances[-1]}ms"] if in_seconds else results[f"{tolerances[-1]}"]
    
    return results'''

def get_evaluation_results(
    ref_annots,         # 참조 어노테이션 (그라운드 트루스)
    perf_annots,        # 실행 어노테이션
    total_length,       # 전체 길이
    warping_path=None,  # DTW 워핑 경로
    frame_rate=None,    # 프레임 레이트
    tolerances=TOLERANCES,
    in_seconds=True
):
    """
    DTW 워핑 경로를 사용하여 참조 어노테이션 지점에서 알고리즘의 정확도를 평가합니다.
    
    Parameters:
    -----------
    ref_annots : array-like
        참조 어노테이션 (그라운드 트루스, 초 단위)
    perf_annots : array-like
        실행 어노테이션 (초 단위)
    total_length : int
        전체 길이
    warping_path : tuple
        (참조 프레임, 실행 프레임) 형태의 워핑 경로
    frame_rate : int
        프레임 레이트
    tolerances : list
        허용 오차 임계값 목록 (밀리초)
    in_seconds : bool
        어노테이션이 초 단위인지 여부
    
    Returns:
    --------
    dict
        평가 지표를 포함하는 사전
    """
    print(f"\n=== {len(ref_annots)}개의 참조 어노테이션을 사용한 DTW 평가 ===")
    
    if warping_path is not None and frame_rate is not None:
        # 워핑 경로를 사용하여 참조 어노테이션 시간에 해당하는 실행 시간 계산
        ref_frames = np.round(np.array(ref_annots) * frame_rate).astype(int)
        predicted_perf_times = []
        
        ref_path_frames = warping_path[0]
        perf_path_frames = warping_path[1]
        
        for ref_frame in ref_frames:
            # 가장 가까운 참조 프레임 찾기
            closest_idx = np.argmin(np.abs(ref_path_frames - ref_frame))
            # 해당 인덱스의 실행 프레임
            matching_perf_frame = perf_path_frames[closest_idx]
            # 프레임을 시간으로 변환
            matching_perf_time = matching_perf_frame / frame_rate
            predicted_perf_times.append(matching_perf_time)
        
        predicted_perf_times = np.array(predicted_perf_times)
        
        # 수정된 부분: DTW로 매핑된 예측 시간과 실제 성능 어노테이션 비교
        errors_in_delay = []
        for pred_time in predicted_perf_times:
            # 예측 시간과 가장 가까운 실제 성능 어노테이션 찾기
            closest_idx = np.argmin(np.abs(np.array(perf_annots) - pred_time))
            closest_perf_time = perf_annots[closest_idx]
            
            # 오차 계산
            if in_seconds:
                error = (pred_time - closest_perf_time) * 1000  # 밀리초 단위
            else:
                error = pred_time - closest_perf_time
                
            errors_in_delay.append(error)
            
        errors_in_delay = np.array(errors_in_delay)
    else:
        # 워핑 경로가 없으면 단순 비교
        min_length = min(len(ref_annots), len(perf_annots))
        ref_annots = ref_annots[:min_length]
        perf_annots = perf_annots[:min_length]
        
        if in_seconds:
            errors_in_delay = (np.array(ref_annots) - np.array(perf_annots)) * 1000  # 밀리초 단위
        else:
            errors_in_delay = np.array(ref_annots) - np.array(perf_annots)
    
    # 오차 통계 계산
    abs_errors = np.abs(errors_in_delay)
    filtered_errors = abs_errors[abs_errors <= tolerances[-1]]
    
    # 기본 통계
    results = {
        "mean": float(f"{np.nanmean(filtered_errors):.4f}"),
        "median": float(f"{np.nanmedian(filtered_errors):.4f}"),
        "std": float(f"{np.nanstd(filtered_errors):.4f}"),
    }
    
    # scipy 통계 함수 안전하게 처리
    try:
        results["skewness"] = float(f"{scipy.stats.skew(errors_in_delay):.4f}")
        results["kurtosis"] = float(f"{scipy.stats.kurtosis(errors_in_delay):.4f}")
    except:
        results["skewness"] = float("nan")
        results["kurtosis"] = float("nan")
    
    # 각 허용 오차 수준에서의 정확도
    for tau in tolerances:
        tau_str = f"{tau}ms" if in_seconds else f"{tau}"
        results[tau_str] = float(f"{np.sum(abs_errors <= tau) / len(abs_errors):.4f}")
    
    results["count"] = len(filtered_errors)
    results["total_count"] = len(abs_errors)
    results["pcr"] = results[f"{tolerances[-1]}ms"] if in_seconds else results[f"{tolerances[-1]}"]
    
    # 디버깅을 위한 시각화 함수 호출 (선택적)
    visualize_evaluation(predicted_perf_times, perf_annots, errors_in_delay)
    
    return results

def visualize_evaluation(predicted_times, actual_annots, errors):
    """
    DTW 매핑 결과와 실제 어노테이션 비교를 시각화합니다.
    """
    plt.figure(figsize=(12, 6))
    
    # 실제 어노테이션
    plt.scatter(actual_annots, np.zeros_like(actual_annots), color='red', label='Performance Annotation', s=50, marker='x')
    
    # DTW로 매핑된 예측
    plt.scatter(predicted_times, np.zeros_like(predicted_times) + 0.1, color='blue', label='DTW Predctions', s=50, alpha=0.7)
    
    # 연결선
    for pred in predicted_times:
        closest_idx = np.argmin(np.abs(np.array(actual_annots) - pred))
        closest_annot = actual_annots[closest_idx]
        plt.plot([pred, closest_annot], [0.1, 0], 'k-', alpha=0.3)
    
    plt.legend()
    plt.xlabel('Time (s)')
    plt.title('DTW results and annotation comparison')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('dtw_evaluation.png', dpi=300)
    plt.close()

def save_nparray_to_csv(array: NDArray, save_path: str):
    with open(save_path, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerows(array)

def save_score_following_result(model, save_dir, ref_annots, perf_ann_path, frame_rate, name=None, score_reference_wp=None):
    """
    참조 어노테이션만을 그라운드 트루스로 사용하여 시각화합니다.
    
    Parameters:
    -----------
    model : object
        추적 모델 (warping_path, reference_features, input_features 속성 포함)
    save_dir : str
        결과를 저장할 디렉토리
    ref_annots : array-like
        참조 어노테이션 (초 단위)
    perf_ann_path : str
        실행 어노테이션 파일 경로 (평가에는 사용되지만 시각화에는 사용되지 않음)
    frame_rate : int
        프레임 레이트
    name : str, optional
        저장할 파일의 기본 이름
    score_reference_wp : array-like, optional
        악보-참조 워핑 경로 (사용 가능한 경우)
    """
    run_name = name or "results"
    save_path = os.path.join(save_dir, f"wp_{run_name}.tsv")
    np.savetxt(save_path, model.warping_path.T, delimiter="\t")
    
    # 전체 워핑 경로 사용
    warp_path_x = model.warping_path[0]  # 참조 프레임
    warp_path_y = model.warping_path[1]  # 실행 프레임
    
    # 최대 프레임 계산 (제한 없음)
    max_ref_frame = max(np.max(warp_path_x) + 1, int(np.max(ref_annots) * frame_rate) + 1)
    max_perf_frame = np.max(warp_path_y) + 1
    
    # 시각화를 위한 거리 행렬 계산
    dist = scipy.spatial.distance.cdist(
        model.reference_features[:max_ref_frame],
        model.input_features[:max_perf_frame],
        metric=model.distance_func,
    )
    
    plt.figure(figsize=(12, 10))
    plt.imshow(dist, aspect="auto", origin="lower", interpolation="nearest")
    plt.title(f"[{save_dir}] \nDTW Results", fontsize=15)
    plt.xlabel("Performance Audio Frame", fontsize=12)
    plt.ylabel("Reference Audio Frame", fontsize=12)
    
    # 전체 DTW 경로 시각화 (노란색 점)
    for n in range(len(warp_path_x)):
        plt.plot(warp_path_y[n], warp_path_x[n], ".", color="yellow", alpha=0.5, markersize=6)
    
    # 참조 어노테이션 (그라운드 트루스) 시각화 (빨간색 X)
    for i in range(len(ref_annots)):
        ref_frame = ref_annots[i] * frame_rate
        # 어노테이션 참조 프레임에 해당하는 실행 프레임 찾기
        # 가장 가까운 참조 프레임 인덱스 찾기
        closest_idx = np.argmin(np.abs(warp_path_x - ref_frame))
        matching_perf_frame = warp_path_y[closest_idx]
        
        # 참조 어노테이션을 빨간색 X로 표시 (워핑 경로 상의 매핑 지점)
        plt.plot(matching_perf_frame, ref_frame, "x", color="red", alpha=1, markersize=10)
    
    # 범례 추가
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='.', color='yellow', label='Warpng Path', 
               markerfacecolor='yellow', markersize=8, linestyle=''),
        Line2D([0], [0], marker='x', color='red', label='Reference Annotations', 
               markerfacecolor='red', markersize=10, linestyle='')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    
    output_png_path = os.path.join(save_dir, f"{run_name}.png")
    plt.savefig(output_png_path, dpi=150, bbox_inches='tight')
    print(f"online_dtw 결과가 {output_png_path}에 저장되었습니다")
    plt.close()

'''def get_evaluation_results(
    gt_annots,
    predicted_annots,
    total_length,
    tolerances=TOLERANCES,
    in_seconds=True,
):
    # 배열 크기 확인 및 맞추기
    if len(gt_annots) != len(predicted_annots):
        print(f"Warning: Arrays have different sizes - gt: {len(gt_annots)}, pred: {len(predicted_annots)}")
        min_length = min(len(gt_annots), len(predicted_annots))
        gt_annots = gt_annots[:min_length]
        predicted_annots = predicted_annots[:min_length]
        print(f"Using first {min_length} elements from both arrays")
    
    # 시간 오프셋 보정 (첫 번째 지점 기준)
    if len(gt_annots) > 0 and len(predicted_annots) > 0:
        offset = gt_annots[0] - predicted_annots[0]
        predicted_annots_adjusted = predicted_annots + offset
        print(f"Applied time offset: {offset:.4f} seconds")
    else:
        predicted_annots_adjusted = predicted_annots
        
    if in_seconds:
        errors_in_delay = (gt_annots - predicted_annots_adjusted) * 1000  # in milliseconds
    else:
        errors_in_delay = gt_annots - predicted_annots_adjusted

    # 필터링 전 전체 오차 통계 출력
    print(f"Raw error stats - mean: {np.mean(np.abs(errors_in_delay)):.2f}ms, median: {np.median(np.abs(errors_in_delay)):.2f}ms")
    
    filtered_errors_in_delay = errors_in_delay[
        np.abs(errors_in_delay) <= tolerances[-1]
    ]
    filtered_abs_errors_in_delay = np.abs(filtered_errors_in_delay)

    results = {
        "mean": float(f"{np.nanmean(filtered_abs_errors_in_delay):.4f}"),
        "median": float(f"{np.nanmedian(filtered_abs_errors_in_delay):.4f}"),
        "std": float(f"{np.nanstd(filtered_abs_errors_in_delay):.4f}"),
        "skewness": float(f"{scipy.stats.skew(filtered_errors_in_delay):.4f}"),
        "kurtosis": float(f"{scipy.stats.kurtosis(filtered_errors_in_delay):.4f}"),
    }
    
    # 평가 지표 계산 - 어노테이션 개수 기준
    valid_count = len(gt_annots)
    for tau in tolerances:
        if in_seconds:
            results[f"{tau}ms"] = float(
                f"{np.sum(np.abs(errors_in_delay) <= tau) / valid_count:.4f}"
            )
        else:
            results[f"{tau}"] = float(
                f"{np.sum(np.abs(errors_in_delay) <= tau) / valid_count:.4f}"
            )
    
    results["count"] = len(filtered_abs_errors_in_delay)
    results["pcr"] = results[f"{tolerances[-1]}ms"]
    
    # 시간 오프셋 정보 저장
    results["offset_ms"] = float(f"{offset * 1000:.4f}") if 'offset' in locals() else 0.0
    
    return results

def save_nparray_to_csv(array: NDArray, save_path: str):
    with open(save_path, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerows(array)


def save_score_following_result(model, save_dir, ref_annots, perf_ann_path, frame_rate, name=None, score_reference_wp=None):
    run_name = name or "results"
    save_path = os.path.join(save_dir, f"wp_{run_name}.tsv")
    np.savetxt(save_path, model.warping_path.T, delimiter="\t")
    
    # 350 프레임까지만 사용할 수 있도록 자르기
    max_frames = 350
    max_time = max_frames / frame_rate  # 350프레임에 해당하는 시간 (초 단위)

    # 전체 프레임에서 350 프레임까지만 자르기
    warp_path_x = model.warping_path[0][:max_frames]
    warp_path_y = model.warping_path[1][:max_frames]

    dist = scipy.spatial.distance.cdist(
        model.reference_features[:max_frames],
        model.input_features[:max_frames],
        #model.input_features[: model.warping_path[1][-1]],
        metric=model.distance_func,
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(dist, aspect="auto", origin="lower", interpolation="nearest")
    plt.title(f"[{save_dir}] \n Matchmaker alignment path with ground-truth labels", fontsize=15)
    plt.xlabel("Performance Audio frame", fontsize=12)
    plt.ylabel("Reference Audio frame", fontsize=12)

    # Online DTW path plotting (350 프레임까지만)
    for n in range(len(warp_path_x)):
        if warp_path_x[n] < max_frames and warp_path_y[n] < max_frames:  # 350x350 영역 내에서만 표시
            plt.plot(warp_path_y[n], warp_path_x[n], ".", color="yellow", alpha=0.5, markersize=8)
    # Online DTW path plotting
    #for n in range(len(model.warping_path[0])):
    #    plt.plot(model.warping_path[1][n], model.warping_path[0][n], ".", color="yellow", alpha=0.5, markersize=8)

    # Load performance annotations and match to reference annotations
    with open(perf_ann_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        perf_annots = [float(row[0]) for row in reader]

    # Matching annotations plotting
    min_length = min(len(ref_annots), len(perf_annots), max_frames)

    for i in range(min_length):
        # Mapping the performance annotations to reference annotations
        perf_time = perf_annots[i] * frame_rate
        if perf_time < max_frames:
            plt.plot(perf_time, perf_time, "x", color="red", alpha=1, markersize=8)
    
    # 참조 어노테이션을 파란색 "o"로 표시
    for i in range(min_length):
        ref_time = ref_annots[i] * frame_rate
        if ref_time < max_frames:
            plt.plot(ref_time, ref_time, "o", color="blue", alpha=1, markersize=8)

    output_png_path = os.path.join(save_dir, f"{run_name}.png")
    plt.savefig(output_png_path)
    print(f"Plot saved at {output_png_path}")
    plt.close()'''


'''def get_evaluation_results(score_annots, perf_annots, warping_path, frame_rate, tolerance=TOLERANCES, score_reference_wp=None, ref_annots=None):
    """
    평가 결과를 계산합니다.
    """
    if ref_annots is not None:
        print("Using direct comparison between performance and reference annotations")
        min_length = min(len(perf_annots), len(ref_annots))
        perf_annots_used = perf_annots[:min_length]
        ref_annots_used = ref_annots[:min_length]
        
        errors_in_delay = (np.array(perf_annots_used) - np.array(ref_annots_used)) * 1000
    else:
        target_annots_predicted = transfer_positions(
            warping_path, score_annots, frame_rate
        )
        min_length = min(len(score_annots), len(perf_annots))
        score_annots = score_annots[:min_length]
        perf_annots = perf_annots[:min_length]
        target_annots_predicted = target_annots_predicted[:min_length]
    
        errors_in_delay = (perf_annots - target_annots_predicted) * 1000
    
    absolute_errors_in_delay = np.abs(errors_in_delay)
    filtered_abs_errors_in_delay = absolute_errors_in_delay[
        absolute_errors_in_delay <= tolerance[-1]
    ]
    
    results = {
        "mean": float(f"{np.mean(filtered_abs_errors_in_delay):.4f}"),
        "median": float(f"{np.median(filtered_abs_errors_in_delay):.4f}"),
        "std": float(f"{np.std(filtered_abs_errors_in_delay):.4f}"),
        "skewness": float(f"{scipy.stats.skew(filtered_abs_errors_in_delay):.4f}"),
        "kurtosis": float(f"{scipy.stats.kurtosis(filtered_abs_errors_in_delay):.4f}"),
    }
    
    for tau in tolerance:
        results[f"{tau}ms"] = float(f"{np.mean(absolute_errors_in_delay <= tau):.4f}")
    
    results["count"] = len(filtered_abs_errors_in_delay)
    return results'''

'''def get_evaluation_results(score_annots, perf_annots, warping_path, frame_rate, tolerance=TOLERANCES, score_reference_wp=None, ref_annots=None):
    """
    평가 결과를 계산합니다. 필터링을 제거한 더 단순한 버전.
    """
    # 만약 참조 어노테이션이 존재하면, 참조 어노테이션과 공연 어노테이션을 직접 비교
    if ref_annots is not None and len(ref_annots) > 0:
        print("\n** 직접 비교 모드: 참조 어노테이션과 공연 어노테이션 **")
        print(f"참조 어노테이션 수: {len(ref_annots)}")
        print(f"공연 어노테이션 수: {len(perf_annots)}")
        
        # 어노테이션 길이 맞추기
        min_length = min(len(perf_annots), len(ref_annots))
        perf_annots_used = perf_annots[:min_length]
        ref_annots_used = ref_annots[:min_length]
        
        # 시각화와 비슷한 평가 방법 - 프레임 비율 사용
        print(f"참조 어노테이션과 공연 어노테이션의 시간 범위:")
        print(f"  참조: {min(ref_annots_used):.2f}s ~ {max(ref_annots_used):.2f}s")
        print(f"  공연: {min(perf_annots_used):.2f}s ~ {max(perf_annots_used):.2f}s")
        
        # 시간 대신 비율 사용 - 시각화처럼 기울기 기반으로 평가
        # 시간을 0-1 범위로 정규화
        ref_normalized = (ref_annots_used - min(ref_annots_used)) / (max(ref_annots_used) - min(ref_annots_used))
        perf_normalized = (perf_annots_used - min(perf_annots_used)) / (max(perf_annots_used) - min(perf_annots_used))
        
        # 정규화된 시간의 차이를 오차로 사용
        errors_in_delay = (ref_normalized - perf_normalized) * 1000
        print(f"정규화 기반 오차 (처음 5개): {errors_in_delay[:5]}")
    else:
        # 기존 로직: 점수 어노테이션을 워핑 패스로 변환 후 공연 어노테이션과 비교
        print("\n** 기존 비교 모드: 변환된 점수 어노테이션과 공연 어노테이션 **")
        target_annots_predicted = transfer_positions(
            warping_path, score_annots, frame_rate
        )
        min_length = min(len(score_annots), len(perf_annots))
        score_annots = score_annots[:min_length]
        perf_annots = perf_annots[:min_length]
        target_annots_predicted = target_annots_predicted[:min_length]
    
        errors_in_delay = (perf_annots - target_annots_predicted) * 1000
    
    # 절대 오차 계산
    absolute_errors_in_delay = np.abs(errors_in_delay)
    
    # 필터링 제거 - 모든 데이터 포인트 사용
    filtered_abs_errors_in_delay = absolute_errors_in_delay
    
    # 통계 계산
    results = {
        "mean": float(f"{np.mean(filtered_abs_errors_in_delay):.4f}"),
        "median": float(f"{np.median(filtered_abs_errors_in_delay):.4f}"),
        "std": float(f"{np.std(filtered_abs_errors_in_delay):.4f}"),
    }
    
    # scipy 통계 함수 안전하게 처리
    try:
        results["skewness"] = float(f"{scipy.stats.skew(filtered_abs_errors_in_delay):.4f}")
        results["kurtosis"] = float(f"{scipy.stats.kurtosis(filtered_abs_errors_in_delay):.4f}")
    except:
        results["skewness"] = float("nan")
        results["kurtosis"] = float("nan")
    
    # 각 허용 오차별 정확도 계산
    for tau in tolerance:
        results[f"{tau}ms"] = float(f"{np.mean(absolute_errors_in_delay <= tau):.4f}")
    
    # 사용된 데이터 포인트 수
    results["count"] = len(filtered_abs_errors_in_delay)
    
    return results'''

'''def get_evaluation_results(score_annots, perf_annots, warping_path, frame_rate, tolerance=TOLERANCES, score_reference_wp=None, ref_annots=None):
    """
    시각화와 동일한 방식으로 평가합니다.
    """
    # 참조 어노테이션이 존재하면
    if ref_annots is not None and len(ref_annots) > 0:
        print("\n** 시각화 일치 평가 모드 **")
        print(f"참조 어노테이션 수: {len(ref_annots)}")
        print(f"공연 어노테이션 수: {len(perf_annots)}")
        
        # 어노테이션을 프레임으로 변환
        ref_frames = np.round(np.array(ref_annots) * frame_rate).astype(int)
        perf_frames = np.round(np.array(perf_annots) * frame_rate).astype(int)
        
        print(f"프레임 변환 후:")
        print(f"  참조 프레임 범위: {np.min(ref_frames)} ~ {np.max(ref_frames)}")
        print(f"  공연 프레임 범위: {np.min(perf_frames)} ~ {np.max(perf_frames)}")
        
        wp_ref_frames = warping_path[0]
        wp_perf_frames = warping_path[1]
        
        min_length = min(len(ref_frames), len(perf_frames))
        ref_frames_used = ref_frames[:min_length]
        perf_frames_used = perf_frames[:min_length]
        
        frame_errors = []
        
        for i in range(min_length):
            ref_frame = ref_frames_used[i]
            perf_frame = perf_frames_used[i]
            
            distances = []
            for j in range(len(wp_ref_frames)):
                dist = np.sqrt((wp_ref_frames[j] - ref_frame)**2 + (wp_perf_frames[j] - perf_frame)**2)
                distances.append(dist)
            
            min_dist = np.min(distances)
            frame_errors.append(min_dist)
        
        # 결과 출력
        print(f"처음 5개 참조 프레임: {ref_frames_used[:5]}")
        print(f"처음 5개 공연 프레임: {perf_frames_used[:5]}")
        print(f"처음 5개 프레임 오차(워핑 패스와의 거리): {frame_errors[:5] if len(frame_errors) >= 5 else frame_errors}")
        
        # 프레임 오차를 밀리초로 변환 (단순 시각화 목적)
        ms_errors = np.array(frame_errors) * (1000 / frame_rate)
        print(f"처음 5개 밀리초 오차: {ms_errors[:5] if len(ms_errors) >= 5 else ms_errors}")
        
        # 오차 값으로 사용
        errors_in_delay = ms_errors
    else:
        # 기존 로직: 점수 어노테이션을 워핑 패스로 변환 후 공연 어노테이션과 비교
        print("\n** 기존 비교 모드: 변환된 점수 어노테이션과 공연 어노테이션 **")
        target_annots_predicted = transfer_positions(
            warping_path, score_annots, frame_rate
        )
        min_length = min(len(score_annots), len(perf_annots))
        score_annots = score_annots[:min_length]
        perf_annots = perf_annots[:min_length]
        target_annots_predicted = target_annots_predicted[:min_length]
    
        errors_in_delay = (perf_annots - target_annots_predicted) * 1000
    
    # 절대 오차 계산
    absolute_errors_in_delay = np.abs(errors_in_delay)
    
    # 통계 계산
    results = {
        "mean": float(f"{np.mean(absolute_errors_in_delay):.4f}"),
        "median": float(f"{np.median(absolute_errors_in_delay):.4f}"),
        "std": float(f"{np.std(absolute_errors_in_delay):.4f}"),
    }
    
    # 프레임 기반 통계 추가 (프레임 단위)
    if 'frame_errors' in locals() and len(frame_errors) > 0:
        results["mean_frame_error"] = float(f"{np.mean(frame_errors):.4f}")
        results["median_frame_error"] = float(f"{np.median(frame_errors):.4f}")
        results["max_frame_error"] = float(f"{np.max(frame_errors):.4f}")
    
    # scipy 통계 함수 안전하게 처리
    try:
        results["skewness"] = float(f"{scipy.stats.skew(absolute_errors_in_delay):.4f}")
        results["kurtosis"] = float(f"{scipy.stats.kurtosis(absolute_errors_in_delay):.4f}")
    except:
        results["skewness"] = float("nan")
        results["kurtosis"] = float("nan")
    
    # 각 허용 오차별 정확도 계산
    for tau in tolerance:
        results[f"{tau}ms"] = float(f"{np.mean(absolute_errors_in_delay <= tau):.4f}")
    
    # 프레임 기반 정확도 계산
    if 'frame_errors' in locals() and len(frame_errors) > 0:
        frame_tolerances = [1, 3, 5, 10, 20, 30]
        for ftau in frame_tolerances:
            results[f"{ftau}frames"] = float(f"{np.mean(np.array(frame_errors) <= ftau):.4f}")
    
    # 사용된 데이터 포인트 수
    results["count"] = len(absolute_errors_in_delay)
    
    return results'''

'''
def save_score_following_result(model, save_dir, ref_annots, perf_ann_path, frame_rate, name=None, score_reference_wp=None):
    run_name = name or "results"
    save_path = os.path.join(save_dir, f"wp_{run_name}.tsv")
    np.savetxt(save_path, model.warping_path.T, delimiter='\t')

    dist = scipy.spatial.distance.cdist(
        model.reference_features,
        model.input_features[: model.warping_path[1][-1]],
        metric=model.distance_func,
    )

    # 시각화: 거리 행렬
    plt.figure(figsize=(15, 15))
    plt.imshow(dist, aspect="auto", origin="lower", interpolation="nearest")
    plt.title(f"[{save_dir}] \n Matchmaker alignment path with ground-truth labels", fontsize=25)
    plt.xlabel("Performance Audio frame", fontsize=15)
    plt.ylabel("Reference Audio frame", fontsize=15)

    # Online DTW path plotting
    #for n in range(len(model.warping_path[0])):
    #    plt.plot(model.warping_path[1][n], model.warping_path[0][n], ".", color="yellow", alpha=0.5, markersize=8)

    # Performance annotations 로드
    with open(perf_ann_path, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        perf_annots = [float(row[0]) for row in reader]

    # Matching annotations plotting (350프레임까지만)
    min_length = min(len(ref_annots), len(perf_annots), max_frames)
    for i in range(min_length):
        # Mapping the performance annotations to reference annotations
        perf_time = perf_annots[i] * frame_rate
        ref_time = ref_annots[i] * frame_rate  # Assuming ref_annots already matched or transformed to perf_annots times
        plt.plot(perf_time, ref_time, "x", color="red", alpha=1, markersize=8)

    # 저장 경로 설정 및 PNG 파일로 저장
    output_png_path = os.path.join(save_dir, f"{run_name}.png")
    plt.savefig(output_png_path)
    print(f"Plot saved at {output_png_path}")
    plt.close()'''
