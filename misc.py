#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Miscellaneous utilities
"""

import numbers
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Iterable, List, Union

import librosa
import numpy as np
import partitura
import soundfile as sf
import matplotlib.pyplot as plt
import matplotlib
from partitura.io.exportmidi import get_ppq
from partitura.score import ScoreLike
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw 
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features
from synctoolbox.feature.utils import estimate_tuning
from scipy.spatial.distance import cdist

from matchmaker.features.audio import (
    FRAME_RATE,
    SAMPLE_RATE,
    HOP_LENGTH,
)
class MatchmakerInvalidParameterTypeError(Exception):
    """
    Error for flagging an invalid parameter type.
    """

    def __init__(
        self,
        parameter_name: str,
        required_parameter_type: Union[type, Iterable[type]],
        actual_parameter_type: type,
        *args,
    ) -> None:
        if isinstance(required_parameter_type, Iterable):
            rqpt = ", ".join([f"{pt}" for pt in required_parameter_type])
        else:
            rqpt = required_parameter_type
        message = f"`{parameter_name}` was expected to be {rqpt}, but is {actual_parameter_type}"

        super().__init__(message, *args)


class MatchmakerInvalidOptionError(Exception):
    """
    Error for invalid option.
    """

    def __init__(self, parameter_name, valid_options, value, *args) -> None:
        rqop = ", ".join([f"{op}" for op in valid_options])
        message = f"`{parameter_name}` was expected to be in {rqop}, but is {value}"

        super().__init__(message, *args)


class MatchmakerMissingParameterError(Exception):
    """
    Error for flagging a missing parameter
    """

    def __init__(self, parameter_name: Union[str, List[str]], *args) -> None:
        if isinstance(parameter_name, Iterable) and not isinstance(parameter_name, str):
            message = ", ".join([f"`{pn}`" for pn in parameter_name])
            message = f"{message} were not given"
        else:
            message = f"`{parameter_name}` was not given."
        super().__init__(message, *args)


def ensure_rng(
    seed: Union[numbers.Integral, np.random.RandomState],
) -> np.random.RandomState:
    """
    Ensure random number generator is a np.random.RandomState instance

    Parameters
    ----------
    seed : int or np.random.RandomState
        An integer to serve as the seed for the random number generator or a
        `np.random.RandomState` instance.

    Returns
    -------
    rng : np.random.RandomState
        A random number generator.
    """

    if isinstance(seed, numbers.Integral):
        rng = np.random.RandomState(seed)
        return rng
    elif isinstance(seed, np.random.RandomState):
        rng = seed
        return rng
    else:
        raise ValueError(
            "`seed` should be an integer or an instance of "
            f"`np.random.RandomState` but is {type(seed)}"
        )


class RECVQueue(Queue):
    """
    Queue with a recv method (like Pipe)

    This class uses python's Queue.get with a timeout makes it interruptable via KeyboardInterrupt
    and even for the future where that is possibly out-dated, the interrupt can happen after each timeout
    so periodically query the queue with a timeout of 1s each attempt, finding a middleground
    between busy-waiting and uninterruptable blocked waiting
    """

    def __init__(self) -> None:
        Queue.__init__(self)

    def recv(self) -> Any:
        """
        Return and remove an item from the queue.
        """
        while True:
            try:
                return self.get(timeout=1)
            except Empty:  # pragma: no cover
                pass

    def poll(self) -> bool:
        return self.empty()


def get_window_indices(indices: np.ndarray, context: int) -> np.ndarray:
    # Create a range array from -context to context (inclusive)
    range_array = np.arange(-context, context + 1)

    # Reshape indices to be a column vector (len(indices), 1)
    indices = indices[:, np.newaxis]

    # Use broadcasting to add the range array to each index
    out_array = indices + range_array

    return out_array.astype(int)


def is_audio_file(file_path) -> bool:
    audio_extensions = {".wav", ".mp3", ".flac", ".aac", ".ogg", ".m4a"}
    ext = Path(file_path).suffix
    return ext.lower() in audio_extensions


def is_midi_file(file_path) -> bool:
    midi_extensions = {".mid", ".midi"}
    ext = Path(file_path).suffix
    return ext.lower() in midi_extensions


def interleave_with_constant(
    array: np.array,
    constant_row: float = 0,
) -> np.ndarray:
    """
    Interleave a matrix with rows of a constant value.

    Parameters
    -----------
    array : np.ndarray
    """
    # Determine the shape of the input array
    num_rows, num_cols = array.shape

    # Create an output array with interleaved rows (double the number of rows)
    interleaved_array = np.zeros((num_rows * 2, num_cols), dtype=array.dtype)

    # Set the odd rows to the original array and even rows to the constant_row
    interleaved_array[0::2] = array
    interleaved_array[1::2] = constant_row

    return interleaved_array


def adjust_tempo_for_performance_audio(score: ScoreLike, performance_audio: Path):
    """
    Adjust the tempo of the score part to match the performance audio.
    We round up the tempo to the nearest 20 bpm to avoid too much optimization.

    Parameters
    ----------
    score : partitura.score.ScoreLike
        The score to adjust the tempo of.
    performance_audio : Path
        The performance audio file to adjust the tempo to.
    """
    default_tempo = 120
    score_midi = partitura.save_score_midi(score, out=None)
    source_length = score_midi.length
    target_length = librosa.get_duration(path=str(performance_audio))
    ratio = target_length / source_length
    rounded_tempo = int(
        (default_tempo / ratio + 19) // 20 * 20
    )  # round up to nearest 20
    print(
        f"default tempo: {default_tempo} (score length: {source_length}) -> adjusted_tempo: {rounded_tempo} (perf length: {target_length})"
    )
    return rounded_tempo


def get_current_note_bpm(score: ScoreLike, onset_beat: float, tempo: float) -> float:
    """Get the adjusted BPM for a given note onset beat position based on time signature."""
    current_time = score.inv_beat_map(onset_beat)
    beat_type_changes = [
        {"start": time_sig.start, "beat_type": time_sig.beat_type}
        for time_sig in score.time_sigs
    ]

    # Find the latest applicable time signature change
    latest_change = next(
        (
            change
            for change in reversed(beat_type_changes)
            if current_time >= change["start"].t
        ),
        None,
    )

    # Return adjusted BPM if time signature change exists, else default tempo
    return latest_change["beat_type"] / 4 * tempo if latest_change else tempo


def generate_score_audio(score: ScoreLike, bpm: float, samplerate: int):
    bpm_array = [
        [onset_beat, get_current_note_bpm(score, onset_beat, bpm)]
        for onset_beat in score.note_array()["onset_beat"]
    ]
    bpm_array = np.array(bpm_array)
    soundfont_path = "/Users/maclab/workspace/matchmaker/aaviolin.sf2"
    score_audio = partitura.save_wav_fluidsynth(
        score,
        bpm=bpm_array,
        soundfont=soundfont_path,
        samplerate=samplerate,
    )

    first_onset_in_beat = score.note_array()["onset_beat"].min()
    first_onset_in_time = (
        score.inv_beat_map(first_onset_in_beat) / get_ppq(score) * (60 / bpm)
    )
    # add padding to the beginning of the score audio
    padding_size = int(first_onset_in_time * samplerate)
    score_audio = np.pad(score_audio, (padding_size, 0))

    last_onset_in_div = np.floor(score.note_array()["onset_div"].max())
    last_onset_in_time = last_onset_in_div / get_ppq(score) * (60 / bpm)

    buffer_size = 0.1  # for assuring the last onset is included (in seconds)
    last_onset_in_time += buffer_size
    score_audio = score_audio[: int(last_onset_in_time * samplerate)]
    return score_audio

def _get_DLNCO_features_from_audio(
    audio,
    tuning_offset,
    feature_sequence_length,
    Fs=SAMPLE_RATE,
    feature_rate=FRAME_RATE,
    verbose=False,
):
    f_pitch_onset = audio_to_pitch_onset_features(
        f_audio=audio, Fs=Fs, tuning_offset=tuning_offset, verbose=verbose
    )

    f_DLNCO = pitch_onset_features_to_DLNCO(
        f_peaks=f_pitch_onset,
        feature_rate=feature_rate,
        feature_sequence_length=feature_sequence_length,
        visualize=verbose,
    )

    return f_DLNCO

def run_offline_alignment(score_audio_path: Path, ref_audio_path: Path, method="synctoolbox"):
    """
    두 오디오 파일 간의 오프라인 정렬을 수행하고, 거리 행렬과 워핑 경로를 시각화합니다.
    
    매개변수:
    -----------
    score_audio_path : Path
        악보 오디오 파일 경로
    ref_audio_path : Path
        참조 오디오 파일 경로
    method : str
        사용할 방법 ("synctoolbox" 또는 "librosa")
        
    반환:
    --------
    wp : numpy.ndarray
        워핑 경로
    """
    # 파일 이름 추출
    score_name = Path(score_audio_path).stem
    ref_name = Path(ref_audio_path).stem
    
    # 오디오 파일 로드
    audio_1, _ = librosa.load(score_audio_path.as_posix(), sr=SAMPLE_RATE)
    audio_2, _ = librosa.load(ref_audio_path.as_posix() if isinstance(ref_audio_path, Path) else ref_audio_path, sr=SAMPLE_RATE)
    print(f"Loaded score audio {score_name} and reference audio {ref_name}")

    # 크로마 특성 생성
    f_chroma_librosa_1 = librosa.feature.chroma_cens(
        y=audio_1,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )
    f_chroma_librosa_2 = librosa.feature.chroma_cens(
        y=audio_2,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )

    # 시각화 파일 이름 생성
    vis_filename = f"alignment_{score_name}_vs_{ref_name}_{method}"

    if method == "synctoolbox":
        # 튜닝 오프셋 설정 (현재 비활성화됨)
        tuning_offset_1 = 0
        tuning_offset_2 = 0

        # DLNCO 특성 생성
        f_DLNCO_1 = _get_DLNCO_features_from_audio(
            audio=audio_1,
            tuning_offset=tuning_offset_1,
            feature_sequence_length=f_chroma_librosa_1.shape[1],
        )

        f_DLNCO_2 = _get_DLNCO_features_from_audio(
            audio=audio_2,
            tuning_offset=tuning_offset_2,
            feature_sequence_length=f_chroma_librosa_2.shape[1],
        )
        
        # MRMSDTW를 사용한 워핑 경로 계산
        wp = sync_via_mrmsdtw(
            f_chroma1=f_chroma_librosa_1,
            f_onset1=f_DLNCO_1,
            f_chroma2=f_chroma_librosa_2,
            f_onset2=f_DLNCO_2,
            input_feature_rate=FRAME_RATE,
            verbose=False,
        )
        
        # DLNCO 거리 행렬 시각화
        dlnco1_frames = f_DLNCO_1.T
        dlnco2_frames = f_DLNCO_2.T
        
        dlnco_dist_matrix = cdist(dlnco1_frames, dlnco2_frames, metric='euclidean')
        
        plt.figure(figsize=(15, 15))
        plt.imshow(dlnco_dist_matrix, origin='lower', cmap='plasma_r', aspect='auto', interpolation='nearest')
        plt.colorbar(label='Euclidean Distance')
        
        for i in range(len(wp[0])):
            plt.plot(wp[1][i], wp[0][i], '.', color='white', alpha=0.7, markersize=1)
        
        plt.xlabel('Score Frame', fontsize=14)
        plt.ylabel('Reference Audio Frame', fontsize=14)
        plt.title('DLNCO Distance Matrix with Warping Path', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(f'{vis_filename}_dlnco_distance.png')
        
    elif method == "librosa":
        # librosa의 DTW 사용
        D, wp_raw = librosa.sequence.dtw(
            f_chroma_librosa_1,  # (n_chroma, n_frames1)
            f_chroma_librosa_2,  # (n_chroma, n_frames2)
        )
        
        # 워핑 경로를 synctoolbox 형식으로 변환
        wp = wp_raw.T
        
        # 시각화: 누적 비용 행렬
        plt.figure(figsize=(15, 15))
        plt.imshow(D, origin='lower', cmap='viridis_r', aspect='auto', interpolation='nearest')
        plt.colorbar(label='Accumulated Cost')
        plt.title('DTW Accumulated Cost Matrix', fontsize=16)
        plt.xlabel('Score Frame', fontsize=14)
        plt.ylabel('Reference Audio Frame', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{vis_filename}_dtw_accumulated_cost.png')
    
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # 크로마 거리 행렬 + 워핑 경로
    chroma1_frames = f_chroma_librosa_1.T
    chroma2_frames = f_chroma_librosa_2.T
    
    dist_matrix = cdist(chroma1_frames, chroma2_frames, metric='cosine')
    
    plt.figure(figsize=(15, 15))
    plt.imshow(dist_matrix, origin='lower', cmap='viridis_r', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Cosine Distance')
    
    for i in range(len(wp[0])):
        plt.plot(wp[1][i], wp[0][i], '.', color='yellow', alpha=0.5, markersize=1)
    
    plt.xlabel('Score Frame', fontsize=14)
    plt.ylabel('Reference Audio Frame', fontsize=14)
    plt.title(f'Distance Matrix: {ref_name} vs {score_name} ({method})', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f'{vis_filename}_chroma_distance.png')
    
    # 워핑 경로 시각화
    plt.figure(figsize=(10, 8))
    plt.scatter(wp[1], wp[0], s=0.5, color='blue', alpha=0.7, marker='.')
    plt.xlabel('Score Frame', fontsize=14)
    plt.ylabel('Reference Audio Frame', fontsize=14)
    plt.title(f'Warping Path ({method})', fontsize=16)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{vis_filename}_warping_path.png')
    
    return wp


def generate_reference_annotations(ref_annots, frame_rate=FRAME_RATE):#warping_path, score_annots, frame_rate=FRAME_RATE):
    """
    워핑 패스를 사용하여 악보 어노테이션을 참조 오디오 어노테이션으로 변환합니다.
    
    매개변수:
    -----------
    warping_path : tuple of np.ndarray
        워핑 패스. warping_path[0]은 참조 프레임, warping_path[1]은 악보 프레임.
    score_annots : np.ndarray
        초 단위의 악보 어노테이션
    frame_rate : float
        프레임 레이트(Hz)
        
    반환:
    --------
    ref_annots : np.ndarray
        참조 오디오 어노테이션(초 단위)
    """
    '''ref_frames, score_frames = warping_path
    
    # 악보 어노테이션을 프레임으로 변환
    score_annots_frames = np.round(score_annots * frame_rate).astype(int)
    
    # 각 악보 어노테이션 프레임에 대해 참조 오디오 프레임 찾기
    ref_annots_frames = []
    for score_frame in score_annots_frames:
        idx = np.where(score_frames >= score_frame)[0]
        if len(idx) == 0:
            ref_frame = ref_frames[-1]
        else:
            ref_frame = ref_frames[idx[0]]
        ref_annots_frames.append(ref_frame)
    
    # 프레임을 시간(초)으로 변환
    ref_annots = np.array(ref_annots_frames) / frame_rate'''
    return ref_annots

def add_click_to_audio(audio_path, annotations, output_path=None, sr=SAMPLE_RATE):
    """
    오디오 파일에 어노테이션 시간에 클릭 사운드를 추가합니다.
    
    매개변수:
    -----------
    audio_path : str or Path
        오디오 파일 경로
    annotations : np.ndarray
        어노테이션 시간(초 단위)
    output_path : str or Path, optional
        출력 파일 경로. None이면 자동 생성합니다.
    sr : int
        샘플링 레이트(Hz)
        
    반환:
    --------
    output_path : str
        저장된 출력 파일 경로
    """
    
    print(f"\n=== 오디오에 클릭 추가 ===")
    print(f"오디오 파일: {audio_path}")
    print(f"어노테이션 포인트 수: {len(annotations)}")
    
    # 오디오 로드
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    
    # 어노테이션 기반 클릭 생성
    click_audio = librosa.clicks(
        times=annotations,
        sr=sr,
        click_freq=1000,  # 클릭 주파수 (Hz)
        length=len(audio)
    )
    
    # 오디오와 클릭 혼합
    audio_with_clicks = audio + click_audio
    
    # 출력 경로 설정
    if output_path is None:
        output_path = f"{Path(audio_path).stem}_with_clicks.wav"
    
    # 혼합된 오디오 저장
    sf.write(
        output_path,
        audio_with_clicks,
        sr,
        subtype="PCM_24"
    )
    print(f"클릭이 추가된 오디오가 {output_path}에 저장됨")
    
    return output_path

'''def _get_DLNCO_features_from_audio(
    audio,
    tuning_offset,
    feature_sequence_length,
    Fs=SAMPLE_RATE,
    feature_rate=FRAME_RATE,
    verbose=False,
):
    f_pitch_onset = audio_to_pitch_onset_features(
        f_audio=audio, Fs=Fs, tuning_offset=tuning_offset, verbose=verbose
    )

    f_DLNCO = pitch_onset_features_to_DLNCO(
        f_peaks=f_pitch_onset,
        feature_rate=feature_rate,
        feature_sequence_length=feature_sequence_length,
        visualize=verbose,
    )

    return f_DLNCO

def run_offline_alignment(score_audio_path: Path, ref_audio_path: Path, method="librosa"):
    """
    Perform offline alignment between two audio files and visualize the distance matrix and warping path.
    
    Parameters:
    -----------
    score_audio_path : Path
        Path to the score audio file
    ref_audio_path : Path
        Path to the reference audio file
    method : str
        Method to use ("synctoolbox" or "librosa")
        
    Returns:
    --------
    wp : numpy.ndarray
        Warping path
    """
    # Extract filenames
    score_name = Path(score_audio_path).stem
    ref_name = Path(ref_audio_path).stem
    
    # Load audio files
    audio_1, _ = librosa.load(score_audio_path.as_posix(), sr=SAMPLE_RATE)
    audio_2, _ = librosa.load(ref_audio_path.as_posix() if isinstance(ref_audio_path, Path) else ref_audio_path, sr=SAMPLE_RATE)

    # Generate chroma features
    f_chroma_librosa_1 = librosa.feature.chroma_cens(
        y=audio_1,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )
    f_chroma_librosa_2 = librosa.feature.chroma_cens(
        y=audio_2,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )

    # Create filename for visualizations
    vis_filename = f"alignment_{score_name}_vs_{ref_name}_{method}"

    if method == "synctoolbox":
        # Set tuning offset (currently disabled)
        tuning_offset_1 = 0
        tuning_offset_2 = 0

        # Generate DLNCO features
        f_DLNCO_1 = _get_DLNCO_features_from_audio(
            audio=audio_1,
            tuning_offset=tuning_offset_1,
            feature_sequence_length=f_chroma_librosa_1.shape[1],
        )

        f_DLNCO_2 = _get_DLNCO_features_from_audio(
            audio=audio_2,
            tuning_offset=tuning_offset_2,
            feature_sequence_length=f_chroma_librosa_2.shape[1],
        )
        
        # Calculate warping path using MRMSDTW
        wp = sync_via_mrmsdtw(
            f_chroma1=f_chroma_librosa_1,
            f_onset1=f_DLNCO_1,
            f_chroma2=f_chroma_librosa_2,
            f_onset2=f_DLNCO_2,
            input_feature_rate=FRAME_RATE,
            verbose=False,
        )
        
        # DLNCO distance matrix visualization
        dlnco1_frames = f_DLNCO_1.T
        dlnco2_frames = f_DLNCO_2.T
        
        dlnco_dist_matrix = cdist(dlnco1_frames, dlnco2_frames, metric='euclidean')
        
        plt.figure(figsize=(15, 15))
        plt.imshow(dlnco_dist_matrix, origin='lower', cmap='plasma_r', aspect='auto', interpolation='nearest')
        plt.colorbar(label='Euclidean Distance')
        
        for i in range(len(wp[0])):
            plt.plot(wp[1][i], wp[0][i], '.', color='white', alpha=0.7, markersize=1)
        
        plt.xlabel('Score Frame', fontsize=14)
        plt.ylabel('Reference Audio Frame', fontsize=14)
        plt.title('DLNCO Distance Matrix with Warping Path', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(f'{vis_filename}_dlnco_distance.png')
        
    elif method == "librosa":
        # Use librosa's DTW
        D, wp_raw = librosa.sequence.dtw(
            f_chroma_librosa_1,  # (n_chroma, n_frames1)
            f_chroma_librosa_2,  # (n_chroma, n_frames2)
        )
        
        # Convert warping path to synctoolbox format
        wp = wp_raw.T
        
        # Visualize: Accumulated Cost Matrix
        plt.figure(figsize=(15, 15))
        plt.imshow(D, origin='lower', cmap='viridis_r', aspect='auto', interpolation='nearest')
        plt.colorbar(label='Accumulated Cost')
        plt.title('DTW Accumulated Cost Matrix', fontsize=16)
        plt.xlabel('Score Frame', fontsize=14)
        plt.ylabel('Reference Audio Frame', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{vis_filename}_dtw_accumulated_cost.png')
    
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    # Common visualization: chroma distance matrix + warping path
    chroma1_frames = f_chroma_librosa_1.T
    chroma2_frames = f_chroma_librosa_2.T
    
    dist_matrix = cdist(chroma1_frames, chroma2_frames, metric='cosine')
    
    plt.figure(figsize=(15, 15))
    plt.imshow(dist_matrix, origin='lower', cmap='viridis_r', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Cosine Distance')
    
    for i in range(len(wp[0])):
        plt.plot(wp[1][i], wp[0][i], '.', color='yellow', alpha=0.5, markersize=1)
    
    plt.xlabel('Score Frame', fontsize=14)
    plt.ylabel('Reference Audio Frame', fontsize=14)
    plt.title(f'Distance Matrix: {ref_name} vs {score_name} ({method})', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(f'{vis_filename}_chroma_distance.png')
    
    # Visualize warping path
    plt.figure(figsize=(10, 8))
    plt.scatter(wp[1], wp[0], s=0.5, color='blue', alpha=0.7, marker='.')
    plt.xlabel('Score Frame', fontsize=14)
    plt.ylabel('Reference Audio Frame', fontsize=14)
    plt.title(f'Warping Path ({method})', fontsize=16)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{vis_filename}_warping_path.png')
    
    return wp

def generate_reference_annotations(warping_path, score_annots, frame_rate=FRAME_RATE, tempo=120):
    """
    워핑 패스를 사용하여 악보 어노테이션을 참조 오디오 어노테이션으로 변환합니다.
    
    Parameters:
    -----------
    score_part : partitura.score.Part
        악보 파트. score_annots가 None인 경우 어노테이션을 생성하는 데 사용됩니다.
    warping_path : tuple of np.ndarray
        워핑 패스. warping_path[0]은 참조 프레임, warping_path[1]은 악보 프레임.
    score_annots : np.ndarray, optional
        미리 계산된 악보 어노테이션(초 단위). None이면 score_part에서 생성합니다.
    frame_rate : int
        프레임 레이트(Hz)
    tempo : float
        악보 템포(BPM)
        
    Returns:
    --------
    ref_annots : np.ndarray
        참조 오디오 어노테이션(초 단위)
    """
    ref_frames, score_frames = warping_path
    
    # 악보 어노테이션을 프레임으로 변환
    score_annots_frames = np.round(score_annots * frame_rate).astype(int)
    
    # 각 악보 어노테이션 프레임에 대해 참조 오디오 프레임 찾기
    ref_annots_frames = []
    for score_frame in score_annots_frames:
        idx = np.where(score_frames >= score_frame)[0]
        if len(idx) == 0:
            ref_frame = ref_frames[-1]
        else:
            ref_frame = ref_frames[idx[0]]
        ref_annots_frames.append(ref_frame)
    
    # 프레임을 시간(초)으로 변환
    ref_annots = np.array(ref_annots_frames) / frame_rate
    return ref_annots

def add_click_to_audio(audio_path, annotations, output_path=None, sr=SAMPLE_RATE):
    """
    오디오 파일에 어노테이션 시간에 클릭 사운드를 추가합니다.
    
    Parameters:
    -----------
    audio_path : str or Path
        오디오 파일 경로
    annotations : np.ndarray
        어노테이션 시간(초 단위)
    output_path : str or Path, optional
        출력 파일 경로. None이면 자동 생성합니다.
    sr : int
        샘플링 레이트(Hz)
        
    Returns:
    --------
    output_path : str
        저장된 출력 파일 경로
    """
    
    print(f"\n=== 오디오에 클릭 추가 ===")
    print(f"오디오 파일: {audio_path}")
    print(f"어노테이션 포인트 수: {len(annotations)}")
    
    # 오디오 로드
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    
    # 어노테이션 기반 클릭 생성
    click_audio = librosa.clicks(
        times=annotations,
        sr=sr,
        click_freq=1000,  # 클릭 주파수 (Hz)
        length=len(audio)
    )
    
    # 오디오와 클릭 혼합
    audio_with_clicks = audio + click_audio
    
    # 출력 경로 설정
    if output_path is None:
        output_path = f"{Path(audio_path).stem}_with_clicks.wav"
    
    # 혼합된 오디오 저장
    sf.write(
        output_path,
        audio_with_clicks,
        sr,
        subtype="PCM_24"
    )
    print(f"클릭이 추가된 오디오가 {output_path}에 저장됨")
    
    return output_path


def run_offline_alignment(score_audio_path: Path, ref_audio_path: Path):
    # read audio
    audio_1, _ = librosa.load(score_audio_path.as_posix(), sr= SAMPLE_RATE)
    audio_2, _ = librosa.load(ref_audio_path.as_posix(), sr= SAMPLE_RATE)

    # estimate tuning
    # tuning_offset_1 = estimate_tuning(audio_1, SAMPLE_RATE, N=HOP_LENGTH * 2)
    # tuning_offset_2 = estimate_tuning(audio_2, SAMPLE_RATE, N=HOP_LENGTH * 2)
    tuning_offset_1 = 0
    tuning_offset_2 = 0

    # generate chroma features from librosa
    f_chroma_librosa_1 = librosa.feature.chroma_cens(
        y=audio_1,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )
    f_chroma_librosa_2 = librosa.feature.chroma_cens(
        y=audio_2,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
    )

    # generate DLNCO features
    f_DLNCO_1 = _get_DLNCO_features_from_audio(
        audio=audio_1,
        tuning_offset=tuning_offset_1,
        feature_sequence_length=f_chroma_librosa_1.shape[1],
    )

    f_DLNCO_2 = _get_DLNCO_features_from_audio(
        audio=audio_2,
        tuning_offset=tuning_offset_2,
        feature_sequence_length=f_chroma_librosa_2.shape[1],
    )
    wp_chroma_dlnco = sync_via_mrmsdtw(
        f_chroma1=f_chroma_librosa_1,
        f_onset1=f_DLNCO_1,
        f_chroma2=f_chroma_librosa_2,
        f_onset2=f_DLNCO_2,
        input_feature_rate=FRAME_RATE,
        #step_weights=STEP_WEIGHTS, 어디?
        #threshold_rec=THRESHOLD_REC, 어디?
        verbose=False,
    )

    chroma1_frames = f_chroma_librosa_1.T  # (n_frames1, 12)
    chroma2_frames = f_chroma_librosa_2.T  # (n_frames2, 12)
    
    dist_matrix = cdist(chroma1_frames, chroma2_frames, metric='cosine')
    
    
    plt.figure(figsize=(30, 24), dpi=600)
    
    plt.imshow(dist_matrix, origin='lower', cmap='viridis_r', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Cosine Distance')
    
    # 워핑 패스 오버레이 (노란색 점으로 표시)
    for i in range(len(wp_chroma_dlnco[0])):
        #plt.plot(wp_chroma_dlnco[0][i], wp_chroma_dlnco[1][i], '.', color='yellow', alpha=0.5, markersize=1)
        plt.plot(wp_chroma_dlnco[1][i], wp_chroma_dlnco[0][i], '.', color='yellow', alpha=0.5, markersize=1)
    
    plt.xlabel('Reference Audio Frame', fontsize=15)
    plt.ylabel('Score Frame', fontsize=15)
    plt.title('Distance Matrix with Warping Path Overlay (High Resolution)', fontsize=16)
   
    plt.tight_layout()
    plt.savefig('chroma_distance_matrix.png')
    
    # 워핑 패스 전용 시각화
    plt.figure(figsize=(15, 12), dpi=300)
    plt.scatter(wp_chroma_dlnco[1], wp_chroma_dlnco[0], s=0.5, color='blue', alpha=0.7, marker='.')
    plt.xlabel('Reference Audio Frame', fontsize=14)
    plt.ylabel('Score Frame', fontsize=14)
    plt.title('Warping Path', fontsize=16)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('warping_path_only.png', dpi=300)
    
    #DLNCO 특성 기반 거리 행렬 시각화
    dlnco1_frames = f_DLNCO_1.T
    dlnco2_frames = f_DLNCO_2.T
    
    dlnco_dist_matrix = cdist(dlnco1_frames, dlnco2_frames, metric='euclidean')
    
    plt.figure(figsize=(15, 15))
    
    # 자세한 거리 행렬 표시 (히트맵, 보다 세밀한 컬러맵 사용)
    plt.imshow(dlnco_dist_matrix, origin='lower', cmap='plasma_r', aspect='auto', interpolation='nearest')
    plt.colorbar(label='Euclidean Distance')
    
    # 워핑 패스 오버레이 (매우 작은 흰색 점으로 표시)
    for i in range(len(wp_chroma_dlnco[0])):
        plt.plot(wp_chroma_dlnco[1][i], wp_chroma_dlnco[0][i], '.', color='white', alpha=0.7, markersize=1)
    
    plt.xlabel('Reference Audio Frame', fontsize=16)
    plt.ylabel('Score Frame', fontsize=16)
    plt.title('DLNCO Distance Matrix with Warping Path', fontsize=20)
    
    plt.tight_layout()
    plt.savefig('dlnco_distance_matrix.png')
    
    return wp_chroma_dlnco'''


