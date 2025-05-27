import os
from pathlib import Path
from typing import Optional, Union

import librosa
import numpy as np
import partitura
import soundfile as sf
from partitura.io.exportmidi import get_ppq
from partitura.score import Part

from matchmaker.dp import OnlineTimeWarpingArzt, OnlineTimeWarpingDixon
from matchmaker.features.audio import (
    FRAME_RATE,
    SAMPLE_RATE,
    ChromagramProcessor,
    MelSpectrogramProcessor,
    MFCCProcessor,
    ViolinProcessor,
)
from matchmaker.features.midi import PianoRollProcessor, PitchIOIProcessor
from matchmaker.io.audio import AudioStream
from matchmaker.io.midi import MidiStream
from matchmaker.prob.hmm import PitchIOIHMM
from matchmaker.utils.eval import TOLERANCES, transfer_positions, get_evaluation_results
from matchmaker.utils.misc import (
    adjust_tempo_for_performance_audio,
    generate_score_audio,
    is_audio_file,
    is_midi_file,
    _get_DLNCO_features_from_audio,
    run_offline_alignment,
)

PathLike = Union[str, bytes, os.PathLike]
DEFAULT_TEMPO = 120
DEFAULT_DISTANCE_FUNCS = {
    "arzt": OnlineTimeWarpingArzt.DEFAULT_DISTANCE_FUNC,
    "dixon": OnlineTimeWarpingDixon.DEFAULT_DISTANCE_FUNC,
    "hmm": None,
}

DEFAULT_METHODS = {
    "audio": "arzt",
    "midi": "hmm",
}

AVAILABLE_METHODS = ["arzt", "dixon", "hmm"]


class Matchmaker(object):
    """
    A class to perform online score following with I/O support for audio and MIDI

    Parameters
    ----------
    score_file : Union[str, bytes, os.PathLike]
        Path to the score file
    performance_file : Union[str, bytes, os.PathLike, None]
        Path to the performance file. If None, live input is used.
    wait : bool (default: True)
        only for offline option. For debugging or fast testing, set to False
    input_type : str
        Type of input to use: audio or midi
    feature_type : str
        Type of feature to use
    method : str
        Score following method to use
    device_name_or_index : Union[str, int]
        Name or index of the audio device to be used.
        Ignored if `file_path` is given.

    """

    def __init__(
        self,
        score_file: PathLike,
        performance_file: Union[PathLike, None] = None,
        reference_audio_file: Union[PathLike, None] = None, #추가함
        dataset = None,
        wait: bool = True,  # only for offline option. For debugging or fast testing, set to False
        input_type: str = "audio",  # 'audio' or 'midi'
        feature_type: str = None,
        method: str = None,
        distance_func: Optional[str] = None,
        device_name_or_index: Union[str, int] = None,
        sample_rate: int = SAMPLE_RATE,
        frame_rate: int = FRAME_RATE,
    ):
        self.score_file = score_file
        self.dataset = dataset
        self.performance_file = performance_file
        self.reference_audio_file = reference_audio_file #추가함
        self.input_type = input_type
        self.feature_type = feature_type
        self.frame_rate = frame_rate
        self.score_part: Optional[Part] = None
        self.distance_func = distance_func
        self.device_name_or_index = device_name_or_index
        self.processor = None
        self.stream = None
        self.score_follower = None
        self.reference_features = None
        self.tempo = DEFAULT_TEMPO  # bpm for quarter note
        self._has_run = False

        self.score_reference_wp = None #워핑경로 저장 변수

        # setup score file
        if score_file is None:
            raise ValueError("Score file is required")

        try:
            self.score_part = partitura.load_score_as_part(self.score_file)

        except Exception as e:
            raise ValueError(f"Invalid score file: {e}")

        # setup feature processor
        if self.feature_type is None:
            self.feature_type = "chroma" if input_type == "audio" else "pitchclass"

        if self.feature_type == "chroma":
            self.processor = ChromagramProcessor(
                sample_rate=sample_rate,
            )
        elif self.feature_type == "mfcc":
            self.processor = MFCCProcessor(
                sample_rate=sample_rate,
            )
        elif self.feature_type == "mel":
            self.processor = MelSpectrogramProcessor(
                sample_rate=sample_rate,
            )
        elif self.feature_type == "violin":  # 여기에 violin 처리 추가
            self.processor = ViolinProcessor(
                sample_rate=sample_rate,
            )
        elif self.feature_type == "pitchclass":
            self.processor = PitchIOIProcessor(piano_range=True)
        elif self.feature_type == "pianoroll":
            self.processor = PianoRollProcessor(piano_range=True)
        else:
            raise ValueError("Invalid feature type")

        # validate performance file and input_type
        if self.performance_file is not None:
            # check performance file type matches input type
            if self.input_type == "audio" and not is_audio_file(self.performance_file):
                raise ValueError(
                    f"Invalid performance file. Expected audio file, but got {self.performance_file}"
                )
            elif self.input_type == "midi" and not is_midi_file(self.performance_file):
                raise ValueError(
                    f"Invalid performance file. Expected MIDI file, but got {self.performance_file}"
                )

        # setup stream device
        if self.input_type == "audio":
            self.stream = AudioStream(
                processor=self.processor,
                device_name_or_index=self.device_name_or_index,
                file_path=self.performance_file,
                wait=wait,
                target_sr=SAMPLE_RATE,
            )
        elif self.input_type == "midi":
            self.stream = MidiStream(
                processor=self.processor,
                port=self.device_name_or_index,
                file_path=self.performance_file,
            )
        else:
            raise ValueError("Invalid input type")

        # preprocess score (setting reference features, tempo)
        self.preprocess_score()

        # validate method first
        if method is None:
            method = DEFAULT_METHODS[self.input_type]
        elif method not in AVAILABLE_METHODS:
            raise ValueError(f"Invalid method. Available methods: {AVAILABLE_METHODS}")

        # setup distance function
        if distance_func is None:
            distance_func = DEFAULT_DISTANCE_FUNCS[method]

        # setup score follower
        if method == "arzt":
            self.score_follower = OnlineTimeWarpingArzt(
                reference_features=self.reference_features,
                queue=self.stream.queue,
                distance_func=distance_func,
                frame_rate=self.frame_rate,
            )
        elif method == "dixon":
            self.score_follower = OnlineTimeWarpingDixon(
                reference_features=self.reference_features,
                queue=self.stream.queue,
                distance_func=distance_func,
                frame_rate=self.frame_rate,
            )
        elif method == "hmm":
            self.score_follower = PitchIOIHMM(
                reference_features=self.reference_features,
                queue=self.stream.queue,
            )

    def preprocess_score(self):
        print("\n=== Preprocessing Score ===")
        
        if self.input_type == "audio":
            # 1. 악보 오디오 생성
            score_audio = generate_score_audio(self.score_part, self.tempo, SAMPLE_RATE)
            score_file_stem = Path(self.score_file).stem
            print(f"Generated score audio (length: {len(score_audio)/SAMPLE_RATE:.2f}s)")
            
            # 2. 악보 어노테이션 생성
            score_annots = self.build_score_annotations()
            print(f"Generated {len(score_annots)} score annotations")
            
            # 3. 악보 오디오 임시 저장 (클릭 없는 버전 - 워핑 경로 계산용)
            score_audio_path = Path(f"temp_score_audio_{score_file_stem}.wav")
            sf.write(
                str(score_audio_path), 
                score_audio,  # 클릭 없는 원본 오디오
                SAMPLE_RATE, 
                subtype="PCM_24"
            )
            print(f"Saved original score audio to {score_audio_path}")
            
            # 4. 악보 오디오에 클릭 추가 및 저장 (디버깅용)
            score_annots_audio = librosa.clicks(
                times=score_annots,
                sr=SAMPLE_RATE,
                click_freq=1000,
                length=len(score_audio),
            )
            score_audio_with_clicks = score_audio + score_annots_audio
            score_audio_with_clicks_path = f"score_audio_with_clicks_{score_file_stem}.wav"
            sf.write(
                score_audio_with_clicks_path,
                score_audio_with_clicks,
                SAMPLE_RATE,
                subtype="PCM_24",
            )
            print(f"Saved score audio with clicks to {score_audio_with_clicks_path}")
            
            # 5. 참조 오디오가 있는 경우 처리
            if self.reference_audio_file is not None:
                print(f"Reference audio file detected: {self.reference_audio_file}")
                reference_audio_path = Path(self.reference_audio_file)
                
                # 5.1 참조 오디오 로드
                reference_audio_path = Path(self.reference_audio_file)
                reference_audio, _ = librosa.load(self.reference_audio_file, sr=SAMPLE_RATE, mono=True)
                print(f"Loaded reference audio (length: {len(reference_audio)/SAMPLE_RATE:.2f}s)")
                
                # 5.2 참조 어노테이션 파일 로드
                ref_annots = []  # 기본값으로 빈 리스트 초기화
                
                # dataset이 None이 아니고 딕셔너리인 경우에만 get 메서드 호출
                if self.dataset is not None and isinstance(self.dataset, dict):
                    ref_annots_file = self.dataset.get("ref_annotations", None)
                    if ref_annots_file:
                        print(f"Loading reference annotations from {ref_annots_file}")
                        try:
                            ref_annots = np.loadtxt(ref_annots_file, delimiter="\t", usecols=0)
                            print(f"Loaded {len(ref_annots)} reference annotations from file.")
                        except Exception as e:
                            print(f"Error loading reference annotations: {e}")
                            ref_annots = []  # 문제가 생기면 빈 리스트로 초기화
                else:
                    print("No dataset provided or invalid dataset format - skipping reference annotations loading")

                # 5.3 참조 어노테이션이 존재하는 경우 클릭 추가
                if len(ref_annots) > 0:
                    print(f"Adding clicks to reference audio at {len(ref_annots)} annotation points")

                    ref_annots_array = np.array(ref_annots)  # Numpy 배열로 명시적 변환
                    ref_annots_audio = librosa.clicks(
                        times=ref_annots_array,
                        sr=SAMPLE_RATE,
                        click_freq=1000,  # 중복 제거
                        click_duration=0.1,  # 클릭의 지속 시간을 더 길게 설정
                        length=len(reference_audio)
                    )

                    # 참조 오디오와 클릭 혼합
                    reference_audio_with_clicks = reference_audio + ref_annots_audio
                    
                    # 클릭이 포함된 참조 오디오 저장
                    ref_file_stem = Path(self.reference_audio_file).stem
                    reference_audio_with_clicks_path = f"reference_audio_with_clicks_{ref_file_stem}.wav"
                    sf.write(
                        reference_audio_with_clicks_path,
                        reference_audio_with_clicks,
                        SAMPLE_RATE,
                        subtype="PCM_24",
                    )
                    print(f"클릭이 추가된 참조 오디오가 {reference_audio_with_clicks_path}에 저장됨")

                # self.reference_features에 참조 오디오의 특징을 설정
                self.reference_features = self.processor(reference_audio.astype(np.float32))

                # 5.4 오프라인 정렬 (워핑 경로 계산)
                self.score_reference_wp = run_offline_alignment(
                    score_audio_path, 
                    Path(self.reference_audio_file)
                )
                print(f"Calculated warping path between score and reference audio")
                    
            else:
                print("No reference audio file - using score audio features as reference")
                self.reference_features = self.processor(score_audio.astype(np.float32))

            # Ensure reference_features is not None before proceeding
            if self.reference_features is None:
                print("Error: reference_features is None, unable to proceed.")
                return

        else:
            print("Using MIDI input type")
            self.reference_features = self.score_part.note_array()
        
        print("Preprocessing complete\n")
    '''def preprocess_score(self):
        print("\n=== Preprocessing Score ===")
        
        if self.input_type == "audio":
            # 1. 악보 오디오 생성
            score_audio = generate_score_audio(self.score_part, self.tempo, SAMPLE_RATE)
            score_file_stem = Path(self.score_file).stem
            print(f"Generated score audio (length: {len(score_audio)/SAMPLE_RATE:.2f}s)")
            
            # 2. 악보 어노테이션 생성
            score_annots = self.build_score_annotations()
            print(f"Generated {len(score_annots)} score annotations")
            
            # 3. 악보 오디오 임시 저장 (클릭 없는 버전 - 워핑 경로 계산용)
            score_audio_path = Path(f"temp_score_audio_{score_file_stem}.wav")
            sf.write(
                str(score_audio_path), 
                score_audio,  # 클릭 없는 원본 오디오
                SAMPLE_RATE, 
                subtype="PCM_24"
            )
            print(f"Saved original score audio to {score_audio_path}")
            
            # 4. 악보 오디오에 클릭 추가 및 저장 (디버깅용)
            score_annots_audio = librosa.clicks(
                times=score_annots,
                sr=SAMPLE_RATE,
                click_freq=1000,
                length=len(score_audio),
            )
            score_audio_with_clicks = score_audio + score_annots_audio
            score_audio_with_clicks_path = f"score_audio_with_clicks_{score_file_stem}.wav"
            sf.write(
                score_audio_with_clicks_path,
                score_audio_with_clicks,
                SAMPLE_RATE,
                subtype="PCM_24",
            )
            print(f"Saved score audio with clicks to {score_audio_with_clicks_path}")
            
            # 5. 참조 오디오가 있는 경우 처리
            if self.reference_audio_file is not None:
                print(f"Reference audio file detected: {self.reference_audio_file}")
                reference_audio_path = Path(self.reference_audio_file)
                
                # 5.1 참조 오디오 로드
                reference_audio_path = Path(self.reference_audio_file)
                reference_audio, _ = librosa.load(self.reference_audio_file, sr=SAMPLE_RATE, mono=True)
                print(f"Loaded reference audio (length: {len(reference_audio)/SAMPLE_RATE:.2f}s)")
                
                # 5.2 참조 어노테이션 파일 로드
                ref_annots_file = self.dataset.get("ref_annotations", None)  # dataset에서 ref_annotations 경로 가져오기
                if ref_annots_file:
                    print(f"Loading reference annotations from {ref_annots_file}")
                    try:
                        ref_annots = np.loadtxt(ref_annots_file, delimiter="\t", usecols=0)  # 첫 번째 열만 읽기
                        print(f"Loaded {len(ref_annots)} reference annotations from file.")
                    except Exception as e:
                        print(f"Error loading reference annotations: {e}")
                        ref_annots = []  # 문제가 생기면 빈 리스트로 초기화

                # 5.3 참조 어노테이션이 존재하는 경우 클릭 추가
                if ref_annots is not None and len(ref_annots) > 0:
                    print(f"Adding clicks to reference audio at {len(ref_annots)} annotation points")

                    ref_annots_array = np.array(ref_annots)  # Numpy 배열로 명시적 변환
                    ref_annots_audio = librosa.clicks(
                        times=ref_annots_array,
                        sr=SAMPLE_RATE,
                        click_freq=1000,  # 중복 제거
                        click_duration=0.1,  # 클릭의 지속 시간을 더 길게 설정
                        length=len(reference_audio)
                    )

                    # 참조 오디오와 클릭 혼합
                    reference_audio_with_clicks = reference_audio + ref_annots_audio
                    
                    # 클릭이 포함된 참조 오디오 저장
                    ref_file_stem = Path(self.reference_audio_file).stem
                    reference_audio_with_clicks_path = f"reference_audio_with_clicks_{ref_file_stem}.wav"
                    sf.write(
                        reference_audio_with_clicks_path,
                        reference_audio_with_clicks,
                        SAMPLE_RATE,
                        subtype="PCM_24",
                    )
                    print(f"클릭이 추가된 참조 오디오가 {reference_audio_with_clicks_path}에 저장됨")

                    # self.reference_features에 참조 오디오의 특징을 설정
                    self.reference_features = self.processor(reference_audio.astype(np.float32))

                    # 5.4 오프라인 정렬 (워핑 경로 계산)
                    self.score_reference_wp = run_offline_alignment(
                        score_audio_path, 
                        Path(self.reference_audio_file)
                    )
                    print(f"Calculated warping path between score and reference audio")
                    
            else:
                print("No reference audio file - using score audio features as reference")
                self.reference_features = self.processor(score_audio.astype(np.float32))

            # Ensure reference_features is not None before proceeding
            if self.reference_features is None:
                print("Error: reference_features is None, unable to proceed.")
                return

        else:
            print("Using MIDI input type")
            self.reference_features = self.score_part.note_array()
        
        print("Preprocessing complete\n")'''
                
    '''# 5.2 악보 오디오와 참조 오디오 간의 워핑 경로 계산
                self.score_reference_wp = run_offline_alignment(
                    score_audio_path, 
                    Path(self.reference_audio_file)
                )
                print(f"Calculated warping path between score and reference audio")
                
                # 5.3 참조 오디오의 특성을 reference_features로 설정
                self.reference_features = self.processor(reference_audio.astype(np.float32))
                print(f"Extracted features from reference audio")
                
                # 5.4 참조 어노테이션 생성
                ref_annots = self.generate_reference_annotations()
                
                # 5.5 참조 어노테이션이 성공적으로 생성된 경우 클릭 추가
                if ref_annots is not None and len(ref_annots) > 0:
                    print(f"Adding clicks to reference audio at {len(ref_annots)} annotation points")

                    ref_annots_array = np.array(ref_annots)  # Numpy 배열로 명시적 변환
    
                    print(f"Type of ref_annots: {type(ref_annots_array)}")
                    print(f"Shape of ref_annots: {ref_annots_array.shape}")
                    print(f"First 5 annotation times: {ref_annots_array[:5]}")
        
                    try:
                        # 명시적으로 numpy 배열로 변환
                        ref_annots_array = np.array(ref_annots, dtype=np.float32)
                        
                        print(f"클릭 생성 전 확인:")
                        print(f"  어노테이션 배열 타입: {type(ref_annots_array)}, 형태: {ref_annots_array.shape}")
                        print(f"  최초 5개 어노테이션 시간: {ref_annots_array[:5]}")
                        print(f"  샘플링 레이트: {SAMPLE_RATE}")
                        
                        # 수정된 librosa.clicks() 호출
                        ref_annots_audio = librosa.clicks(
                            times=ref_annots_array,
                            sr=SAMPLE_RATE,
                            click_freq=1000,  # 중복 제거
                            click_duration=0.1,  # 더 긴 지속 시간으로 클릭 소리 강화
                            length=len(reference_audio)
                        )
                        
                        # 클릭 위치 검증
                        click_positions = []
                        window_size = int(SAMPLE_RATE * 0.05)  # 50ms 윈도우 크기
                        
                        for t in ref_annots_array:
                            start_idx = max(0, int(t * SAMPLE_RATE) - window_size // 2)
                            end_idx = min(len(ref_annots_audio), start_idx + window_size)
                            if start_idx < len(ref_annots_audio) and end_idx <= len(ref_annots_audio):
                                if np.max(np.abs(ref_annots_audio[start_idx:end_idx])) > 0.01:
                                    click_positions.append(t)
                        
                        print(f"생성된 클릭 검증:")
                        print(f"  발견된 클릭 수: {len(click_positions)}")
                        print(f"  최초 5개 클릭 위치: {click_positions[:5]}")
                        
                        # 참조 오디오와 클릭 혼합
                        reference_audio_with_clicks = reference_audio + ref_annots_audio
                        
                        # 클릭이 포함된 참조 오디오 저장
                        ref_file_stem = Path(self.reference_audio_file).stem
                        reference_audio_with_clicks_path = f"reference_audio_with_clicks_{ref_file_stem}.wav"
                        sf.write(
                            reference_audio_with_clicks_path,
                            reference_audio_with_clicks,
                            SAMPLE_RATE,
                            subtype="PCM_24",
                        )
                        print(f"클릭이 추가된 참조 오디오가 {reference_audio_with_clicks_path}에 저장됨")
                    except Exception as e:
                        print(f"librosa.clicks() 호출 중 오류 발생: {str(e)}")
                        import traceback
                        traceback.print_exc()
            else:
                # 참조 오디오가 없는 경우 악보 오디오의 특성을 reference_features로 사용
                print("No reference audio file - using score audio features as reference")
                self.reference_features = self.processor(score_audio.astype(np.float32))
        else:
            # MIDI 입력 타입의 경우
            print("Using MIDI input type")
            self.reference_features = self.score_part.note_array()
        
        print("Preprocessing complete\n")'''

    def _convert_frame_to_beat(self, current_frame: int) -> float:
        """
        프레임 번호를 악보의 상대적인 비트 위치로 변환
        
        Parameters
        ----------
        current_frame : int
            현재 프레임 번호 (참조 오디오 프레임)
        """
        if self.reference_audio_file is not None and self.score_reference_wp is not None:
            # transfer_positions 함수를 사용하여 단일 프레임 변환
            x, y = self.score_reference_wp[0], self.score_reference_wp[1]
            
            ref_frame = np.round(current_frame)
            
            idx = np.where(x >= ref_frame)[0]
            if len(idx) == 0:
                score_frame = y[-1]
            else:
                score_frame = y[idx[0]]
        else:
            score_frame = current_frame
        
        # 악보 프레임을 비트로 변환
        tick = get_ppq(self.score_part)
        timeline_time = (score_frame / self.frame_rate) * tick * (self.tempo / 60)
        beat_position = np.round(
            self.score_part.beat_map(timeline_time),
            decimals=2,
        )
        return beat_position

    def run(self, verbose: bool = True, wait: bool = True):
        """
        Run the score following process

        Yields
        ------
        float
            Beat position in the score (interpolated)

        Returns
        -------
        list
            Alignment results with warping path
        """
        with self.stream:
            for current_frame in self.score_follower.run(verbose=verbose):
                if self.input_type == "audio":
                    position_in_beat = self._convert_frame_to_beat(current_frame)
                    yield position_in_beat
                else:
                    yield float(self.score_follower.state_space[current_frame])

        self._has_run = True
        return self.score_follower.warping_path

    def build_score_annotations(self, level="beat"):
        score_annots = []
        if level == "beat":  # TODO: add bar-level, note-level
            note_array = np.unique(self.score_part.note_array()["onset_beat"])
            start_beat = np.ceil(note_array.min())
            end_beat = np.floor(note_array.max())
            beats = np.arange(start_beat, end_beat + 1)

            beat_timestamp = [
                self.score_part.inv_beat_map(beat)
                / get_ppq(self.score_part)
                * (60 / self.tempo)
                for beat in beats
            ]

            score_annots = np.array(beat_timestamp)
        return score_annots

    def generate_reference_annotations(self, ref_annots_file=None):
        """
        제공된 참조 어노테이션 파일을 사용할 수 있으면 사용하고,
        그렇지 않으면 기존 방식으로 생성합니다.
        """
        if ref_annots_file:
            # 파일로부터 참조 어노테이션을 로드
            print(f"Loading reference annotations from {ref_annots_file}")
            ref_annots = np.loadtxt(ref_annots_file, delimiter="\t")
            print(f"Loaded {len(ref_annots)} reference annotations from file.")
            return ref_annots

        # 파일이 제공되지 않으면 기존 방식으로 어노테이션 생성
        score_annots = self.build_score_annotations()
        print("\n=== Generating Reference Annotations ===")
        print(f"Generated {len(score_annots)} score annotations")

        try:
            # 워핑 경로를 이용해 참조 어노테이션을 생성
            ref_annots = transfer_positions(
                wp=self.score_reference_wp,
                ref_anns=score_annots,
                frame_rate=self.frame_rate,
            )
            if len(ref_annots) == 0:
                print("No reference annotations were generated!")
                return None

            # 생성된 참조 어노테이션 파일로 저장
            ref_annotations_path = Path(f"reference_annotations_{Path(self.reference_audio_file).stem}.tsv")
            np.savetxt(ref_annotations_path, ref_annots, delimiter='\t')
            print(f"Reference annotations saved to {ref_annotations_path}")
            return ref_annots
        except Exception as e:
            print(f"Error generating reference annotations: {str(e)}")
            return None

    def run_evaluation(
        self,
        perf_annotations: PathLike,
        ref_annotations: Optional[PathLike] = None,
        level: str = "beat",
        tolerance: list = TOLERANCES,
    ) -> dict:
        """
        평가를 실행합니다.
        
        Parameters:
        -----------
        perf_annotations : Union[str, bytes, os.PathLike, list, ndarray]
            실행 어노테이션 파일 경로 또는 어노테이션 리스트
        ref_annotations : Optional[Union[str, bytes, os.PathLike, list, ndarray]]
            참조 어노테이션 파일 경로 또는 어노테이션 리스트
        level : str
            평가 레벨 ('beat', 'bar' 등)
        tolerance : list
            허용 오차 임계값 목록 (밀리초)
            
        Returns:
        --------
        dict
            평가 결과를 포함하는 사전
        """
        if not self._has_run:
            raise ValueError("평가 전에 run() 메서드를 호출해야 합니다")

        # 악보 어노테이션 로드
        score_annots = self.build_score_annotations(level)
        
        # 실행 어노테이션 로드 - 타입에 따라 처리
        if isinstance(perf_annotations, (str, bytes, os.PathLike)):
            try:
                # 파일 경로인 경우 파일에서 로드
                perf_annots = np.loadtxt(fname=perf_annotations, delimiter="\t", usecols=0)
                print(f"Loaded {len(perf_annots)} performance annotations from file: {perf_annotations}")
            except Exception as e:
                print(f"Error loading performance annotations from file: {e}")
                perf_annots = np.array([])
        elif isinstance(perf_annotations, (list, np.ndarray)):
            # 이미 리스트나 배열인 경우 그대로 사용
            perf_annots = np.array(perf_annotations)
            print(f"Using provided performance annotations array with {len(perf_annots)} items")
        else:
            print(f"Invalid performance annotations type: {type(perf_annotations)}")
            perf_annots = np.array([])
        
        # 빈 어노테이션 처리
        if len(perf_annots) == 0:
            print("No performance annotations available for evaluation")
            return {
                "error": "No performance annotations available",
                "mean": float('nan'),
                "median": float('nan'),
                "count": 0
            }

        # 참조 어노테이션 처리 - 타입에 따라 처리
        if isinstance(ref_annotations, (str, bytes, os.PathLike)):
            try:
                # 파일 경로인 경우 파일에서 로드
                ref_annots = np.loadtxt(fname=ref_annotations, delimiter="\t", usecols=0)
                print(f"Loaded {len(ref_annots)} reference annotations from file: {ref_annotations}")
            except Exception as e:
                print(f"Error loading reference annotations from file: {e}")
                ref_annots = None
        elif isinstance(ref_annotations, (list, np.ndarray)):
            # 이미 리스트나 배열인 경우 그대로 사용
            ref_annots = np.array(ref_annotations)
            print(f"Using provided reference annotations array with {len(ref_annots)} items")
        else:
            print(f"No reference annotations provided or invalid type: {type(ref_annotations)}")
            ref_annots = None
        
        # ref_annotations가 None이고 self.reference_audio_file이 있으면 generate_reference_annotations 호출
        if ref_annots is None and self.reference_audio_file is not None:
            print("Generating reference annotations from warping path...")
            ref_annots = self.generate_reference_annotations()
        
        # 평가 결과 계산
        return get_evaluation_results(
            ref_annots, 
            perf_annots, 
            len(perf_annots),
            warping_path=self.score_follower.warping_path,
            frame_rate=self.frame_rate,
            tolerances=tolerance
        )

    '''def preprocess_score(self):
        if self.input_type == "audio":
            # if self.performance_file is not None:
            #     # tempo is slightly adjusted to reflect the tempo of the performance audio
            #     self.tempo = adjust_tempo_for_performance_audio(
            #         self.score_part, self.performance_file
            #     )

            # generate score audio
            score_audio = generate_score_audio(self.score_part, self.tempo, SAMPLE_RATE)

            # save score audio (for debugging)
            score_annots = self.build_score_annotations()
            score_annots_audio = librosa.clicks(
                times=score_annots,
                sr=SAMPLE_RATE,
                click_freq=1000,
                length=len(score_audio),
            )
            score_audio_mixed = score_audio + score_annots_audio
            sf.write(
                f"score_audio_{Path(self.score_file).stem}.wav",
                score_audio_mixed,
                SAMPLE_RATE,
                subtype="PCM_24",
            )

            reference_features = self.processor(score_audio.astype(np.float32))
            self.reference_features = reference_features
        else:
            self.reference_features = self.score_part.note_array()

    def _convert_frame_to_beat(self, current_frame: int) -> float:
        """
        Convert frame number to relative beat position in the score.

        Parameters
        ----------
        frame_rate : int
            Frame rate of the audio stream
        current_frame : int
            Current frame number
        """
        tick = get_ppq(self.score_part)
        timeline_time = (current_frame / self.frame_rate) * tick * (self.tempo / 60)
        beat_position = np.round(
            self.score_part.beat_map(timeline_time),
            decimals=2,
        )
        return beat_position
    def preprocess_score(self):
        if self.input_type == "audio":
            if self.reference_audio_file is not None:
                score_audio = generate_score_audio(self.score_part, self.tempo, SAMPLE_RATE)
                
                # 악보 오디오 임시 저장 (워핑 경로 계산을 위해)
                score_audio_path = Path(f"temp_score_audio_{Path(self.score_file).stem}.wav")
                sf.write(str(score_audio_path), score_audio, SAMPLE_RATE, subtype="PCM_24")
                
                # score 오디오와 reference 오디오 사이의 워핑 경로 계산
                self.score_reference_wp = run_offline_alignment(
                    score_audio_path, 
                    Path(self.reference_audio_file)
                )
                
                # reference 오디오를 처리하여 feature 추출
                reference_audio, _ = librosa.load(self.reference_audio_file, sr=SAMPLE_RATE, mono=True)
                
                # reference 오디오의 특징을 reference_features에 저장
                self.reference_features = self.processor(reference_audio.astype(np.float32))

                ref_annots = self.generate_reference_annotations()
                
                # 7. 참조 어노테이션 기반 클릭 오디오 생성
                if ref_annots is not None and len(ref_annots) > 0:
                    ref_annots_audio = librosa.clicks(
                        times=ref_annots,
                        sr=SAMPLE_RATE,
                        click_freq=1000,
                        length=len(reference_audio),
                    )
                    ref_file_stem = Path(self.reference_audio_file).stem
                    reference_audio_with_clicks_path = f"reference_audio_with_clicks_{ref_file_stem}.wav"
                    reference_audio_mixed = reference_audio + ref_annots_audio
                    sf.write(
                        #f"reference_audio_with_annotations_{Path(self.reference_audio_file).stem}.wav",
                        reference_audio_with_clicks_path,
                        reference_audio_mixed,
                        SAMPLE_RATE,
                        subtype="PCM_24",
                    )
                    print(f"Reference audio with annotations saved. {len(ref_annots)} annotation points generated.")
            else:
                # 기존 코드
                score_audio = generate_score_audio(self.score_part, self.tempo, SAMPLE_RATE)
                
                # 기존 디버깅용 오디오 저장
                score_annots = self.build_score_annotations()
                score_annots_audio = librosa.clicks(
                    times=score_annots,
                    sr=SAMPLE_RATE,
                    click_freq=1000,
                    length=len(score_audio),
                )
                score_audio_mixed = score_audio + score_annots_audio
                score_audio_with_clicks_path = f"score_audio_with_clicks_{score_file_stem}.wav"
                sf.write(
                    #f"score_audio_{Path(self.score_file).stem}.wav",
                    score_audio_with_clicks_path,
                    score_audio_mixed,
                    SAMPLE_RATE,
                    subtype="PCM_24",
                )
            
            self.reference_features = self.processor(reference_audio.astype(np.float32))
        else:
            self.reference_features = self.score_part.note_array()''' 
    