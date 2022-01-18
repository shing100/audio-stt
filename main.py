import numpy as np
import os
from tqdm.notebook import tqdm
import librosa
import matplotlib.pyplot as plt
import glob
import soundfile as sf
from pathlib import Path
from split_wav_audio import SplitWavAudio

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="C:\google-cloud-sdk\long-nation-332511-7faa6695d9fb.json"

def load_audio(file_path, sr=22050):
    """
       file_path : 파일위치
       sr : 오디오를 읽을 때 Sampling rate 지정
    """
    ## 확장자 추출
    ext = Path(file_path).suffix

    ## 파일 읽기
    if ext in ['.wav', '.flac']:
        wav, sr = librosa.load(file_path, sr=sr)
    elif ext == '.pcm':
        wav = np.memmap(file_path, dtype='h', mode='r').astype('float32') / 32767
    elif ext in ['.raw', '.RAW']:
        wav, sr = sf.read(file_path, channels=1, samlerate=sr, format='RAW', subtype='PCM_16')
    else:
        raise ValueError("Unsupported preprocess method : {0}".format(ext))

    return wav, sr


def trim_audio_data(audio_file, save_file):
    sr = 96000
    sec = 30

    y, sr = librosa.load(audio_file, sr=sr)

    ny = y[:sr*sec]

    sf.write(save_file + '.wav', ny, sr)

## 공백 자르기(패딩 추가)
def trim_audio(wav, top_db=10, pad_len=4000):
    """

    """
    ## 최대 db에 따라 음성의 자를 위치 판별
    non_silence_indices = librosa.effects.split(wav, top_db=top_db)
    start = non_silence_indices[0][0]
    end = non_silence_indices[-1][1]

    ## 음성 자르기
    wav = wav[start:end]

    ## padding 추가
    wav = np.hstack([np.zeros(pad_len), wav, np.zeros(pad_len)])

    return wav


## WAV 그려보기
def plot_wav(wav, sr):
    ## 그려보기
    plt.figure(1)

    plot_a = plt.subplot(211)
    plot_a.plot(wav)
    plot_a.set_xlabel('sample rate * time')
    plot_a.set_ylabel('energy')

    plot_b = plt.subplot(212)
    plot_b.specgram(wav, NFFT=1024, Fs=sr, noverlap=900)
    plot_b.set_xlabel('Time')
    plot_b.set_ylabel('Frequency')

    plt.show()


def transcribe_file(speech_file):
    """Transcribe the given audio file."""
    from google.cloud import speech
    import io

    client = speech.SpeechClient()

    with io.open(speech_file, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=22050,
        language_code="ko",
    )

    response = client.recognize(config=config, audio=audio)

    # Each result is for a consecutive portion of the audio. Iterate through
    # them to get the transcripts for the entire audio file.
    for result in response.results:
        # The first alternative is the most likely one for this portion.
        print(speech_file + "|{}".format(result.alternatives[0].transcript))


if __name__ == '__main__':
    ## 시작하기
    ## 타코트론2는 기본적으로 22050 sampling rate에서 동작
    sampling_rate = 22050
    
    ## 개인설정에 따라 특정 소리보다 작은 음성을 삭제하도록 설정
    decibel = 20

    ## Wav 파일 읽어오기  pcm 또는 다른 확장자도 사용 가능.
    root_path = 'C:/Users/shing/PycharmProjects/voiceCreator/datasets/in/audio/'

    ## 변경할 파일 이름
    origin_file_name = "four.wav"


    split = SplitWavAudio(root_path, origin_file_name)
    split.multiple_split(1)

    file_list = glob.glob(os.path.join(root_path, "*.wav"))
    # file_list = glob.glob(os.path.join(root_path, "*.pcm"))

    ## 저장할 위치 선택
    save_path = 'datasets/in/save'
    os.makedirs(save_path, exist_ok=True)

    for file_path in tqdm(file_list):
        ## 파일 불러오기(타코트론2는 기본적으로 22050 sampling rate에서 동작)
        wav, sr = load_audio(file_path, sr=sampling_rate)

        ## 오디오 자르기(패딩 추가)
        trimed_wav = trim_audio(wav, top_db=decibel)

        filename = Path(file_path).name
        temp_save_path = os.path.join(save_path, filename)

        ## 저장하기
        sf.write(temp_save_path, trimed_wav, sampling_rate)

    # 저장된 파일 STT 하여 데이터셋 만들기
    save_path = 'C:/Users/shing/PycharmProjects/voiceCreator/datasets/in/save'
    file_list = glob.glob(os.path.join(save_path, "*.wav"))
    for file_path in tqdm(file_list):
        if Path(file_path).name != origin_file_name:
             transcribe_file(file_path)
