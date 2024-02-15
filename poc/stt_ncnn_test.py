#!/usr/bin/env python3

# Real-time speech recognition from a microphone with sherpa-ncnn Python API
#
# Please refer to
# https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
# to download pre-trained models

import sys
import numpy as np

try:
    import sounddevice as sd
except ImportError as e:
    print("Please install sounddevice first. You can use")
    print()
    print("  pip install sounddevice")
    print()
    print("to install it")
    sys.exit(-1)

import sherpa_ncnn


def create_recognizer():
    # Please replace the model files if needed.
    # See https://k2-fsa.github.io/sherpa/ncnn/pretrained_models/index.html
    # for download links.
    recognizer = sherpa_ncnn.Recognizer(
        tokens="./sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23/tokens.txt",
        encoder_param="./sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23/encoder_jit_trace-pnnx.ncnn.param",
        encoder_bin="./sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23/encoder_jit_trace-pnnx.ncnn.bin",
        decoder_param="./sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23/decoder_jit_trace-pnnx.ncnn.param",
        decoder_bin="./sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23/decoder_jit_trace-pnnx.ncnn.bin",
        joiner_param="./sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23/joiner_jit_trace-pnnx.ncnn.param",
        joiner_bin="./sherpa-ncnn-streaming-zipformer-zh-14M-2023-02-23/joiner_jit_trace-pnnx.ncnn.bin",
        num_threads=4,
        hotwords_file="",
        hotwords_score=1.5,
    )
    return recognizer

def is_silence(samples, threshold=0.01):
    """检测是否静音。"""
    return np.max(np.abs(samples)) < threshold


def main():
    print("Started! Please speak")
    recognizer = create_recognizer()
    sample_rate = recognizer.sample_rate
    samples_per_read = int(0.1 * sample_rate)  # 0.1 second = 100 ms
    last_result = ""
    silence_duration = 0.0  # 静音持续时间
    max_silence_duration = 0.5  # 最大静音持续时间（秒），超过此值认为句子结束

    with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
        while True:
            samples, _ = s.read(samples_per_read)
            samples = samples.reshape(-1)

            # 检测静音
            if is_silence(samples):
                silence_duration += 0.1  # 假设 samples_per_read 为 0.1 秒
                if silence_duration >= max_silence_duration:
                    print("\nSentence End")  # 句子结束
                    silence_duration = 0.0
            else:
                silence_duration = 0.0  # 重置静音持续时间

            recognizer.accept_waveform(sample_rate, samples)
            result = recognizer.text
            if last_result != result:
                last_result = result
                print("\r{}".format(result), end="", flush=True)


if __name__ == "__main__":
    devices = sd.query_devices()
    print(devices)
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')

    try:
        main()
    except KeyboardInterrupt:
        print("\nCaught Ctrl + C. Exiting")