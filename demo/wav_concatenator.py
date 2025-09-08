import soundfile as sf
import numpy as np
import argparse

def concatenate_wavs(input_files, output_file):
    """
    将多个WAV文件按顺序合并为一个文件
    :param input_files: 输入文件路径列表
    :param output_file: 输出文件路径
    """
    audio_data = []
    sample_rate = None
    
    for file in input_files:
        data, sr = sf.read(file)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"采样率不匹配: {file}的采样率是{sr}, 但期望的是{sample_rate}")
        audio_data.append(data)
    
    combined = np.concatenate(audio_data)
    sf.write(output_file, combined, sample_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='合并多个WAV文件')
    parser.add_argument('-i', '--input', nargs='+', required=True, 
                        help='输入WAV文件列表(按顺序)')
    parser.add_argument('-o', '--output', default='output.wav',
                        help='输出WAV文件名(默认: output.wav)')
    args = parser.parse_args()
    
    try:
        concatenate_wavs(args.input, args.output)
        print(f"成功合并{len(args.input)}个文件到 {args.output}")
    except Exception as e:
        print(f"合并失败: {str(e)}")
