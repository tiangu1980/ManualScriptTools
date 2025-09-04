https://github.com/great-wind/MicroSoft_VibeVoice

frpc_windows_amd64.exe

AMD GPU: torch -> torch-directml
	python -s ComfyUI\main.py --directml

python demo/gradio_demo.py --model_path microsoft/VibeVoice-1.5B --share

bf -> fp32

Usage 1: Launch Gradio demo
	apt update && apt install ffmpeg -y # for demo

	# For 1.5B model
	python demo/gradio_demo.py --model_path microsoft/VibeVoice-1.5B --share

	# For Large model
	python demo/gradio_demo.py --model_path microsoft/VibeVoice-Large --share

Usage 2: Inference from files directly
	# We provide some LLM generated example scripts under demo/text_examples/ for demo
	# 1 speaker
	python demo/inference_from_file.py --model_path microsoft/VibeVoice-Large --txt_path demo/text_examples/1p_abs.txt --speaker_names Alice

	# or more speakers
	python demo/inference_from_file.py --model_path microsoft/VibeVoice-Large --txt_path demo/text_examples/2p_music.txt --speaker_names Alice Frank
