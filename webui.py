# coding=utf-8

import os
import librosa
import base64
import io
import gradio as gr
import re
import locale

import numpy as np
import torch
import torchaudio


from funasr import AutoModel

# Auto detect device: CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU
device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

model = "iic/SenseVoiceSmall"
model = AutoModel(model=model,
		  vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
		  vad_kwargs={"max_single_segment_time": 30000},
		  trust_remote_code=True,
		  device=device
		  )

import re

emo_dict = {
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
}

event_dict = {
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|Cry|>": "😭",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "🤧",
}

emoji_dict = {
	"<|nospeech|><|Event_UNK|>": "❓",
	"<|zh|>": "",
	"<|en|>": "",
	"<|yue|>": "",
	"<|ja|>": "",
	"<|ko|>": "",
	"<|nospeech|>": "",
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
	"<|Cry|>": "😭",
	"<|EMO_UNKNOWN|>": "",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "😷",
	"<|Sing|>": "",
	"<|Speech_Noise|>": "",
	"<|withitn|>": "",
	"<|woitn|>": "",
	"<|GBG|>": "",
	"<|Event_UNK|>": "",
}

lang_dict =  {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

# 中英文翻译字典
translations = {
    "en": {
        "html_title": "Voice Understanding Model: SenseVoice-Small",
        "html_desc": "SenseVoice-Small is an encoder-only speech foundation model designed for rapid voice understanding. It encompasses a variety of features including automatic speech recognition (ASR), spoken language identification (LID), speech emotion recognition (SER), and acoustic event detection (AED). SenseVoice-Small supports multilingual recognition for Chinese, English, Cantonese, Japanese, and Korean. Additionally, it offers exceptionally low inference latency, performing 7 times faster than Whisper-small and 17 times faster than Whisper-large.",
        "html_usage": "Usage",
        "html_usage_desc": "Upload an audio file or input through a microphone, then select the task and language. the audio is transcribed into corresponding text along with associated emotions (😊 happy, 😡 angry/exicting, 😔 sad) and types of sound events (😀 laughter, 🎼 music, 👏 applause, 🤧 cough&sneeze, 😭 cry). The event labels are placed in the front of the text and the emotion are in the back of the text.",
        "html_usage_recommend": "Recommended audio input duration is below 30 seconds. For audio longer than 30 seconds, local deployment is recommended.",
        "html_repo": "Repo",
        "html_sensevoice": "SenseVoice: multilingual speech understanding model",
        "html_funasr": "FunASR: fundamental speech recognition toolkit",
        "html_cosyvoice": "CosyVoice: high-quality multilingual TTS model",
        "audio_label": "Upload audio or use the microphone",
        "config_title": "Configuration",
        "language_label": "Language",
        "language_auto": "auto",
        "language_zh": "zh",
        "language_en": "en",
        "language_yue": "yue",
        "language_ja": "ja",
        "language_ko": "ko",
        "language_nospeech": "nospeech",
        "start_button": "Start",
        "results_label": "Results",
        "copy_button": "Copy Results",
        "download_button": "Download Results",
        "download_filename": "sensevoice_results.txt"
    },
    "zh": {
        "html_title": "语音理解模型: SenseVoice-Small",
        "html_desc": "SenseVoice-Small是一个仅使用编码器的语音基础模型，专为快速语音理解而设计。它包含多种功能，包括自动语音识别（ASR）、口语语言识别（LID）、语音情感识别（SER）和声学事件检测（AED）。SenseVoice-Small支持中文、英文、粤语、日语和韩语的多语言识别。此外，它具有极低的推理延迟，比Whisper-small快7倍，比Whisper-large快17倍。",
        "html_usage": "使用方法",
        "html_usage_desc": "上传音频文件或通过麦克风输入，然后选择任务和语言。音频将被转录为相应的文本，并带有相关的情感 (😊 happy, 😡 angry/exicting, 😔 sad) and types of sound events (😀 laughter, 🎼 music, 👏 applause, 🤧 cough&sneeze, 😭 cry)。事件标签位于文本前面，情感标签位于文本后面。",
        "html_usage_recommend": "建议音频输入时长在30秒以内。对于超过30秒的音频，建议本地部署。",
        "html_repo": "项目仓库",
        "html_sensevoice": "SenseVoice: 多语言语音理解模型",
        "html_funasr": "FunASR: 基础语音识别工具包",
        "html_cosyvoice": "CosyVoice: 高质量多语言TTS模型",
        "audio_label": "上传音频或使用麦克风",
        "config_title": "配置",
        "language_label": "语言",
        "language_auto": "auto",
        "language_zh": "zh",
        "language_en": "en",
        "language_yue": "yue",
        "language_ja": "ja",
        "language_ko": "ko",
        "language_nospeech": "nospeech",
        "start_button": "开始",
        "results_label": "结果",
        "copy_button": "复制结果",
        "download_button": "下载结果",
        "download_filename": "sensevoice_结果.txt"
    }
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷",}

def format_str(s):
	for sptk in emoji_dict:
		s = s.replace(sptk, emoji_dict[sptk])
	return s


def format_str_v2(s):
	sptk_dict = {}
	for sptk in emoji_dict:
		sptk_dict[sptk] = s.count(sptk)
		s = s.replace(sptk, "")
	emo = "<|NEUTRAL|>"
	for e in emo_dict:
		if sptk_dict[e] > sptk_dict[emo]:
			emo = e
	for e in event_dict:
		if sptk_dict[e] > 0:
			s = event_dict[e] + s
	s = s + emo_dict[emo]

	for emoji in emo_set.union(event_set):
		s = s.replace(" " + emoji, emoji)
		s = s.replace(emoji + " ", emoji)
	return s.strip()

def format_str_v3(s):
	def get_emo(s):
		return s[-1] if s[-1] in emo_set else None
	def get_event(s):
		return s[0] if s[0] in event_set else None

	s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
	for lang in lang_dict:
		s = s.replace(lang, "<|lang|>")
	s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
	new_s = " " + s_list[0]
	cur_ent_event = get_event(new_s)
	for i in range(1, len(s_list)):
		if len(s_list[i]) == 0:
			continue
		if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
			s_list[i] = s_list[i][1:]
		#else:
		cur_ent_event = get_event(s_list[i])
		if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
			new_s = new_s[:-1]
		new_s += s_list[i].strip().lstrip()
	new_s = new_s.replace("The.", " ")
	return new_s.strip()

def model_inference(input_wav, language, fs=16000):
	# task_abbr = {"Speech Recognition": "ASR", "Rich Text Transcription": ("ASR", "AED", "SER")}
	language_abbr = {"auto": "auto", "zh": "zh", "en": "en", "yue": "yue", "ja": "ja", "ko": "ko",
					 "nospeech": "nospeech"}
	
	# task = "Speech Recognition" if task is None else task
	language = "auto" if len(language) < 1 else language
	selected_language = language_abbr[language]
	# selected_task = task_abbr.get(task)
	
	# print(f"input_wav: {type(input_wav)}, {input_wav[1].shape}, {input_wav}")
	
	if isinstance(input_wav, tuple):
		fs, input_wav = input_wav
		input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max
		if len(input_wav.shape) > 1:
			input_wav = input_wav.mean(-1)
		if fs != 16000:
			print(f"audio_fs: {fs}")
			resampler = torchaudio.transforms.Resample(fs, 16000)
			input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
			input_wav = resampler(input_wav_t[None, :])[0, :].numpy()

	# Create a list to store logs
	logs = []

	merge_vad = True #False if selected_task == "ASR" else True
	log_msg = f"language: {language}, merge_vad: {merge_vad}"
	print(log_msg)
	# logs.append(log_msg)
	
	text = model.generate(input=input_wav,
					  cache={},
					  language=language,
					  use_itn=True,
					  batch_size_s=60, merge_vad=merge_vad)
	
	log_msg = str(text)
	print(log_msg)
	logs.append(log_msg)
	
	text = text[0]["text"]
	formatted_text = format_str_v3(text)
	
	log_msg = formatted_text
	print(log_msg)
	# logs.append(log_msg)
	
	# Join logs with newlines
	log_output = "\n\n".join(logs)
	
	return formatted_text, log_output


audio_examples = [
    ["example/zh.mp3", "zh"],
    ["example/yue.mp3", "yue"],
    ["example/en.mp3", "en"],
    ["example/ja.mp3", "ja"],
    ["example/ko.mp3", "ko"],
    # ["example/emo_1.wav", "auto"],
    # ["example/emo_2.wav", "auto"],
    # ["example/emo_3.wav", "auto"],
    #["example/emo_4.wav", "auto"],
    #["example/event_1.wav", "auto"],
    #["example/event_2.wav", "auto"],
    #["example/event_3.wav", "auto"],
    # ["example/rich_1.wav", "auto"],
    # ["example/rich_2.wav", "auto"],
    #["example/rich_3.wav", "auto"],
    # ["example/longwav_1.wav", "auto"],
    # ["example/longwav_2.wav", "auto"],
    # ["example/longwav_3.wav", "auto"],
    #["example/longwav_4.wav", "auto"],
]



def generate_html_content(lang):
    # 根据语言生成HTML内容
    t = translations[lang]
    return f"""
<div>
    <h2 style="font-size: 22px;margin-left: 0px;">{t['html_title']}</h2>
    <p style="font-size: 18px;margin-left: 20px;">{t['html_desc']}</p>
    <h2 style="font-size: 22px;margin-left: 0px;">{t['html_usage']}</h2> <p style="font-size: 18px;margin-left: 20px;">{t['html_usage_desc']}</p>
	<p style="font-size: 18px;margin-left: 20px;">{t['html_usage_recommend']}</p>
	<h2 style="font-size: 22px;margin-left: 0px;">{t['html_repo']}</h2>
	<p style="font-size: 18px;margin-left: 20px;"><a href="https://github.com/FunAudioLLM/SenseVoice" target="_blank">SenseVoice</a>: {t['html_sensevoice'].split(': ')[1]}</p>
	<p style="font-size: 18px;margin-left: 20px;"><a href="https://github.com/modelscope/FunASR" target="_blank">FunASR</a>: {t['html_funasr'].split(': ')[1]}</p>
	<p style="font-size: 18px;margin-left: 20px;"><a href="https://github.com/FunAudioLLM/CosyVoice" target="_blank">CosyVoice</a>: {t['html_cosyvoice'].split(': ')[1]}</p>
</div>
"""


def launch():
	# 检测系统语言环境
	lang = None
	try:
		lang, _ = locale.getlocale()
		# 检测中文语言环境，支持不同格式：zh_CN、Chinese (Simplified)_China等
		current_lang = 'zh' if lang and ('zh' in lang.lower() or 'chinese' in lang.lower()) else 'en'
	except Exception as e:
		# 如果获取语言环境失败，默认使用英文
		print(f"Failed to get locale: {e}")
		current_lang = 'en'
	

	print(f"lang: {lang}")
	print(f"current_lang: {current_lang}")

	
	with gr.Blocks() as demo:
		# gr.Markdown(description)
		gr.HTML(generate_html_content(current_lang))
		with gr.Row():
			with gr.Column():
				t = translations[current_lang]
				audio_inputs = gr.Audio(label=t['audio_label'])
				
				with gr.Accordion(t['config_title']):
					language_inputs = gr.Dropdown(
					value="auto",
					label=t['language_label'],
					# 动态设置下拉选项的显示文本和实际值
					choices=[
						(t['language_auto'], "auto"),
						(t['language_zh'], "zh"),
						(t['language_en'], "en"),
						(t['language_yue'], "yue"),
						(t['language_ja'], "ja"),
						(t['language_ko'], "ko"),
						(t['language_nospeech'], "nospeech")
					]
				)
				fn_button = gr.Button(t['start_button'], variant="primary")
				text_outputs = gr.Textbox(label=t['results_label'], lines=10, max_lines=20, scale=2)
				with gr.Row():
					copy_button = gr.Button(t['copy_button'], variant="secondary")
					download_button = gr.Button(t['download_button'], variant="secondary")
			
				# Add copy functionality using JavaScript (lightweight without alert)
				copy_button.click(
					None,
					inputs=[text_outputs],
					outputs=None,
					js="""
					async (text) => {
						if (text) {
							await navigator.clipboard.writeText(text);
						}
					}
					"""
				)
				
				# Add download functionality using JavaScript (direct download)
				download_button.click(
					None,
					inputs=[text_outputs],
					outputs=None,
					js=f"""
					(text) => {{
						if (text) {{
							const blob = new Blob([text], {{ type: 'text/plain;charset=utf-8' }});
							const url = URL.createObjectURL(blob);
							const a = document.createElement('a');
							a.href = url;
							a.download = '{t["download_filename"]}';
							document.body.appendChild(a);
							a.click();
							document.body.removeChild(a);
							URL.revokeObjectURL(url);
						}}
					}}
					"""
				)
				
				# Add log display component
				log_outputs = gr.Textbox(label="Logs", lines=5, max_lines=20, interactive=False)
			
			gr.Examples(examples=audio_examples, inputs=[audio_inputs, language_inputs], examples_per_page=20)
		
		# Update button click to handle multiple outputs
		fn_button.click(model_inference, inputs=[audio_inputs, language_inputs], outputs=[text_outputs, log_outputs])

	demo.launch()


if __name__ == "__main__":
	# iface.launch()
	launch()


