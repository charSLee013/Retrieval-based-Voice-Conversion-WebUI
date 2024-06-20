from scipy.io import wavfile
from fairseq import checkpoint_utils
import torchaudio
from lib.audio import load_audio
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from vc_infer_pipeline import VC
from multiprocessing import cpu_count
import numpy as np
import torch
import sys
import glob
import argparse
import os
import sys
import pdb
import torch

now_dir = os.getcwd()
sys.path.append(now_dir)

####
# USAGE
#
# In your Terminal or CMD or whatever
# python infer_cli.py [TRANSPOSE_VALUE] "[INPUT_PATH]" "[OUTPUT_PATH]" "[MODEL_PATH]" "[INDEX_FILE_PATH]" "[INFERENCE_DEVICE]" "[METHOD]"

using_cli = False
device = "cuda:0" if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
is_half = False

if len(sys.argv) > 2:
    f0_up_key = int(sys.argv[1])  # transpose value
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    model_path = sys.argv[4]
    file_index = sys.argv[5]  # .index file
    device = sys.argv[6]
    f0_method = sys.argv[7]  # pm or harvest or crepe

    using_cli = True

    # file_index2=sys.argv[8]
    # index_rate=float(sys.argv[10]) #search feature ratio
    # filter_radius=float(sys.argv[11]) #median filter
    # resample_sr=float(sys.argv[12]) #resample audio in post processing
    # rms_mix_rate=float(sys.argv[13]) #search feature
    print(sys.argv)
else:
    # 硬编码
    f0_up_key = 0
    input_path = "output.wav"
    output_path = "opt/record.wav"
    model_path = "weights/three_moon_e20_s10000.pth"
    file_index = ''
    f0_method = 'rmvpe'
    using_cli = True
    
if device == 'mps':
    # 设置环境变量 PYTORCH_ENABLE_MPS_FALLBACK=1
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

class Config:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available() and device != "cpu":
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                print("16系/10系显卡和P40强制单精度")
                self.is_half = False
                for config_file in ["32k.json", "40k.json", "48k.json"]:
                    with open(f"configs/{config_file}", "r") as f:
                        strr = f.read().replace("true", "false")
                    with open(f"configs/{config_file}", "w") as f:
                        f.write(strr)
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("trainset_preprocess_pipeline_print.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("trainset_preprocess_pipeline_print.py", "w") as f:
                    f.write(strr)
        elif torch.backends.mps.is_available():
            print("没有发现支持的N卡, 使用MPS进行推理")
            self.device = "mps"
        else:
            print("没有发现支持的N卡, 使用CPU进行推理")
            self.device = "cpu"
            self.is_half = False

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max


config = Config(device, is_half)
now_dir = os.getcwd()
sys.path.append(now_dir)

hubert_model = None


def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


def vc_single(
    sid=0,
    input_audio_path=None,
    f0_up_key=0,
    f0_file=None,
    f0_method="pm",
    file_index="",  # .index file
    file_index2="",
    # file_big_npy,
    index_rate=0.0,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=0,
    model_path="",
    output_path="",
    protect=0.33,
):
    # 打印每个参数
    print(f"sid: {sid}")
    print(f"f0_up_key: {f0_up_key}")
    print(f"f0_method: {f0_method}")
    print(f"file_index: {file_index}")
    print(f"file_index2: {file_index2}")
    print(f"index_rate: {index_rate}")
    print(f"filter_radius: {filter_radius}")
    print(f"resample_sr: {resample_sr}")
    print(f"rms_mix_rate: {rms_mix_rate}")
    print(f"protect: {protect}")
    
    global tgt_sr, net_g, vc, hubert_model, version
    get_vc(model_path)
    if input_audio_path is None:
        return "You need to upload an audio file", None

    f0_up_key = int(f0_up_key)
    audio = load_audio(input_audio_path, 16000)
    print(audio.shape)
    audio_max = np.abs(audio).max() / 0.95

    if audio_max > 1:
        audio /= audio_max
    times = [0, 0, 0]

    if hubert_model == None:
        load_hubert()

    if_f0 = cpt.get("f0", 1)

    file_index = (
        (
            file_index.strip(" ")
            .strip('"')
            .strip("\n")
            .strip('"')
            .strip(" ")
            .replace("trained", "added")
        )
        if file_index != ""
        else file_index2
    )

    audio_opt = vc.pipeline(
        hubert_model,
        net_g,
        sid,
        audio,
        input_audio_path,
        times,
        f0_up_key,
        f0_method,
        file_index,
        # file_big_npy,
        index_rate,
        if_f0,
        filter_radius,
        tgt_sr,
        resample_sr,
        rms_mix_rate,
        version,
        f0_file=f0_file,
        protect=protect,
    )
    # audio_opt = audio_opt.astype(np.int32)  # convert to pcm_f32le
    wavfile.write(output_path, tgt_sr, audio_opt)
    return "processed"


def get_vc(model_path):
    global n_spk, tgt_sr, net_g, vc, cpt, device, is_half, version
    print("loading pth %s" % model_path)
    cpt = torch.load(model_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(device)
    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    # return {"visible": True,"maximum": n_spk, "__type__": "update"}


if using_cli:
    vc_single(
        sid=0,
        input_audio_path=input_path,
        f0_up_key=f0_up_key,
        f0_file=None,
        f0_method=f0_method,
        file_index=file_index,
        file_index2="",
        index_rate=1,
        filter_radius=3,
        resample_sr=0,
        rms_mix_rate=1,
        model_path=model_path,
        output_path=output_path,
    )
