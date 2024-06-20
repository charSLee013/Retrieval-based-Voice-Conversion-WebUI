import numpy as np, parselmouth, torch, pdb, sys, os
from time import time as ttime
import torch.nn.functional as F
import scipy.signal as signal
import pyworld, os, traceback, faiss, librosa, torchcrepe
from scipy import signal
from functools import lru_cache

now_dir = os.getcwd()
sys.path.append(now_dir)

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

input_audio_path2wav = {}


@lru_cache
def cache_harvest_f0(input_audio_path, fs, f0max, f0min, frame_period):
    audio = input_audio_path2wav[input_audio_path]
    f0, t = pyworld.harvest(
        audio,
        fs=fs,
        f0_ceil=f0max,
        f0_floor=f0min,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(audio, f0, t, fs)
    return f0


def change_rms(data1, sr1, data2, sr2, rate):  # 1是输入音频，2是输出音频,rate是2的占比
    # print(data1.max(),data2.max())
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


class VC(object):
    def __init__(self, tgt_sr, config):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.sr = 16000  # hubert输入采样率
        self.window = 160  # 每帧点数
        self.t_pad = self.sr * self.x_pad  # 每条前后pad时间
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # 查询切点前后查询时间
        self.t_center = self.sr * self.x_center  # 查询切点位置
        self.t_max = self.sr * self.x_max  # 免查询时长阈值
        self.device = config.device

    def get_f0(
        self,
        input_audio_path,
        x,
        p_len,
        f0_up_key,
        f0_method,
        filter_radius,
        inp_f0=None,
    ):
        global input_audio_path2wav
        time_step = self.window / self.sr * 1000
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        if f0_method == "pm":
            f0 = (
                parselmouth.Sound(x, self.sr)
                .to_pitch_ac(
                    time_step=time_step / 1000,
                    voicing_threshold=0.6,
                    pitch_floor=f0_min,
                    pitch_ceiling=f0_max,
                )
                .selected_array["frequency"]
            )
            pad_size = (p_len - len(f0) + 1) // 2
            if pad_size > 0 or p_len - len(f0) - pad_size > 0:
                f0 = np.pad(
                    f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant"
                )
        elif f0_method == "harvest":
            input_audio_path2wav[input_audio_path] = x.astype(np.double)
            f0 = cache_harvest_f0(input_audio_path, self.sr, f0_max, f0_min, 10)
            if filter_radius > 2:
                f0 = signal.medfilt(f0, 3)
        elif f0_method == "crepe":
            model = "full"
            # Pick a batch size that doesn't cause memory errors on your gpu
            batch_size = 512
            # Compute pitch using first gpu
            audio = torch.tensor(np.copy(x))[None].float()
            f0, pd = torchcrepe.predict(
                audio,
                self.sr,
                self.window,
                f0_min,
                f0_max,
                model,
                batch_size=batch_size,
                device=self.device,
                return_periodicity=True,
            )
            pd = torchcrepe.filter.median(pd, 3)
            f0 = torchcrepe.filter.mean(f0, 3)
            f0[pd < 0.1] = 0
            f0 = f0[0].cpu().numpy()
        elif f0_method == "rmvpe":
            if hasattr(self, "model_rmvpe") == False:
                from lib.rmvpe import RMVPE

                print("loading rmvpe model")
                self.model_rmvpe = RMVPE(
                    "rmvpe.pt", is_half=self.is_half, device=self.device
                )

            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
            if "privateuseone" in str(self.device):  # clean ortruntime memory
                del self.model_rmvpe.model
                del self.model_rmvpe
                print("cleaning ortruntime memory")

        f0 *= pow(2, f0_up_key / 12)
        # with open("test.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        tf0 = self.sr // self.window  # 每秒f0点数
        if inp_f0 is not None:
            delta_t = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
            ).astype("int16")
            replace_f0 = np.interp(
                list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1]
            )
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[
                :shape
            ]
        # with open("test_opt.txt","w")as f:f.write("\n".join([str(i)for i in f0.tolist()]))
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(np.int32)
        return f0_coarse, f0bak  # 1-0

    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        times,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):  # ,file_index,file_big_npy
        # 将输入的音频数据 audio0 转换为 PyTorch 张量 feats
        feats = torch.from_numpy(audio0)
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()
        if feats.dim() == 2:  # double channels
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        # 创建一个填充掩码 padding_mask，用于模型输入，初始化为 False
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs = {
            "source": feats.to(self.device),    # 源特征
            "padding_mask": padding_mask,   # 填充掩码
            "output_layer": 9 if version == "v1" else 12,   # 输出层的选择
        }
        t0 = ttime()
        
        # 使用HubertModel模型提取音频特征
        with torch.no_grad():
            #  将音频信号传递给模型推理并且获取模型的中间特征
            # 返回feature 提取的特征，通常是模型中间层的输出 和 padding_mask填充掩码，用于指示输入序列中的填充部分，这些部分在计算中应该被忽略
            logits = model.extract_features(**inputs)   
            # final_proj函数是一个线性层，其作用是将模型的输出从编码器的嵌入维度映射到一个更小的维度，或者根据模型配置映射到多个目标维度。这个过程通常发生在模型的最后阶段，用于准备输出以进行最终的预测或损失计算。
            # 如果模型初始化的时候配置了untie_final_proj为True，final_proj会为每个目标（如不同的语言或任务）分别进行投影，这意味着每个目标都有自己的投影矩阵。
            # 总结：final_proj是一个线性层，用于进行最终的线性变换，将模型的输出映射到一个适合进行预测的维度
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
        
        # 特征融合（如果需要）
        if protect < 0.5 and pitch != None and pitchf != None:
            feats0 = feats.clone()
        # 如果提供了索引 index 和特征数据库 big_npy，则进行特征融合
        if (
            isinstance(index, type(None)) == False
            and isinstance(big_npy, type(None)) == False
            and index_rate != 0
        ):
            npy = feats[0].cpu().numpy()
            if self.is_half:
                npy = npy.astype("float32")

            # _, I = index.search(npy, 1)
            # npy = big_npy[I.squeeze()]
            # 使用 index 在 big_npy 中搜索相似特征，并计算权重。
            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            # 根据权重对搜索到的特征进行加权求和，得到融合后的特征 npy
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            if self.is_half:
                npy = npy.astype("float16")
            # 将 npy 转换回 PyTorch 张量，并与原始特征 feats 按比例融合。
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        # 对特征 feats 进行上采样，以匹配音频的时间分辨率
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch != None and pitchf != None:
            # 如果需要保护机制，则对 feats0 也进行上采样。
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
        t1 = ttime()
        p_len = audio0.shape[0] // self.window  # 计算音频信号被分割成多少个帧（frames）。这里的 self.window 通常指的是分析窗口的长度，即每一帧音频数据的时间长度。
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch != None and pitchf != None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]
        # 如果 protect 小于 0.5 且提供了音高数据 pitch 和 pitchf，则应用保护机制
        if protect < 0.5 and pitch != None and pitchf != None:
            # 创建保护掩码 pitchff，根据音高数据调整其值。
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            # 将特征 feats 与保护掩码 pitchff 相乘，然后加上原始特征 feats0 与 (1 - pitchff) 的乘积。
            feats = feats * pitchff + feats0 * (1 - pitchff)
            # 确保 feats 的数据类型与 feats0 一致。
            feats = feats.to(feats0.dtype)
        # 计算音频帧的长度 p_len
        p_len = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            if pitch != None and pitchf != None:
                # 如果提供了音高数据则一起推理
                audio1 = (
                    (net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0])
                    .data.cpu()
                    .float()
                    .numpy()
                )
            else:
                # 使用SynthesizerTrnMs768NSFsid_nono根据目标音频特征feats和声码编码sid重新推理出音频
                audio1 = (
                    (net_g.infer(feats, p_len, sid)[0][0, 0]).data.cpu().float().numpy()
                )
        del feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        t2 = ttime()
        times[0] += t1 - t0
        times[2] += t2 - t1
        return audio1

    def pipeline(
        self,
        model,
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
        protect,
        f0_file=None,
        ret_audio_opt=True,
    ):
        # 打印所有参数
        print(f"Parameters of the 'pipeline' function:")
        print(f"sid: {sid}")
        print(f"audio: {audio}")
        print(f"input_audio_path: {input_audio_path}")
        print(f"times: {times}")
        print(f"f0_up_key: {f0_up_key}")
        print(f"f0_method: {f0_method}")
        print(f"file_index: {file_index}")
        print(f"index_rate: {index_rate}")
        print(f"if_f0: {if_f0}")
        print(f"filter_radius: {filter_radius}")
        print(f"tgt_sr: {tgt_sr}")
        print(f"resample_sr: {resample_sr}")
        print(f"rms_mix_rate: {rms_mix_rate}")
        print(f"version: {version}")
        print(f"protect: {protect}")
        print(f"f0_file: {f0_file}")
        print("+---------------------------------------+")
        if (
            file_index != ""
            # and file_big_npy != ""
            # and os.path.exists(file_big_npy) == True
            and os.path.exists(file_index) == True
            and index_rate != 0
        ):
            try:
                index = faiss.read_index(file_index)
                # big_npy = np.load(file_big_npy)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except:
                traceback.print_exc()
                index = big_npy = None
        else:
            index = big_npy = None
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t = None
        t1 = ttime()
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        if hasattr(f0_file, "name") == True:
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = []
                for line in lines:
                    inp_f0.append([float(i) for i in line.split(",")])
                inp_f0 = np.array(inp_f0, dtype="float32")
            except:
                traceback.print_exc()
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = self.get_f0(
                input_audio_path,
                audio_pad,
                p_len,
                f0_up_key,
                f0_method,
                filter_radius,
                inp_f0,
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if self.device == "mps":
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        t2 = ttime()
        times[1] += t2 - t1
        for t in opt_ts:
            t = t // self.window * self.window
            if if_f0 == 1:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        times,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            s = t
        if if_f0 == 1:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    pitch[:, t // self.window :] if t is not None else pitch,
                    pitchf[:, t // self.window :] if t is not None else pitchf,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    None,
                    None,
                    times,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        audio_opt = np.concatenate(audio_opt)
        # 这里根据rms_mix_rate调整音频的响度（Root Mean Square，RMS）。
        # 如果rms_mix_rate不等于1，会调用change_rms函数，可能是为了将原始音频audio与处理后的音频audio_opt在响度上进行混合，或者直接调整audio_opt的响度，使之与某种参考水平相匹配。
        # tgt_sr是目标采样率，此步可能用于响度均衡，确保输出音频的音量与输入或其他音频相协调
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )
        if ret_audio_opt:
            # 直接返回推理后的音频不做任何后处理
            return audio_opt
        # 首先计算音频的最大绝对值并除以0.99以避免精确的饱和，然后根据这个最大值调整量化范围，保证音频信号不会溢出。
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        # 在之前的步骤中，audio_opt 是浮点数类型的音频信号，其振幅已经被归一化到接近1的范围（例如，最大振幅被限制在0.99以内）。这是为了防止音频信号在后续处理中出现溢出。
        # audio_opt * max_int16 这一步操作将每个浮点数振幅乘以 max_int16，从而将振幅放大到 int16 格式的范围内。这样做的目的是为了将浮点数音频信号转换为整数音频信号，以便于存储和播放。
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt
