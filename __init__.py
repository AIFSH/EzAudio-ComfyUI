import os,sys
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)

import torch
import torchaudio
import numpy as np
from accelerate import Accelerator
from diffusers import DDIMScheduler
from ezaudio.utils import load_yaml_with_includes
from ezaudio.models.conditioners import MaskDiT
from ezaudio.models.controlnet import DiTControlNet
from ezaudio.models.conditions import Conditioner
from ezaudio.inference import inference
from ezaudio.inference_controlnet import inference as inference_cn
from ezaudio.modules.autoencoder_wrapper import Autoencoder
from transformers import T5Tokenizer, T5EncoderModel
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model and configs
def load_models(config_name, ckpt_path, vae_path, device):
    params = load_yaml_with_includes(config_name)

    # Load codec model
    autoencoder = Autoencoder(ckpt_path=vae_path,
                              model_type=params['autoencoder']['name'],
                              quantization_first=params['autoencoder']['q_first']).to(device)
    autoencoder.eval()

    # Load text encoder
    tokenizer = T5Tokenizer.from_pretrained(params['text_encoder']['model'])
    text_encoder = T5EncoderModel.from_pretrained(params['text_encoder']['model']).to(device)
    text_encoder.eval()

    # Load main U-Net model
    unet = MaskDiT(**params['model']).to(device)
    unet.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model'])
    unet.eval()

    accelerator = Accelerator(mixed_precision="fp16")
    unet = accelerator.prepare(unet)

    # Load noise scheduler
    noise_scheduler = DDIMScheduler(**params['diff'])

    latents = torch.randn((1, 128, 128), device=device)
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device)
    _ = noise_scheduler.add_noise(latents, noise, timesteps)

    return autoencoder, unet, tokenizer, text_encoder, noise_scheduler, params


# Load cn model and configs
def load_cn_models(config_name, ckpt_path, controlnet_path, vae_path, device):
    params = load_yaml_with_includes(config_name)

    # Load codec model
    autoencoder = Autoencoder(ckpt_path=vae_path,
                              model_type=params['autoencoder']['name'],
                              quantization_first=params['autoencoder']['q_first']).to(device)
    autoencoder.eval()

    # Load text encoder
    tokenizer = T5Tokenizer.from_pretrained(params['text_encoder']['model'])
    text_encoder = T5EncoderModel.from_pretrained(params['text_encoder']['model']).to(device)
    text_encoder.eval()

    # Load main U-Net model
    unet = MaskDiT(**params['model']).to(device)
    unet.load_state_dict(torch.load(ckpt_path, map_location='cpu')['model'])
    unet.eval()

    controlnet_config = params['model'].copy()
    controlnet_config.update(params['controlnet'])
    controlnet = DiTControlNet(**controlnet_config).to(device)
    controlnet.eval()
    controlnet.load_state_dict(torch.load(controlnet_path, map_location='cpu')['model'])
    conditioner = Conditioner(**params['conditioner']).to(device)

    accelerator = Accelerator(mixed_precision="fp16")
    unet, controlnet = accelerator.prepare(unet, controlnet)

    # Load noise scheduler
    noise_scheduler = DDIMScheduler(**params['diff'])

    latents = torch.randn((1, 128, 128), device=device)
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device)
    _ = noise_scheduler.add_noise(latents, noise, timesteps)

    return autoencoder, unet, controlnet, conditioner, tokenizer, text_encoder, noise_scheduler, params


class TextPromptNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}), 
            }
        }
    RETURN_TYPES = ("TEXT",)
    FUNCTION = "encode"

    CATEGORY = "AIFSH_EzAudio"
    
    def encode(self, text):
        return (text, )

class EzAudioNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt":("TEXT",),
                "neg_prompt":("TEXT",),
                "audio_length":("INT",{
                    "default": 10,
                    "min": 1,
                    "max":10,
                    "step":1,
                    "display":"slider"
                }),
                "guidance_scale":("FLOAT",{
                    "default": 5.0,
                    "min": 1.0,
                    "max":10.0,
                    "step":0.5,
                    "round":0.01,
                    "display":"slider"
                }),
                "guidance_rescale":("FLOAT",{
                    "default": 0.75,
                    "min": 0.0,
                    "max":1.0,
                    "step":0.05,
                    "round":0.001,
                    "display":"slider"
                }),
                "ddim_steps":("INT",{
                    "default": 50,
                    "min": 25,
                    "max":200,
                    "step":5,
                    "display":"slider"
                }),
                "eta":("FLOAT",{
                    "default": 1.0,
                    "min": 0.0,
                    "max":1.0,
                    "step":0.1,
                    "round":0.01,
                    "display":"slider"
                }),
                "seed":("INT",{
                    "default":42
                })
            }
        }
    

    RETURN_TYPES = ("AUDIO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_audio"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_EzAudio"

    def cfy2librosa(self,audio,target_sr):
        waveform = audio["waveform"].squeeze(0)
        sr = audio["sample_rate"]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0,keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
        print(waveform.shape)
        print(f"from {sr} to {target_sr}")
        return waveform.numpy()[0]
    
    def gen_audio(self,prompt,neg_prompt,audio_length,guidance_scale,
                  guidance_rescale,ddim_steps,eta,seed):
        # Model and config paths
        config_name = os.path.join(now_dir,'ckpts/ezaudio-xl.yml')
        ckpt_path = os.path.join(now_dir,'ckpts/s3/ezaudio_s3_xl.pt')
        vae_path = os.path.join(now_dir,'ckpts/vae/1m.pt')
        autoencoder, unet, tokenizer, text_encoder, noise_scheduler, params = load_models(config_name, ckpt_path, vae_path,device)
 
        sr = params['autoencoder']['sr']
        print(f"target_sr:{sr}")

        gt, gt_mask = None, None

        pred = inference(autoencoder, unet,
                     gt, gt_mask,
                     tokenizer, text_encoder,
                     params, noise_scheduler,
                     prompt, neg_prompt,
                     audio_length,
                     guidance_scale, guidance_rescale,
                     ddim_steps, eta, seed,
                     device)

        pred = pred.cpu().numpy().squeeze(0).squeeze(0)

        res_audio = {
                        "waveform": torch.FloatTensor(pred).unsqueeze(0).unsqueeze(0),
                        "sample_rate": sr
                    }
        return (res_audio,)

class EzAudioEditNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt":("TEXT",),
                "audio":("AUDIO",),
                "mask_start":("FLOAT",{
                    "default": 0,
                }),
                "mask_length":("FLOAT",{
                    "default": 3.0,
                    "min": 0.5,
                    "max":10.0,
                    "step":0.5,
                    "round":0.01,
                    "display":"slider"
                }),
                "boundary":("FLOAT",{
                    "default": 2.0,
                    "min": 0.5,
                    "max":4.0,
                    "step":0.5,
                    "round":0.01,
                    "display":"slider"
                }),
                "guidance_scale":("FLOAT",{
                    "default": 5.0,
                    "min": 1.0,
                    "max":10.0,
                    "step":0.5,
                    "round":0.01,
                    "display":"slider"
                }),
                "guidance_rescale":("FLOAT",{
                    "default": 0.75,
                    "min": 0.0,
                    "max":1.0,
                    "step":0.05,
                    "round":0.001,
                    "display":"slider"
                }),
                "ddim_steps":("INT",{
                    "default": 50,
                    "min": 25,
                    "max":200,
                    "step":5,
                    "display":"slider"
                }),
                "eta":("FLOAT",{
                    "default": 1.0,
                    "min": 0.0,
                    "max":1.0,
                    "step":0.1,
                    "round":0.01,
                    "display":"slider"
                }),
                "seed":("INT",{
                    "default":42
                })
            }
        }
    

    RETURN_TYPES = ("AUDIO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_audio"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_EzAudio"

    def cfy2librosa(self,audio,target_sr):
        waveform = audio["waveform"].squeeze(0)
        sr = audio["sample_rate"]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0,keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
        print(waveform.shape)
        print(f"from {sr} to {target_sr}")
        return waveform.numpy()[0]
    
    def gen_audio(self,prompt,audio,mask_start, mask_length,
                  boundary,guidance_scale,
                  guidance_rescale,ddim_steps,eta,seed):
        # Model and config paths
        config_name = os.path.join(now_dir,'ckpts/ezaudio-xl.yml')
        ckpt_path = os.path.join(now_dir,'ckpts/s3/ezaudio_s3_xl.pt')
        vae_path = os.path.join(now_dir,'ckpts/vae/1m.pt')
        autoencoder, unet, tokenizer, text_encoder, noise_scheduler, params = load_models(config_name, ckpt_path, vae_path,device)
 
        sr = params['autoencoder']['sr']
        print(f"target_sr:{sr}")

        neg_text = None

        mask_end = mask_start + mask_length

        # Load and preprocess ground truth audio
        gt = self.cfy2librosa(audio,sr)
        gt = gt / (np.max(np.abs(gt)) + 1e-9)

        audio_length = len(gt) / sr
        mask_start = min(mask_start, audio_length)
        if mask_end > audio_length:
            # outpadding mode
            padding = round((mask_end - audio_length)*params['autoencoder']['sr'])
            gt = np.pad(gt, (0, padding), 'constant')
            audio_length = len(gt) / sr

        output_audio = gt.copy()

        gt = torch.tensor(gt).unsqueeze(0).unsqueeze(1).to(device)
        boundary = min((mask_end - mask_start)/2, boundary)
        # print(boundary)

        # Calculate start and end indices
        start_idx = max(mask_start - boundary, 0)
        end_idx = min(mask_end + boundary, audio_length)
        # print(start_idx)
        # print(end_idx)

        mask_start -= start_idx
        mask_end -= start_idx

        gt = gt[:, :, round(start_idx*params['autoencoder']['sr']):round(end_idx*params['autoencoder']['sr'])]

        # Encode the audio to latent space
        gt_latent = autoencoder(audio=gt)
        B, D, L = gt_latent.shape
        length = L

        gt_mask = torch.zeros(B, D, L).to(device)
        latent_sr = params['autoencoder']['latent_sr']
        gt_mask[:, :, round(mask_start * latent_sr): round(mask_end * latent_sr)] = 1
        gt_mask = gt_mask.bool()


        # Perform inference to get the edited latent representation
        pred = inference(autoencoder, unet,
                        gt_latent, gt_mask,
                        tokenizer, text_encoder,
                        params, noise_scheduler,
                        prompt, neg_text,
                        length,
                        guidance_scale, guidance_rescale,
                        ddim_steps, eta, seed,
                        device)

        pred = pred.cpu().numpy().squeeze(0).squeeze(0)

        chunk_length = end_idx - start_idx
        pred = pred[:round(chunk_length*params['autoencoder']['sr'])]

        output_audio[round(start_idx*params['autoencoder']['sr']):round(end_idx*params['autoencoder']['sr'])] = pred

        pred = output_audio

        res_audio = {
                        "waveform": torch.FloatTensor(pred).unsqueeze(0).unsqueeze(0),
                        "sample_rate": sr
                    }
        return (res_audio,)

class EzAudioControlNetNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt":("TEXT",),
                "ref_audio":("AUDIO",),
                "surpass_noise":("FLOAT",{
                    "default": 0.0,
                    "min": 0,
                    "max":1,
                    "step":0.01,
                    "round":0.001,
                    "display":"slider"
                }),
                "guidance_scale":("FLOAT",{
                    "default": 5.0,
                    "min": 1.0,
                    "max":10.0,
                    "step":0.5,
                    "round":0.01,
                    "display":"slider"
                }),
                "guidance_rescale":("FLOAT",{
                    "default": 0.5,
                    "min": 0.0,
                    "max":1.0,
                    "step":0.05,
                    "round":0.001,
                    "display":"slider"
                }),
                "ddim_steps":("INT",{
                    "default": 50,
                    "min": 25,
                    "max":200,
                    "step":5,
                    "display":"slider"
                }),
                "eta":("FLOAT",{
                    "default": 1.0,
                    "min": 0.0,
                    "max":1.0,
                    "step":0.1,
                    "round":0.01,
                    "display":"slider"
                }),
                "conditioning_scale":("FLOAT",{
                    "default": 1.0,
                    "min": 0.0,
                    "max":2.0,
                    "step":0.25,
                    "round":0.001,
                    "display":"slider"
                }),
                "seed":("INT",{
                    "default":42
                })
            }
        }
    

    RETURN_TYPES = ("AUDIO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_audio"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_EzAudio"

    def cfy2librosa(self,audio,target_sr):
        waveform = audio["waveform"].squeeze(0)
        sr = audio["sample_rate"]
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0,keepdim=True)
        if sr != target_sr:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(waveform)
        print(waveform.shape)
        print(f"from {sr} to {target_sr}")
        return waveform.numpy()[0]
    
    def gen_audio(self,prompt,ref_audio,surpass_noise,guidance_scale,
                  guidance_rescale,ddim_steps,eta,conditioning_scale,seed):
        # Model and config paths
        config_name = os.path.join(now_dir,'ckpts/controlnet/energy_l.yml')
        ckpt_path = os.path.join(now_dir,'ckpts/s3/ezaudio_s3_l.pt')
        controlnet_path = os.path.join(now_dir,'ckpts/controlnet/s3_l_energy.pt')
        vae_path = os.path.join(now_dir,'ckpts/vae/1m.pt')
        (autoencoder, unet, controlnet, conditioner, 
        tokenizer, text_encoder, noise_scheduler, params) = load_cn_models(config_name, ckpt_path, controlnet_path, vae_path, device)
        sr = params['autoencoder']['sr']
        print(f"target_sr:{sr}")

        gt = self.cfy2librosa(ref_audio,sr)
        gt = gt / (np.max(np.abs(gt)) + 1e-9)  # Normalize audio

        if surpass_noise > 0:
            mask = np.abs(gt) <= surpass_noise
            gt[mask] = 0
        
        original_length = len(gt)
        # Ensure the audio is of the correct length by padding or trimming
        duration_seconds = min(len(gt) / sr, 10)
        quantized_duration = np.ceil(duration_seconds * 2) / 2  # This rounds to the nearest 0.5 seconds
        num_samples = int(quantized_duration * sr)
        audio_frames = round(num_samples / sr * params['autoencoder']['latent_sr'])

        if len(gt) < num_samples:
            padding = num_samples - len(gt)
            gt = np.pad(gt, (0, padding), 'constant')
        else:
            gt = gt[:num_samples]

        gt_audio = torch.tensor(gt).unsqueeze(0).unsqueeze(1).to(device)
        gt = autoencoder(audio=gt_audio)
        condition = conditioner(gt_audio.squeeze(1), gt.shape)

        # Perform inference
        pred = inference_cn(autoencoder, unet, controlnet,
                        None, None, condition,
                        tokenizer, text_encoder, 
                        params, noise_scheduler,
                        prompt, neg_text=None,
                        audio_frames=audio_frames, 
                        guidance_scale=guidance_scale, guidance_rescale=guidance_rescale, 
                        ddim_steps=ddim_steps, eta=eta, random_seed=seed, 
                        conditioning_scale=conditioning_scale, device=device)

        pred = pred.cpu().numpy().squeeze(0).squeeze(0)[:original_length]
        
        res_audio = {
                        "waveform": torch.FloatTensor(pred).unsqueeze(0).unsqueeze(0),
                        "sample_rate": sr
                    }
        return (res_audio,)

                              

NODE_CLASS_MAPPINGS = {
    "TextPromptNode": TextPromptNode,
    "EzAudioNode":EzAudioNode,
    "EzAudioEditNode":EzAudioEditNode,
    "EzAudioControlNetNode":EzAudioControlNetNode,
}