"""
The main separation by localization inference algorithm
"""

import argparse
import os
import re
from collections import namedtuple

import librosa
import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import glob
from tqdm import tqdm
from pydub import AudioSegment
import cos.helpers.utils as utils

from cos.helpers.constants import ALL_WINDOW_SIZES, \
    FAR_FIELD_RADIUS
from cos.helpers.visualization import draw_diagram
from cos.training.network import CoSNetwork, center_trim, \
    normalize_input, unnormalize_input
from cos.helpers.eval_utils import si_sdr

# Constants which may be tweaked based on your setup
ENERGY_CUTOFF = 0.002
NMS_RADIUS = np.pi / 4
NMS_SIMILARITY_SDR = -7.0  # SDR cutoff for different candidates

CandidateVoice = namedtuple("CandidateVoice", ["angle", "energy", "data"])


def nms(candidate_voices, nms_cutoff):
    """
    Runs non-max suppression on the candidate voices
    """
    final_proposals = []
    initial_proposals = candidate_voices

    while len(initial_proposals) > 0:
        new_initial_proposals = []
        sorted_candidates = sorted(initial_proposals,
                                   key=lambda x: x[1],
                                   reverse=True)

        # Choose the loudest voice
        best_candidate_voice = sorted_candidates[0]
        final_proposals.append(best_candidate_voice)
        sorted_candidates.pop(0)

        # See if any of the rest should be removed
        for candidate_voice in sorted_candidates:
            different_locations = utils.angular_distance(
                candidate_voice.angle, best_candidate_voice.angle) > NMS_RADIUS

            # different_content = abs(
            #     candidate_voice.data -
            #     best_candidate_voice.data).mean() > nms_cutoff

            different_content = si_sdr(
                candidate_voice.data[0],
                best_candidate_voice.data[0]) < nms_cutoff

            if different_locations or different_content:
                new_initial_proposals.append(candidate_voice)

        initial_proposals = new_initial_proposals

    return final_proposals


def forward_pass(model, target_angle, mixed_data, conditioning_label, args):
    """
    Runs the network on the mixed_data
    with the candidate region given by voice
    """
    target_pos = np.array([
        FAR_FIELD_RADIUS * np.cos(target_angle),
        FAR_FIELD_RADIUS * np.sin(target_angle)
    ])

    data, _ = utils.shift_mixture(
        torch.tensor(mixed_data).to(args.device), target_pos, .03231,
        44100)
    data = data.float().unsqueeze(0)  # Batch size is 1

    # Normalize input
    data, means, stds = normalize_input(data)

    # Run through the model
    valid_length = model.valid_length(data.shape[-1])
    delta = valid_length - data.shape[-1]
    padded = F.pad(data, (delta // 2, delta - delta // 2))

    output_signal = model(padded, conditioning_label)
    output_signal = center_trim(output_signal, data)

    output_signal = unnormalize_input(output_signal, means, stds)
    output_voices = output_signal[:, 0]  # batch x n_mics x n_samples

    output_np = output_voices.detach().cpu().numpy()[0]
    energy = librosa.feature.rms(output_np).mean()

    return output_np, energy


def run_separation(mixed_data, model, args,
                   energy_cutoff=ENERGY_CUTOFF,
                   nms_cutoff=NMS_SIMILARITY_SDR): # yapf: disable
    """
    The main separation by localization algorithm
    """
    # Get the initial candidates
    num_windows = len(ALL_WINDOW_SIZES) if not args.moving else 3
    starting_angles = utils.get_starting_angles(ALL_WINDOW_SIZES[0])
    candidate_voices = [CandidateVoice(x, None, None) for x in starting_angles]

    # All steps of the binary search
    for window_idx in range(num_windows):
        if args.debug:
            print("---------")
        conditioning_label = torch.tensor(utils.to_categorical(
            window_idx, 5)).float().to(args.device).unsqueeze(0)

        curr_window_size = ALL_WINDOW_SIZES[window_idx]
        new_candidate_voices = []

        # Iterate over all the potential locations
        for voice in candidate_voices:
            output, energy = forward_pass(model, voice.angle, mixed_data,
                                          conditioning_label, args)

            if args.debug:
                print("Angle {:.2f} energy {}".format(voice.angle, energy))
                fname = "out{}_angle{:.2f}.wav".format(
                    window_idx, voice.angle * 180 / np.pi)

            # If there was something there
            if energy > energy_cutoff:

                # We're done searching so undo the shifts
                if window_idx == num_windows - 1:
                    target_pos = np.array([
                        FAR_FIELD_RADIUS * np.cos(voice.angle),
                        FAR_FIELD_RADIUS * np.sin(voice.angle)
                    ])
                    unshifted_output, _ = utils.shift_mixture(output,
                                                           target_pos,
                                                           .03231,
                                                           44100,
                                                           inverse=True)

                    new_candidate_voices.append(
                        CandidateVoice(voice.angle, energy, unshifted_output))

                else:
                    new_candidate_voices.append(
                        CandidateVoice(
                            voice.angle + curr_window_size / 4,
                            energy, output))
                    new_candidate_voices.append(
                        CandidateVoice(
                            voice.angle - curr_window_size / 4,
                            energy, output))

        candidate_voices = new_candidate_voices

    # Run NMS on the final output and return
    return nms(candidate_voices, nms_cutoff)


def main(args):
    device = torch.device('cuda')

    args.device = device
    model = CoSNetwork(n_audio_channels=4)
    model.load_state_dict(torch.load("./checkpoints/realdata_4mics_.03231m_44100kHz.pt"), strict=True)
    model.train = False
    model.to(device)
    
    #==============================================================================================================
    apply = False
    error = 0
    error_1 = []
    for path in tqdm(glob.glob(os.path.join("mix", "*"))):
        
        file_dir = re.sub(r"mix","mix_sep",path)
        
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        
        count = 0
    #==============================================================================================================
        basename_1 = os.path.basename(path)
        print(basename_1)
        #if basename_1 == str(3470):
        apply = True
        
        if apply == True:
    #==============================================================================================================
            for path_1 in glob.glob(os.path.join(path, "*.flac")):
                
                print("=================================================================")
                print("\033[92m"+path_1+"\033[0m")
                
                try:
                    mixed_data = librosa.core.load(path_1, mono=False, sr=44100)[0]
                    assert mixed_data.shape[0] == 4
            
                    temporal_chunk_size = int(44100 * 20.0)
                    num_chunks = (mixed_data.shape[1] // temporal_chunk_size) + 1

                    audio_data, sample_rate = sf.read(path_1)
                    audio_duration = len(audio_data) / sample_rate
                    print("音訊檔案的時間長度（秒）：", audio_duration)
                    print("=================================================================")
                    
                    
                    for chunk_idx in range(num_chunks):
                
                        curr_mixed_data = mixed_data[:, (chunk_idx *
                                                         temporal_chunk_size):(chunk_idx + 1) *
                                                         temporal_chunk_size]
    
                        output_voices = run_separation(curr_mixed_data, model, args)
                
                        count2 = 0
                
                        for voice in output_voices:
                            fname = str(count) + "_" + str(count2) + "_output_angle.wav"
                            sf.write(os.path.join(file_dir, fname), voice.data[0],
                                     44100)
                            count2 +=1
                            if count2 == 4: break
                    
                    count += 1
                    
                except Exception as e:
                    print("\033[91m"+f"Error processing {path_1}: {str(e)}"+"\033[0m")
                    error_1.append(path_1)
                    error += 1
                    
    print("error = " + str(error))
    print("error_name = " + str(error_1))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--debug',
                        action='store_true',
                        help="Save intermediate outputs")
    parser.add_argument('--moving',
                        action='store_true',
                        help="If the sources are moving then stop at a coarse window")
    main(parser.parse_args())
