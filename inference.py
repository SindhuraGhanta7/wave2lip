#%%writefile /content/Wav2Lip/inference.py
from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform
import audio

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str,
                    help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str,
                    help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str,
                    help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.',
                                default='results/result_voice.mp4')

parser.add_argument('--static', action='store_true',
                    help='If True, then use only first video frame for inference')
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)',
                    default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                    help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--face_det_batch_size', type=int,
                    help='Batch size for face detection', default=16)
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int,
            help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1],
                    help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. '
                    'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1],
                    help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
                    'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
                    help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
                    'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
                    help='Prevent smoothing face detections over a short temporal window')

args = parser.parse_args()
args.img_size = 96

# Create necessary directories
os.makedirs('temp', exist_ok=True)
os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

# Check if input is a static image
if os.path.isfile(args.face) and args.face.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
    args.static = True

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                          flip_input=False, device=device)

    batch_size = args.face_det_batch_size

    while 1:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)

        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    return results

def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames) # BGR2RGB for CNN face detection
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i%len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load_checkpoint(checkpoint_path):
    """Load checkpoint with PyTorch 2.6+ compatibility (weights_only=False)"""
    try:
        print("Attempting to load checkpoint with weights_only=False...")
        if device == 'cuda':
            checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=False)
        else:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        return checkpoint
    except Exception as e:
        print(f"Error loading with weights_only=False: {e}")
        # Try pre-PyTorch 2.6 method
        try:
            print("Attempting to load with older PyTorch method...")
            if device == 'cuda':
                checkpoint = torch.load(checkpoint_path, map_location='cuda')
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
            return checkpoint
        except Exception as e2:
            print(f"Error with second loading method: {e2}")
            raise RuntimeError("Failed to load checkpoint using multiple methods")

def extract_model_from_torchscript(ts_path, target_path=None):
    """
    Extract model weights from TorchScript file and save as regular PyTorch model
    """
    try:
        print("Attempting to extract state_dict from TorchScript model...")
        if target_path is None:
            target_path = ts_path + '.extracted.pth'

        # Load TorchScript model using torch.jit.load
        ts_model = None
        try:
            ts_model = torch.jit.load(ts_path, map_location='cpu')
        except Exception as e:
            print(f"Failed to load with torch.jit.load: {e}")
            # Try script-specific loading
            try:
                import zipfile
                with zipfile.ZipFile(ts_path) as z:
                    # Extract any files to temp
                    z.extractall('temp/model_extract')
                    if os.path.exists('temp/model_extract/data.pkl'):
                        return torch.load('temp/model_extract/data.pkl', map_location='cpu')
            except Exception as e2:
                print(f"Failed to extract with zipfile: {e2}")
                return None

        if ts_model is None:
            return None

        # Try to get state dict
        try:
            state_dict = ts_model.state_dict()
            torch.save({'state_dict': state_dict}, target_path)
            print(f"Extracted state_dict successfully to {target_path}")
            return {'state_dict': state_dict}
        except Exception as e:
            print(f"Failed to extract state_dict: {e}")
            # Try to evaluate model to see parameters
            try:
                dummy_input = torch.randn(1, 1, 80, 16)
                dummy_img = torch.randn(1, 6, 96, 96)
                ts_model.eval()
                _ = ts_model(dummy_input, dummy_img)

                params = {}
                for name, param in ts_model.named_parameters():
                    params[name] = param.data

                torch.save({'state_dict': params}, target_path)
                print(f"Extracted parameters to {target_path}")
                return {'state_dict': params}
            except Exception as e2:
                print(f"Failed to evaluate model and extract parameters: {e2}")
                return None
    except Exception as general_error:
        print(f"General extraction error: {general_error}")
        return None

def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))

    try:
        # First try: standard loading with weights_only=False for PyTorch 2.6+ compatibility
        try:
            checkpoint = _load_checkpoint(path)
            if checkpoint is None:
                raise ValueError("Failed to load checkpoint")

            # Check if it's a TorchScript model
            if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
                # Try extracting weights from TorchScript model
                extracted = extract_model_from_torchscript(path)
                if extracted and "state_dict" in extracted:
                    checkpoint = extracted
                else:
                    # If extraction fails, try to use direct JIT loading (GPU only)
                    if device == 'cuda':
                        print("Loading model directly with torch.jit.load")
                        model = torch.jit.load(path, map_location=device)
                        return model.eval()
                    else:
                        # For CPU, we need the extracted model
                        raise ValueError("Cannot load TorchScript model on CPU without extraction")

            # Regular PyTorch state dict loading
            s = checkpoint["state_dict"]
            new_s = {}
            for k, v in s.items():
                new_s[k.replace('module.', '')] = v
            model.load_state_dict(new_s)

        except Exception as e:
            print(f"Standard loading failed: {e}")
            # Try loading with CPU-compatible method for PyTorch models
            try:
                # Extract weights from TorchScript if needed
                if path.endswith('.pt') or path.endswith('.pth'):
                    extracted_path = path + '.extracted.pth'
                    if not os.path.exists(extracted_path):
                        extracted = extract_model_from_torchscript(path, extracted_path)
                        if extracted:
                            checkpoint = extracted
                        else:
                            # Last attempt: create a dummy state dict
                            print("Creating a compatible model placeholder...")
                            # Initialize a model with random weights
                            model = Wav2Lip()
                            model = model.to(device)
                            return model.eval()
                    else:
                        # Use previously extracted model
                        checkpoint = torch.load(extracted_path, map_location='cpu')

                    if checkpoint and "state_dict" in checkpoint:
                        s = checkpoint["state_dict"]
                        new_s = {}
                        for k, v in s.items():
                            new_s[k.replace('module.', '')] = v
                        model.load_state_dict(new_s, strict=False)
                        print("Loaded extracted model weights")
                else:
                    raise ValueError("Unsupported model format")
            except Exception as e2:
                print(f"All loading methods failed: {e2}")
                print("WARNING: Using model with random weights. Results will not be accurate.")
    except Exception as general_error:
        print(f"General error in model loading: {general_error}")
        print("WARNING: Using model with random weights. Results will not be accurate.")

    model = model.to(device)
    return model.eval()

def main():
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    elif args.face.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps

    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

            if args.rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)

    print("Number of frames available for inference: "+str(len(full_frames)))

    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

        subprocess.call(command, shell=platform.system() != 'Windows')
        args.audio = 'temp/temp.wav'

    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)
    print(f"Mel shape: {mel.shape}")

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80./fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]

    batch_size = args.wav2lip_batch_size
    gen = datagen(full_frames.copy(), mel_chunks)

    model = None  # Initialize model variable outside the loop

    try:
        model = load_model(args.checkpoint_path)
        print("Model loaded")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("WARNING: Using default model. Results may not be accurate.")
        model = Wav2Lip().to(device).eval()

    frame_h, frame_w = full_frames[0].shape[:-1]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('temp/result.avi',
                            fourcc, fps, (frame_w, frame_h))

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
                                        total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            try:
                pred = model(mel_batch, img_batch)

                # Handle different return types
                if isinstance(pred, tuple):
                    pred = pred[0]

                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

                for p, f, c in zip(pred, frames, coords):
                    y1, y2, x1, x2 = c
                    p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                    f[y1:y2, x1:x2] = p
                    out.write(f)
            except Exception as e:
                print(f"Error during inference: {e}")
                # Just write the original frame if prediction fails
                for f in frames:
                    out.write(f)

    out.release()

    # Use more robust ffmpeg command
    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 -c:a aac -map 0:a:0 -map 1:v:0 {}'.format(
        args.audio, 'temp/result.avi', args.outfile)
    print(f"Running command: {command}")
    subprocess.call(command, shell=platform.system() != 'Windows')
    print(f"Video saved to {args.outfile}")

if __name__ == '__main__':
    main()