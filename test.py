import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer, util
from faster_whisper import WhisperModel
from datasets import load_dataset
import timeit
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from tqdm.auto import tqdm

# Constants and initial setup
MODEL_IDS = {
    "large": "distil-whisper/distil-large-v2",
    "medium": "distil-whisper/distil-medium.en",
    "small": "distil-whisper/distil-small.en"
}
MODEL_IDS_FOR_FASTER_WHISPER = {
    "large": "distil-large-v2",
    "medium": "distil-medium.en",
    "small": "distil-small.en"
}

SPLIT = "validation"
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')


def setup_mps():
    """Check if MPS is available and set the appropriate device and data type."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32
    return device, torch_dtype

def initialize_model_and_pipeline(model_id, device, torch_dtype):
    """Initialize the model and pipeline."""
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
    model.to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    asr_pipeline = pipeline("automatic-speech-recognition", model=model, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, max_new_tokens=128, torch_dtype=torch_dtype, device=device)
    return asr_pipeline

def transcribe_with_transformers(pipeline, sample):
    """Transcribe audio sample with Transformers pipeline."""
    start_time = timeit.default_timer()
    result = pipeline(sample)
    full_transcription = result["text"]
    elapsed_time = timeit.default_timer() - start_time
    return full_transcription, elapsed_time

def load_and_prepare_dataset():
    DATASET_NAME = "hf-internal-testing/librispeech_asr_dummy"
    dataset = load_dataset(DATASET_NAME, "clean", split=SPLIT, trust_remote_code=True)
    sample = dataset[0]['audio']
    ground_truth = dataset[0]['text']
    return sample, ground_truth

def transcribe_and_time(model_id, sample_path, device="cpu", compute_type="int8"):
    start_time = timeit.default_timer()
    model = WhisperModel(model_id, device=device, compute_type=compute_type)
    segments, _ = model.transcribe(sample_path, word_timestamps=True)
    full_transcription = " ".join(segment.text for segment in segments)
    elapsed_time = timeit.default_timer() - start_time
    return full_transcription, elapsed_time

def calculate_semantic_similarity(ground_truth, transcription):
    ground_truth_embedding = sentence_model.encode(ground_truth, convert_to_tensor=True)
    transcription_embedding = sentence_model.encode(transcription, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(ground_truth_embedding, transcription_embedding)
    return similarity.item()

def visualize_comparisons(models, model_types, times, similarities):
    n_groups = len(models)
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    # Time comparison
    ax[0].bar(index, times[::2], bar_width, alpha=opacity, color='b', label='Faster Whisper')
    ax[0].bar(index + bar_width, times[1::2], bar_width, alpha=opacity, color='r', label='Transformers')
    ax[0].set_ylabel('Time (s)')
    ax[0].set_title('Speed Comparison by Model Size')
    ax[0].set_xticks(index + bar_width / 2)
    ax[0].set_xticklabels(models)
    ax[0].legend()

    # Accuracy comparison with adjusted y-axis and annotations
    ax[1].bar(index, similarities[::2], bar_width, alpha=opacity, color='b', label='Faster Whisper')
    ax[1].bar(index + bar_width, similarities[1::2], bar_width, alpha=opacity, color='r', label='Transformers')
    ax[1].set_ylabel('Semantic Similarity')
    ax[1].set_title('Accuracy Comparison by Model Size')
    ax[1].set_ylim(0.95, 1)  
    vertical_offset = 0.005  

    for i, sim in enumerate(similarities):
        ax[1].text(i % len(models) + (bar_width / 2 if i % 2 else 0), sim + vertical_offset, f"{sim:.4f}", 
                ha='center', va='bottom')  
    ax[1].set_xticks(index + bar_width / 2)
    ax[1].set_xticklabels(models)
    ax[1].legend()

    fig.tight_layout()
    plt.savefig('output.png')


def evaluate():
    original_sample, ground_truth = load_and_prepare_dataset()
    sample_path = original_sample['path']

    device, torch_dtype = setup_mps()

    times = []
    similarities = []
    model_types = ["Faster Whisper", "Transformers"]
    models = list(MODEL_IDS.keys())

    for model_type in model_types:
        for model_name in models:
            print(f"Processing with {model_name} model using {model_type}...")
            if model_type == "Faster Whisper":
                # For Faster Whisper, use the adjusted model name and sample path
                transcription, elapsed_time = transcribe_and_time(MODEL_IDS_FOR_FASTER_WHISPER[model_name], sample_path, "cpu", "int8")
            else:  # Transformers
                # Reconstruct the sample dictionary for Transformers to ensure it's not modified
                sample_for_transformers = {
                    "path": sample_path,
                    "array": original_sample['array'],
                    "sampling_rate": original_sample['sampling_rate']
                }
                asr_pipeline = initialize_model_and_pipeline(MODEL_IDS[model_name], device, torch_dtype)
                transcription, elapsed_time = transcribe_with_transformers(asr_pipeline, sample_for_transformers)
            
            similarity = calculate_semantic_similarity(ground_truth, transcription)
            
            times.append(elapsed_time)
            similarities.append(similarity)
            print(f"{model_name} ({model_type}) - Time: {elapsed_time:.2f}s, Similarity: {similarity:.4f}")
            
    # Visualization
    visualize_comparisons(models, model_types, times, similarities)
    

import numpy as np

def evaluate_2():
    DATASET_NAME = "mozilla-foundation/common_voice_16_1"
    SUBSET = "en"
    SPLIT = "validation"
    
    # Define the models to evaluate
    models_info = {
        'Faster Whisper Small': {'id': MODEL_IDS_FOR_FASTER_WHISPER['small'], 'type': 'fw'},
        'Tiny Model': {'id': 'tiny.en', 'type': 'fw'}, 
        'Transformers Small': {'id': MODEL_IDS['small'], 'type': 'tf'}
    }
    
    # Initialize device and torch_dtype
    device, torch_dtype = setup_mps()
    
    # Initialize metrics storage
    metrics = {name: {'time': [], 'similarity': []} for name in models_info}
    
    # Loop through models
    for model_name, model_info in models_info.items():
        print(f"Evaluating {model_name}...")
        
        # Load dataset streamingly
        dataset = load_dataset(DATASET_NAME, SUBSET, split=SPLIT, use_auth_token=True, trust_remote_code=True)
        
        if model_info['type'] == 'tf':
            asr_pipeline = initialize_model_and_pipeline(model_info['id'], device, torch_dtype)
        
        # Loop through samples in the dataset
        for sample in tqdm(dataset, desc=f"Evaluating {model_name}"):
            sample_path = sample['path'] if 'path' in sample else None
            
            if model_info['type'] == 'fw':
                transcription, elapsed_time = transcribe_and_time(model_info['id'], sample_path, device, torch_dtype)
            elif model_info['type'] == 'tf':
                # Make sure sample['audio'] is in the correct format for transformers
                transcription, elapsed_time = transcribe_with_transformers(asr_pipeline, {"array": sample['audio']['array'], "sampling_rate": sample['audio']['sampling_rate']})
            
            similarity = calculate_semantic_similarity(sample['sentence'], transcription)
            
            # Store metrics
            metrics[model_name]['time'].append(elapsed_time)
            metrics[model_name]['similarity'].append(similarity)
        
        # Compute averages for the current model
        avg_time = np.mean(metrics[model_name]['time'])
        avg_similarity = np.mean(metrics[model_name]['similarity'])
        print(f"{model_name} - Average Time: {avg_time:.2f}s, Average Similarity: {avg_similarity:.4f}")
    
    # Plot results
    plot_metrics_2(models_info.keys(), metrics)

    
def plot_metrics_2(labels, metrics):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Time
    avg_times = [np.mean(metrics[label]['time']) for label in labels]
    axs[0].bar(labels, avg_times, color=['blue', 'green', 'red'])
    axs[0].set_title('Average Time (s)')
    axs[0].set_ylabel('Seconds')
    
    # Similarity
    avg_similarities = [np.mean(metrics[label]['similarity']) for label in labels]
    axs[1].bar(labels, avg_similarities, color=['blue', 'green', 'red'])
    axs[1].set_title('Average Semantic Similarity')
    axs[1].set_ylabel('Similarity Score')
    
    for ax in axs:
        ax.set_xticklabels(labels, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
def main():
    #evaluate()
    evaluate_2()
    print("Finished!")

if __name__ == "__main__":
    main()