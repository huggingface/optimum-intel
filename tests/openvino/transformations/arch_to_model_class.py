
# Used by test_transformations.py to dynamically select the right class per architecture

ARCH_TO_MODEL_CLASS = {
    # text-generation
    "afmoe":                    "OVModelForCausalLM",
    "gpt2":                     "OVModelForCausalLM",
    "llama":                    "OVModelForCausalLM",
    "mistral":                  "OVModelForCausalLM",
    "qwen2":                    "OVModelForCausalLM",
    "qwen3":                    "OVModelForCausalLM",
    

    # image-text-to-text
    "llava":                    "OVModelForVisualCausalLM",
    

    # text-to-image / text-to-video
    "stable-diffusion":      "OVDiffusionPipeline",
    

    # automatic-speech-recognition
    "whisper":          "OVModelForSpeechSeq2Seq",
  

    # text2text-generation
   
    "bart":             "OVModelForSeq2SeqLM",
    



    # feature-extraction 
    "bert":             "OVModelForFeatureExtraction",
    "electra":          "OVModelForFeatureExtraction",
 

   

    # zero-shot-image-classification
    "clip":             "OVModelForZeroShotImageClassification",
    "siglip":           "OVModelForZeroShotImageClassification",
    
}


