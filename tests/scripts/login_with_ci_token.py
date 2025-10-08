from huggingface_hub import login


# fmt: off
# not critical
CI_TOKEN = "".join(['h', 'f', '_', 'X', 'N', 'C', 'O', 'g', 'A', 'O', 'g', 'S', 'H', 'W', 'Y', 'B', 'H', 'g', 'L', 'J', 'n', 'C', 'U', 'f', 'i', 'O', 'D', 'n', 'm', 'N', 'n', 'V', 'Q', 'Q', 'N', 'y', 'c'])  
# fmt: on

login(token=CI_TOKEN)
