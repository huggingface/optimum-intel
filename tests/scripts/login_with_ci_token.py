from pathlib import Path

from huggingface_hub import constants, get_token


# fmt: off
# not critical
CI_TOKEN = "".join(['h', 'f', '_', 'X', 'N', 'C', 'O', 'g', 'A', 'O', 'g', 'S', 'H', 'W', 'Y', 'B', 'H', 'g', 'L', 'J', 'n', 'C', 'U', 'f', 'i', 'O', 'D', 'n', 'm', 'N', 'n', 'V', 'Q', 'Q', 'N', 'y', 'c'])
# fmt: on

# save token directly to avoid whoami api call that causes rate limiting when many CI jobs run in parallel
token_path = Path(constants.HF_TOKEN_PATH)
token_path.parent.mkdir(parents=True, exist_ok=True)
token_path.write_text(CI_TOKEN)

assert get_token() == CI_TOKEN, f"Token was not saved correctly to {constants.HF_TOKEN_PATH}"
print(get_token(), CI_TOKEN)