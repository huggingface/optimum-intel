from pathlib import Path

from huggingface_hub import constants
from huggingface_hub.utils._auth import _write_secret


# fmt: off
# not critical
CI_TOKEN = "".join(['h', 'f', '_', 'X', 'N', 'C', 'O', 'g', 'A', 'O', 'g', 'S', 'H', 'W', 'Y', 'B', 'H', 'g', 'L', 'J', 'n', 'C', 'U', 'f', 'i', 'O', 'D', 'n', 'm', 'N', 'n', 'V', 'Q', 'Q', 'N', 'y', 'c'])
# fmt: on

# save token directly to avoid whoami api call that causes rate limiting when many CI jobs run in parallel
_write_secret(Path(constants.HF_TOKEN_PATH), CI_TOKEN)
