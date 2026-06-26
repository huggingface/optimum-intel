#  Copyright 2026 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, constants, scan_cache_dir, snapshot_download


sys.path.insert(0, str(Path(__file__).parents[1] / "openvino"))

from utils_tests import MODEL_NAMES


# github actions repo cache limit is 10 GB so using 9 GB
CACHE_LIMIT = 9 * 10**9
# exclude model larger than 200 MB
MODEL_SIZE_THRESHOLD = 200 * 10**6


def hub_size(api: HfApi, repo_id: str) -> int:
    try:
        info = api.model_info(repo_id, files_metadata=True)
        return sum(sib.size or 0 for sib in (info.siblings or []))
    except Exception as e:
        print(f"Size query failed for {repo_id}: {e}")
        return -1


def main() -> None:

    api = HfApi()
    candidates = {}
    for name, repo_id in MODEL_NAMES.items():
        if not isinstance(repo_id, str) or not repo_id or repo_id.startswith("/"):
            continue
        candidates[repo_id] = name

    if os.path.exists(constants.HF_HUB_CACHE):
        cache_info = scan_cache_dir()
        cached_repos = {repo.repo_id for repo in cache_info.repos}
        sum_repo_size = cache_info.size_on_disk
    else:
        cached_repos = set()
        sum_repo_size = 0

    downloaded = []
    skipped_too_large = []
    skipped_cache_full = []
    failed = []
    force_update = os.environ.get("FORCE_DOWNLOAD", "").lower() in ("1", "true", "yes")
    for repo_id, name in candidates.items():
        if not force_update and repo_id in cached_repos:
            continue

        size = hub_size(api, repo_id)
        if size < 0:
            failed.append(name)
            continue

        if size > MODEL_SIZE_THRESHOLD:
            skipped_too_large.append(name)
            continue

        if sum_repo_size + size > CACHE_LIMIT:
            skipped_cache_full.append(name)
            continue

        try:
            snapshot_download(repo_id)
            sum_repo_size += size
            downloaded.append(name)
        except Exception as e:
            print(f"[FAIL]  {name}: {e}", file=sys.stderr)
            failed.append(repo_id)

    print(f"Downloaded          : {len(downloaded)}  ({', '.join(downloaded)})")
    print(f"Skipped (too large) : {len(skipped_too_large)}  ({', '.join(skipped_too_large)})")
    print(f"Skipped (cache full): {len(skipped_cache_full)}  ({', '.join(skipped_cache_full)})")
    print(f"Loading failed      : {len(failed)}  ({', '.join(failed)})")
    final_cache_info = scan_cache_dir()
    actual_size = final_cache_info.size_on_disk / 10**9
    print(f"Actual cache        : {actual_size:.2f} GB, sum_repo_size {sum_repo_size / 10**9:.0f} GB")

    final_cached_repos = {repo.repo_id for repo in final_cache_info.repos}
    name_of = {repo_id: name for name, repo_id in MODEL_NAMES.items()}
    downloaded_repo_ids = {repo_id for name, repo_id in MODEL_NAMES.items() if name in set(downloaded)}

    in_cache_not_downloaded = final_cached_repos - downloaded_repo_ids
    in_downloaded_not_in_cache = downloaded_repo_ids - final_cached_repos

    print(
        f"In cache but not in downloaded ({len(in_cache_not_downloaded)}): "
        f"{', '.join(name_of.get(r, r) for r in sorted(in_cache_not_downloaded))}"
    )
    print(
        f"In downloaded but not in cache ({len(in_downloaded_not_in_cache)}): "
        f"{', '.join(name_of.get(r, r) for r in sorted(in_downloaded_not_in_cache))}"
    )


if __name__ == "__main__":
    main()
