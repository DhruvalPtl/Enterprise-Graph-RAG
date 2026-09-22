"""
Multi-Account Gemini API Key Pool Manager (Sequential Single-Account Mode).

Provides single-account execution: uses 1 account exclusively until its rate limit
or quota is reached, then automatically pauses that account and shifts all subsequent
requests to the next ready account in the pool.
"""
import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("key_pool")

DEFAULT_KEY_FILE = Path(__file__).resolve().parent.parent / "api_key.text"


@dataclass
class AccountInfo:
    account_id: int
    name: str
    api_key: str
    client: Any = None
    cooldown_until: float = 0.0
    requests_count: int = 0
    rate_limits_hit: int = 0
    is_disabled: bool = False
    disabled_reason: str = ""
    model_cooldowns: Dict[str, float] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        return (not self.is_disabled) and (time.time() >= self.cooldown_until)

    def is_model_ready(self, model_name: str) -> bool:
        """Returns True if account is enabled, overall cooldown expired, and specific model not in cooldown."""
        if not self.is_ready:
            return False
        return time.time() >= self.model_cooldowns.get(model_name, 0.0)

    def mark_model_cooldown(self, model_name: str, delay: float = 60.0):
        """Places a single model on cooldown for this account without necessarily blocking other models."""
        self.model_cooldowns[model_name] = time.time() + delay

    def get_ready_models(self, candidate_models: List[str]) -> List[str]:
        """Returns subset of candidate_models that are currently ready on this account."""
        return [m for m in candidate_models if self.is_model_ready(m)]

    @property
    def remaining_cooldown(self) -> float:
        if self.is_disabled:
            return float("inf")
        return max(0.0, self.cooldown_until - time.time())

    @property
    def masked_key(self) -> str:
        if len(self.api_key) <= 16:
            return self.api_key[:6] + "..."
        return self.api_key[:12] + "..." + self.api_key[-4:]


EXCLUDED_ACCOUNTS = {"plotveil", "dhruval0012", "account3", "account5"}


class KeyPoolManager:
    """
    Manages a pool of Gemini API accounts.
    Runs in sequential mode: uses one account at a time until its limit is reached,
    then automatically shifts to the next available account.
    """

    def __init__(self, key_file: Optional[Path] = None, api_keys: Optional[List[str]] = None):
        self._lock = threading.Lock()
        self._active_index = 0
        self.accounts: List[AccountInfo] = []
        self._initialized = False

        if key_file and Path(key_file).exists():
            self._load_from_file(Path(key_file))
        elif api_keys:
            self._load_from_list(api_keys)
        elif DEFAULT_KEY_FILE.exists():
            self._load_from_file(DEFAULT_KEY_FILE)
        else:
            env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if env_key:
                self._load_from_list([env_key])
            else:
                logger.warning("No API keys found in api_key.text or environment.")

    def _load_from_file(self, file_path: Path):
        from google import genai

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Ignore commented-out lines
            active_lines = [line for line in lines if not line.strip().startswith(("#", "//"))]
            text = "".join(active_lines)

            keys_map = dict(
                re.findall(
                    r"account(\d+)\s*=\s*\{\s*api_key\s*=\s*[\"']([^\"']+)[\"']",
                    text,
                )
            )
            names_map = dict(
                re.findall(
                    r"account(\d+)_name\s*=\s*([^\r\n]+)",
                    text,
                )
            )

            if not keys_map:
                raw_keys = re.findall(r"api_key\s*=\s*[\"']([^\"']+)[\"']", text)
                if not raw_keys:
                    raw_keys = re.findall(r"AQ\.[A-Za-z0-9_\-]+", text)
                keys_map = {str(i + 1): k for i, k in enumerate(raw_keys)}

            for str_id, key in sorted(keys_map.items(), key=lambda x: int(x[0])):
                acc_id = int(str_id)
                acc_name = names_map.get(str_id, f"account_{acc_id}").strip()

                # Explicitly skip excluded accounts (e.g. plotveil, dhruval0012)
                if acc_name.lower() in EXCLUDED_ACCOUNTS or f"account{acc_id}" in EXCLUDED_ACCOUNTS or f"account_{acc_id}" in EXCLUDED_ACCOUNTS:
                    logger.info(f"Skipping explicitly excluded account: '{acc_name}' (ID {acc_id})")
                    continue

                try:
                    client = genai.Client(api_key=key.strip())
                    self.accounts.append(
                        AccountInfo(
                            account_id=acc_id,
                            name=acc_name,
                            api_key=key.strip(),
                            client=client,
                        )
                    )
                except Exception as exc:
                    logger.error(f"Failed to initialize client for account {acc_name}: {exc}")

            self._initialized = len(self.accounts) > 0
            logger.info(f"Loaded {len(self.accounts)} API accounts from {file_path}")

        except Exception as exc:
            logger.error(f"Error loading key pool file {file_path}: {exc}")

    def _load_from_list(self, keys: List[str]):
        from google import genai

        for idx, key in enumerate(keys, start=1):
            if not key or not key.strip():
                continue
            client = genai.Client(api_key=key.strip())
            self.accounts.append(
                AccountInfo(
                    account_id=idx,
                    name=f"account_{idx}",
                    api_key=key.strip(),
                    client=client,
                )
            )
        self._initialized = len(self.accounts) > 0

    @property
    def total_accounts(self) -> int:
        return len(self.accounts)

    @property
    def active_account(self) -> Optional[AccountInfo]:
        if not self.accounts:
            return None
        return self.accounts[self._active_index % len(self.accounts)]

    def _shift_to_next_ready(self) -> bool:
        """Finds the next ready and enabled account. Returns True if found, False otherwise."""
        total = len(self.accounts)
        for i in range(1, total + 1):
            idx = (self._active_index + i) % total
            if self.accounts[idx].is_ready:
                self._active_index = idx
                return True
        return False

    def get_client_and_account(self) -> Tuple[Any, AccountInfo]:
        """
        Returns the current active account client.
        Keeps using this account until a rate limit or error causes a shift.
        If all accounts are in cooldown, pauses until the earliest becomes ready.
        """
        if not self.accounts:
            raise ValueError("No Gemini API accounts available in KeyPoolManager.")

        while True:
            with self._lock:
                now = time.time()
                curr_acc = self.accounts[self._active_index % len(self.accounts)]

                # 1. If the current active account is ready, use it!
                if curr_acc.is_ready:
                    curr_acc.requests_count += 1
                    return curr_acc.client, curr_acc

                # 2. Current account is not ready; try to shift to the next ready account
                if self._shift_to_next_ready():
                    new_acc = self.accounts[self._active_index]
                    print(
                        f"\n[KEY POOL] Active account shifted to '{new_acc.name}' ({new_acc.masked_key}).",
                        flush=True,
                    )
                    new_acc.requests_count += 1
                    return new_acc.client, new_acc

                # 3. All non-disabled accounts are in cooldown; find earliest reset
                non_disabled = [a for a in self.accounts if not a.is_disabled]
                if not non_disabled:
                    raise RuntimeError("All Gemini accounts in key pool are disabled/unauthorized.")

                earliest_acc = min(non_disabled, key=lambda a: a.cooldown_until)
                sleep_seconds = max(1.0, earliest_acc.cooldown_until - now + 0.5)
                should_log = (now - getattr(self, "_last_pause_log", 0.0)) > 3.0
                if should_log:
                    self._last_pause_log = now

            if should_log:
                print(
                    f"\n[KEY POOL] All available accounts currently on cooldown. "
                    f"Pausing for {sleep_seconds:.1f}s until '{earliest_acc.name}' resets...",
                    flush=True,
                )
            # Sleep in small slices so Ctrl+C / SIGINT is processed immediately on Windows
            end_sleep = time.time() + sleep_seconds
            while time.time() < end_sleep:
                time.sleep(min(0.5, max(0.05, end_sleep - time.time())))

    def mark_rate_limited(self, account_id: int, suggested_delay: Optional[float] = None):
        """
        Marks an account as rate-limited and shifts to the next ready account.
        """
        with self._lock:
            for acc in self.accounts:
                if acc.account_id == account_id:
                    delay = max(60.0, suggested_delay or 120.0)
                    acc.cooldown_until = time.time() + delay
                    acc.rate_limits_hit += 1
                    old_name = acc.name

                    # Advance active index to the next account
                    self._shift_to_next_ready()
                    new_acc = self.accounts[self._active_index]
                    print(
                        f"\n[ACCOUNT SHIFT] Account '{old_name}' reached rate limit (Cooldown #{acc.rate_limits_hit}, {delay:.0f}s). "
                        f"Switched active account to '{new_acc.name}' ({new_acc.masked_key}).",
                        flush=True,
                    )
                    break

    def mark_disabled(self, account_id: int, reason: str = "401 Unauthenticated"):
        """
        Permanently marks an account as disabled (e.g. 401 error) and shifts to next account.
        """
        with self._lock:
            for acc in self.accounts:
                if acc.account_id == account_id:
                    acc.is_disabled = True
                    acc.disabled_reason = reason
                    acc.cooldown_until = float("inf")
                    old_name = acc.name

                    self._shift_to_next_ready()
                    new_acc = self.accounts[self._active_index]
                    print(
                        f"\n[ACCOUNT DISABLED] Account '{old_name}' is disabled ({reason}). "
                        f"Skipping permanently. Active account is now '{new_acc.name}' ({new_acc.masked_key}).",
                        flush=True,
                    )
                    break

    def get_pool_status(self) -> List[Dict[str, Any]]:
        with self._lock:
            now = time.time()
            return [
                {
                    "account_id": a.account_id,
                    "name": a.name,
                    "masked_key": a.masked_key,
                    "is_active": (a.account_id == self.accounts[self._active_index].account_id),
                    "status": (
                        f"DISABLED ({a.disabled_reason})"
                        if a.is_disabled
                        else (
                            "ready"
                            if a.is_ready
                            else f"cooldown ({max(0.0, a.cooldown_until - now):.1f}s remaining)"
                        )
                    ),
                    "requests_served": a.requests_count,
                    "rate_limits_hit": a.rate_limits_hit,
                }
                for a in self.accounts
            ]


# Global singleton instance
_GLOBAL_KEY_POOL: Optional[KeyPoolManager] = None
_POOL_LOCK = threading.Lock()


def get_key_pool(key_file: Optional[Path] = None, force_reload: bool = False) -> KeyPoolManager:
    global _GLOBAL_KEY_POOL
    with _POOL_LOCK:
        if _GLOBAL_KEY_POOL is None or force_reload:
            _GLOBAL_KEY_POOL = KeyPoolManager(key_file=key_file)
        return _GLOBAL_KEY_POOL
