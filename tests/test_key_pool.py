"""
Unit tests for KeyPoolManager (Sequential Single-Account Mode).
"""
import time
from unittest.mock import MagicMock
from app.key_pool import KeyPoolManager, AccountInfo, DEFAULT_KEY_FILE


def test_key_pool_loading_from_file():
    if not DEFAULT_KEY_FILE.exists():
        return
    pool = KeyPoolManager(key_file=DEFAULT_KEY_FILE)
    assert pool.total_accounts == 4, f"Expected 4 active accounts, got {pool.total_accounts}"
    names = [a.name for a in pool.accounts]
    assert "dhruval00" in names
    assert "dspatel" in names


def test_key_pool_single_account_sticky():
    """Verify that requests stick to 1 account as long as it's ready."""
    pool = KeyPoolManager()
    acc1 = AccountInfo(account_id=1, name="acc1", api_key="key1", client=MagicMock())
    acc2 = AccountInfo(account_id=2, name="acc2", api_key="key2", client=MagicMock())
    pool.accounts = [acc1, acc2]

    # Call multiple times: must ALL be acc1
    for i in range(1, 5):
        c, a = pool.get_client_and_account()
        assert a.name == "acc1"
        assert a.requests_count == i


def test_key_pool_shift_on_rate_limit():
    """Verify that when account 1 hits rate limit, it shifts to account 2 and stays on account 2."""
    pool = KeyPoolManager()
    acc1 = AccountInfo(account_id=1, name="acc1", api_key="key1", client=MagicMock())
    acc2 = AccountInfo(account_id=2, name="acc2", api_key="key2", client=MagicMock())
    acc3 = AccountInfo(account_id=3, name="acc3", api_key="key3", client=MagicMock())
    pool.accounts = [acc1, acc2, acc3]

    # Starts on acc1
    _, a = pool.get_client_and_account()
    assert a.name == "acc1"

    # acc1 hits rate limit
    pool.mark_rate_limited(account_id=1, suggested_delay=60.0)

    # Next call should be acc2
    _, a = pool.get_client_and_account()
    assert a.name == "acc2"

    # Subsequent call MUST still be acc2 (sticky)
    _, a = pool.get_client_and_account()
    assert a.name == "acc2"


def test_key_pool_skip_disabled_account():
    """Verify that disabled accounts (e.g. 401) are permanently skipped."""
    pool = KeyPoolManager()
    acc1 = AccountInfo(account_id=1, name="acc1", api_key="key1", client=MagicMock())
    acc2 = AccountInfo(account_id=2, name="acc2", api_key="key2", client=MagicMock(), is_disabled=True)
    acc3 = AccountInfo(account_id=3, name="acc3", api_key="key3", client=MagicMock())
    pool.accounts = [acc1, acc2, acc3]

    # Starts on acc1
    _, a = pool.get_client_and_account()
    assert a.name == "acc1"

    # acc1 hits rate limit -> should skip disabled acc2 and pick acc3!
    pool.mark_rate_limited(account_id=1, suggested_delay=60.0)
    _, a = pool.get_client_and_account()
    assert a.name == "acc3"
