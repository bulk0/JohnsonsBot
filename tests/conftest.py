import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_update():
    update = MagicMock()
    update.effective_user.id = 12345
    update.message.text = "test message"
    return update

@pytest.fixture
def mock_context():
    context = MagicMock()
    context.user_data = {}
    return context
