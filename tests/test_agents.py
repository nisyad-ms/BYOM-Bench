"""Tests for agent lifecycle: MemoryAgent build_context, cleanup, and state."""

from unittest.mock import patch

from tests.conftest import make_multisession_output


class TestMemoryAgentBuildContext:
    def test_memory_agent_build_context_calls_populate(self, mock_memory_store):
        """build_context calls store.populate exactly once."""
        with patch("ream_bench.agents.memory_agent.PooledLLMClient"):
            from ream_bench.agents.memory_agent import MemoryAgent

            agent = MemoryAgent(mock_memory_store)
            mso = make_multisession_output()
            agent.build_context(mso)

            mock_memory_store.populate.assert_called_once_with(mso)

    def test_memory_agent_build_context_idempotent(self, mock_memory_store):
        """Calling build_context twice only calls populate once."""
        with patch("ream_bench.agents.memory_agent.PooledLLMClient"):
            from ream_bench.agents.memory_agent import MemoryAgent

            agent = MemoryAgent(mock_memory_store)
            mso = make_multisession_output()
            agent.build_context(mso)
            agent.build_context(mso)

            mock_memory_store.populate.assert_called_once()


class TestMemoryAgentCleanup:
    def test_memory_agent_cleanup_forwards_to_store(self, mock_memory_store):
        """cleanup delegates to store.cleanup."""
        from ream_bench.agents.memory_agent import MemoryAgent

        agent = MemoryAgent(mock_memory_store)
        agent.cleanup()

        mock_memory_store.cleanup.assert_called_once()

    def test_memory_agent_cleanup_resets_state(self, mock_memory_store):
        """After cleanup, _memory_populated is False so build_context repopulates."""
        with patch("ream_bench.agents.memory_agent.PooledLLMClient"):
            from ream_bench.agents.memory_agent import MemoryAgent

            agent = MemoryAgent(mock_memory_store)
            mso = make_multisession_output()
            agent.build_context(mso)
            assert agent._memory_populated is True

            agent.cleanup()
            assert agent._memory_populated is False
