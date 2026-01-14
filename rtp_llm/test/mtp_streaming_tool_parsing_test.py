"""
MTP-Safe Streaming Tool Call Parsing Tests

Tests for tool call parsing under MTP (Speculative Decoding) conditions where
multiple tokens may arrive in a single chunk, including scenarios where:
1. Complete tool call blocks arrive in single chunk
2. Think-end tag and tool-start tag arrive in same chunk
3. Multiple complete tool calls arrive in single chunk
"""

import unittest

from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import (
    Function,
    Tool,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.deepseekv31_detector import (
    DeepSeekV31Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.kimik2_detector import (
    KimiK2Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.qwen25_detector import (
    Qwen25Detector,
)


def create_tools():
    """Create test tool definitions."""
    return [
        Tool(
            type="function",
            function=Function(
                name="get_current_weather",
                description="Get the current weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "The city name"},
                    },
                    "required": ["location"],
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="get_time",
                description="Get current time",
                parameters={"type": "object", "properties": {}},
            ),
        ),
    ]


class TestQwen25DetectorMTP(unittest.TestCase):
    """Test Qwen25Detector MTP compatibility."""

    def setUp(self):
        self.detector = Qwen25Detector()
        self.tools = create_tools()

    def test_mtp_complete_tool_call_single_chunk(self):
        """
        MTP scenario: Complete tool call block arrives in single chunk.
        This simulates MTP returning the entire tool call at once instead of
        token-by-token.
        """
        # Complete tool call in one chunk
        chunk = '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "杭州"}}\n</tool_call>'
        result = self.detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(
            len(result.calls),
            1,
            f"Expected 1 call, got {len(result.calls)}. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{result.calls[0].name}'. Calls: {result.calls}",
        )
        self.assertIn(
            '"location"',
            result.calls[0].parameters,
            f"Expected '\"location\"' in parameters. Calls: {result.calls}",
        )
        self.assertIn(
            "杭州",
            result.calls[0].parameters,
            f"Expected '杭州' in parameters. Calls: {result.calls}",
        )

    def test_mtp_think_end_and_tool_start_same_chunk(self):
        """
        MTP scenario: Think-end tag and tool-start tag arrive in same chunk.
        This is the most common MTP failure case.
        """
        self.detector = Qwen25Detector()

        # First chunk: reasoning content
        chunk1 = "I need to check the weather"
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertEqual(
            result1.normal_text,
            "I need to check the weather",
            f"Expected normal_text 'I need to check the weather', got '{result1.normal_text}'. Calls: {result1.calls}",
        )
        self.assertEqual(
            len(result1.calls),
            0,
            f"Expected 0 calls, got {len(result1.calls)}. Calls: {result1.calls}",
        )

        # MTP chunk: newlines followed by complete tool call
        # Simulates </think>\n\n<tool_call>... in one chunk
        chunk2 = '\n\n<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "杭州"}}\n</tool_call>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(
            len(result2.calls),
            1,
            f"Expected 1 call, got {len(result2.calls)}. Calls: {result2.calls}",
        )
        self.assertEqual(
            result2.calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{result2.calls[0].name}'. Calls: {result2.calls}",
        )

    def test_mtp_multiple_tool_calls_single_chunk(self):
        """
        MTP scenario: Multiple complete tool calls arrive in single chunk.
        """
        # Two complete tool calls in one chunk
        chunk = (
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "杭州"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_current_weather", "arguments": {"location": "北京"}}\n</tool_call>'
        )
        result = self.detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(
            len(result.calls),
            2,
            f"Expected 2 calls, got {len(result.calls)}. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[0].name,
            "get_current_weather",
            f"Expected calls[0].name 'get_current_weather', got '{result.calls[0].name}'. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[1].name,
            "get_current_weather",
            f"Expected calls[1].name 'get_current_weather', got '{result.calls[1].name}'. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[0].tool_index,
            0,
            f"Expected calls[0].tool_index 0, got {result.calls[0].tool_index}. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[1].tool_index,
            1,
            f"Expected calls[1].tool_index 1, got {result.calls[1].tool_index}. Calls: {result.calls}",
        )

    def test_mtp_partial_then_complete(self):
        """
        MTP scenario: Partial tool call followed by completion in next chunk.
        """
        # First chunk: start of tool call
        chunk1 = '<tool_call>\n{"name": "get_current_weather"'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        # Should not have complete call yet
        print(f"result1.calls: {result1.calls}")

        # Second chunk: completion of tool call (MTP style - multiple tokens)
        chunk2 = ', "arguments": {"location": "杭州"}}\n</tool_call>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        # The complete call should be returned
        self.assertEqual(
            len(result2.calls),
            1,
            f"Expected 1 call, got {len(result2.calls)}. Calls: {result2.calls}",
        )
        self.assertEqual(
            result2.calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{result2.calls[0].name}'. Calls: {result2.calls}",
        )

    def test_incremental_still_works(self):
        """
        Verify that traditional single-token incremental streaming still works.
        """
        detector = Qwen25Detector()

        # Simulate token-by-token streaming
        chunks = [
            "<tool_call>",
            "\n",
            "{",
            '"name": "get_current_weather"',
            ', "arguments": {"location": "杭州"}',
            "}",
            "\n</tool_call>",
        ]

        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        # Should have parsed the tool call
        print(f"all_calls: {all_calls}")
        self.assertGreaterEqual(
            len(all_calls),
            1,
            f"Should have at least 1 call, got {len(all_calls)}. Calls: {all_calls}",
        )
        # Find the call with name set (first chunk sends name)
        named_calls = [c for c in all_calls if c.name]
        self.assertTrue(
            len(named_calls) > 0,
            f"Should have a call with name. All calls: {all_calls}",
        )
        self.assertEqual(
            named_calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{named_calls[0].name}'. Named calls: {named_calls}",
        )


class TestKimiK2DetectorMTP(unittest.TestCase):
    """Test KimiK2Detector MTP compatibility."""

    def setUp(self):
        self.detector = KimiK2Detector()
        self.tools = create_tools()

    def test_mtp_complete_tool_call_single_chunk(self):
        """
        MTP scenario: Complete KimiK2 tool call block arrives in single chunk.
        """
        chunk = '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_weather:0 <|tool_call_argument_begin|>{"location": "杭州"}<|tool_call_end|><|tool_calls_section_end|>'
        result = self.detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(
            len(result.calls),
            1,
            f"Expected 1 call, got {len(result.calls)}. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{result.calls[0].name}'. Calls: {result.calls}",
        )
        self.assertIn(
            "杭州",
            result.calls[0].parameters,
            f"Expected '杭州' in parameters. Calls: {result.calls}",
        )

    def test_mtp_multiple_tool_calls_single_chunk(self):
        """
        MTP scenario: Multiple complete KimiK2 tool calls in single chunk.
        """
        chunk = (
            "<|tool_calls_section_begin|>"
            '<|tool_call_begin|>functions.get_current_weather:0 <|tool_call_argument_begin|>{"location": "杭州"}<|tool_call_end|>'
            '<|tool_call_begin|>functions.get_current_weather:1 <|tool_call_argument_begin|>{"location": "北京"}<|tool_call_end|>'
            "<|tool_calls_section_end|>"
        )
        result = self.detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(
            len(result.calls),
            2,
            f"Expected 2 calls, got {len(result.calls)}. Calls: {result.calls}",
        )

    def test_mtp_partial_then_complete(self):
        """
        MTP scenario: Partial tool call followed by completion.
        """
        # First chunk: start of tool call
        chunk1 = '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_weather:0 <|tool_call_argument_begin|>{"location"'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        print(f"result1.calls: {result1.calls}")

        # Second chunk: completion (MTP style)
        chunk2 = ': "杭州"}<|tool_call_end|><|tool_calls_section_end|>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(
            len(result2.calls),
            1,
            f"Expected 1 call, got {len(result2.calls)}. Calls: {result2.calls}",
        )


class TestDeepSeekV31DetectorMTP(unittest.TestCase):
    """Test DeepSeekV31Detector MTP compatibility."""

    def setUp(self):
        self.detector = DeepSeekV31Detector()
        self.tools = create_tools()

    def test_mtp_complete_tool_call_single_chunk(self):
        """
        MTP scenario: Complete DeepSeek tool call block arrives in single chunk.
        """
        chunk = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "杭州"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        result = self.detector.parse_streaming_increment(chunk, self.tools)

        print(f"result.calls: {result.calls}")
        self.assertEqual(
            len(result.calls),
            1,
            f"Expected 1 call, got {len(result.calls)}. Calls: {result.calls}",
        )
        self.assertEqual(
            result.calls[0].name,
            "get_current_weather",
            f"Expected name 'get_current_weather', got '{result.calls[0].name}'. Calls: {result.calls}",
        )
        self.assertIn(
            "杭州",
            result.calls[0].parameters,
            f"Expected '杭州' in parameters. Calls: {result.calls}",
        )

    def test_mtp_multiple_tool_calls_single_chunk(self):
        """
        MTP scenario: Multiple complete DeepSeek tool calls in single chunk.
        """
        chunk = (
            "<｜tool▁calls▁begin｜>"
            '<｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "杭州"}<｜tool▁call▁end｜>'
            '<｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "北京"}<｜tool▁call▁end｜>'
            "<｜tool▁calls▁end｜>"
        )
        result = self.detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(
            len(result.calls),
            2,
            f"Expected 2 calls, got {len(result.calls)}. Calls: {result.calls}",
        )

    def test_mtp_partial_then_complete(self):
        """
        MTP scenario: Partial tool call followed by completion.
        """
        # First chunk: start of tool call
        chunk1 = '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location"'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        print(f"result1.calls: {result1.calls}")

        # Second chunk: completion (MTP style)
        chunk2 = ': "杭州"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        self.assertEqual(
            len(result2.calls),
            1,
            f"Expected 1 call, got {len(result2.calls)}. Calls: {result2.calls}",
        )


if __name__ == "__main__":
    unittest.main()
