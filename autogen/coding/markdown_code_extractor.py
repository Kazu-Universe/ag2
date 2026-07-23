# Copyright (c) 2023 - 2025, AG2ai, Inc., AG2ai open-source projects maintainers and core contributors
#
# SPDX-License-Identifier: Apache-2.0
#
# Portions derived from https://github.com/microsoft/autogen are under the MIT License.
# SPDX-License-Identifier: MIT
import re
from typing import Union

from ..code_utils import CODE_BLOCK_PATTERN, MD_CODE_BLOCK_PATTERN, UNKNOWN, content_str, infer_lang, BASH_CODE_BLOCK_PATTERN
from ..doc_utils import export_module
from ..types import UserMessageImageContentPart, UserMessageTextContentPart
from .base import CodeBlock, CodeExtractor
from ..cmbagent_utils import cmbagent_debug
import json

__all__ = ("MarkdownCodeExtractor",)

# original ag2 code
# @export_module("autogen.coding")
# class MarkdownCodeExtractor(CodeExtractor):
#     """(Experimental) A class that extracts code blocks from a message using Markdown syntax."""

#     def extract_code_blocks(
#         self, message: Union[str, list[Union[UserMessageTextContentPart, UserMessageImageContentPart]], None]
#     ) -> list[CodeBlock]:
#         """(Experimental) Extract code blocks from a message. If no code blocks are found,
#         return an empty list.

#         Args:
#             message (str): The message to extract code blocks from.

#         Returns:
#             List[CodeBlock]: The extracted code blocks or an empty list.
#         """
#         ## cmbagent debug print: 
#         print('in markdown_code_extractor.py extract_code_blocks message: ', message)
#         text = content_str(message)
#         print('in markdown_code_extractor.py extract_code_blocks text: ', text)
#         match = re.findall(CODE_BLOCK_PATTERN, text, flags=re.DOTALL)
#         if not match:
#             return []
#         code_blocks = []
#         for lang, code in match:
#             if lang == "":
#                 lang = infer_lang(code)
#             if lang == UNKNOWN:
#                 lang = ""
#             code_blocks.append(CodeBlock(code=code, language=lang))
#         return code_blocks


# cmbagent modified code
@export_module("autogen.coding")
class MarkdownCodeExtractor(CodeExtractor):
    """(Experimental) A class that extracts code blocks from a message using Markdown syntax,
    and also supports extraction from JSON messages that contain a 'python_code' field.
    """

    def extract_code_blocks(
        self, message: Union[str, list[Union[UserMessageTextContentPart, UserMessageImageContentPart]], None], name: str = ""
    ) -> list[CodeBlock]:
        """Extract code blocks from a message. First, if the message is valid JSON and contains
        a "python_code" field, extract that code. Otherwise, fall back to the Markdown regex extraction.

        Args:
            message (str): The message to extract code blocks from.

        Returns:
            List[CodeBlock]: The extracted code blocks or an empty list.
        """
        # cmbagent debug print: 
        # print('in markdown_code_extractor.py extract_code_blocks message: ', message)
        text = content_str(message)
        if cmbagent_debug:
            print('in markdown_code_extractor.py extract_code_blocks text: ', text)

        if name == "researcher_response_formatter":
            if cmbagent_debug:
                print('in markdown_code_extractor.py name: ', name)
                print('in markdown_code_extractor.py text: ', text)
            match = re.findall(MD_CODE_BLOCK_PATTERN, text, flags=re.DOTALL)
            if cmbagent_debug:
                print('in markdown_code_extractor.py match: ', match)
            if not match:
                return []
            code_blocks = []
            for code in match:
                code_blocks.append(CodeBlock(code=code, language="markdown"))
            return code_blocks

        # Attempt to parse the message as JSON and extract "python_code"
        try:
            data = json.loads(text)
            if "python_code" in data:
                python_code = data["python_code"]
                # Optionally, you could further process python_code here if needed
                return [CodeBlock(code=python_code, language="python")]
            elif "structured_code" in data:  # Added support for structured_code.
                structured_code = data["structured_code"]
                return [CodeBlock(code=structured_code, language="python")]
        except json.JSONDecodeError:
            # The message is not valid JSON; fall back to Markdown extraction.
            ## cmbagent debug print: 
            # print('in markdown_code_extractor.py message is not valid JSON, fall back to Markdown extraction')
            pass

        # Fall back to Markdown extraction using the regex pattern
        match = re.findall(CODE_BLOCK_PATTERN, text, flags=re.DOTALL)
        if cmbagent_debug:
            print('in markdown_code_extractor.py match: ', match)
            print('in markdown_code_extractor.py match: ', match)
            print('in markdown_code_extractor.py name: ', name)
        if name=="installer":
            # print('in markdown_code_extractor.py bash in name')
            match = re.findall(BASH_CODE_BLOCK_PATTERN, text, flags=re.DOTALL)
            # code_blocks = []
            # # code_blocks.append(CodeBlock(code=match, language="sh"))
            # # return code_blocks
        # print('in markdown_code_extractor.py bash match: ', match)
        if not match:
            return []
        code_blocks = []
        for lang, code in match:
            if lang == "":
                lang = infer_lang(code)
            if lang == UNKNOWN:
                lang = ""
            code_blocks.append(CodeBlock(code=code, language=lang))
        return code_blocks