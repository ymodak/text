// Copyright 2021 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_WHITESPACE_TOKENIZER_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_WHITESPACE_TOKENIZER_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"


namespace tensorflow {
namespace text {

class WhitespaceTokenizer {
 public:
  // Creates an instance.
  //
  // Args:
  //  * config: pointer to the WhitespaceTokenizerConfig which is not owned by
  //    this instance and should be kept alive through the lifetime of the
  //    instance. This should be created using the
  //    WhitespaceTokenizerConfigBuilder
  static WhitespaceTokenizer Create(const absl::string_view config);

  // Tokenizes a string (or series of character codepoints) by whitespace.
  //
  // Example:
  // input = "Show me the way."
  // tokens = ["Show", "me", "the", "way."]
  // start_offsets = [0, 5, 8, 12]
  // end_offsets = [4, 7, 11, 16]
  //
  // The input should be UTF-8 but the tokenization is performed on Unicode
  // codepoints.
  //
  // Args:
  //  * input: The UTF-8 string of an input.
  //  * tokens: The output tokens.
  //  * start_offsets: The start offsets of output tokens in the input
  //    text, in utf-8 bytes.
  //  * end_offsets: The end offsets of output tokens in the input
  //    text, in utf-8 bytes.
  // Note: the start offsets are inclusive and the end offsets are exclusive.
  void Tokenize(const absl::string_view input,
                std::vector<std::string>* tokens,
                std::vector<int>* start_offsets,
                std::vector<int>* end_offsets);

  // Tokenizes a string (or series of character codepoints) by whitespace.
  //
  // Example:
  // input = "Show me the way."
  // output = ["Show", "me", "the", "way."]
  //
  // The input should be UTF-8 but the tokenization is performed on Unicode
  // codepoints.
  //
  // Args:
  //  * input: The UTF-8 string of an input.
  //  * tokens: The output tokens.
  void Tokenize(const absl::string_view input,
                std::vector<std::string>* tokens);

 private:
  WhitespaceTokenizer(const absl::string_view cfg, const int max)
      : config(cfg), max_codepoint(max) { }

  const absl::string_view config;
  const int max_codepoint;
};

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_WHITESPACE_TOKENIZER_H_
