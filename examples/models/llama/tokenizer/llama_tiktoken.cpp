/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/examples/models/llama/tokenizer/llama_tiktoken.h>

namespace example {

using ::tokenizers::Tiktoken;
using ::tokenizers::Tokenizer;

namespace {
static constexpr int32_t kSpecialTokensSize = 256;
static constexpr size_t kBOSTokenIndex = 0;
static constexpr size_t kEOSTokenIndex = 1;

// Compile-time special tokens selection using templates
template <Version V>
struct SpecialTokensSelector {
  static std::unique_ptr<std::vector<std::string>> get();
};

// Compile-time special tokens selection using templates
template <>
struct SpecialTokensSelector<Version::Default> {
  static std::unique_ptr<std::vector<std::string>> get() {
    auto special_tokens =
        std::make_unique<std::vector<std::string>>(std::vector<std::string>{
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|finetune_right_pad_id|>",
            "<|step_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eom_id|>",
            "<|eot_id|>",
            "<|python_tag|>"});
    // pad the rest of the special tokens with reserved tokens
    ssize_t reserved_special_token_num = 2;
    while (special_tokens->size() < kSpecialTokensSize) {
      special_tokens->emplace_back(
          "<|reserved_special_token_" +
          std::to_string(reserved_special_token_num++) + "|>");
    }
    return special_tokens;
  }
};

// Specialization for Multimodal version
template <>
struct SpecialTokensSelector<Version::Multimodal> {
  static std::unique_ptr<std::vector<std::string>> get() {
    return get_multimodal_special_tokens();
  }
};

} // namespace

namespace detail {
// Helper function to create a Tiktoken with the given version
template <typename PtrType, Version V>
PtrType create_tiktoken() {
  std::unique_ptr<std::vector<std::string>> special_tokens =
      example::SpecialTokensSelector<V>::get();
  if constexpr (is_shared_ptr_of_tokenizer<PtrType>()) {
    return std::make_shared<Tiktoken>(
        std::move(special_tokens), kBOSTokenIndex, kEOSTokenIndex);
  } else if constexpr (is_unique_ptr_of_tokenizer<PtrType>()) {
    return std::make_unique<Tiktoken>(
        std::move(special_tokens), kBOSTokenIndex, kEOSTokenIndex);
  } else {
    static_assert(
        is_shared_ptr_of_tokenizer<PtrType>() ||
            is_unique_ptr_of_tokenizer<PtrType>(),
        "PtrType must be either std::shared_ptr<Tiktoken> or std::unique_ptr<Tiktoken>");
    // This line is never reached due to the static_assert, but needed for
    // compilation
    return PtrType{};
  }
}
} // namespace detail

// Function that returns a shared_ptr
template <typename PtrType, Version V>
PtrType get_tiktoken_for_llama() {
  return detail::create_tiktoken<PtrType, V>();
}

std::unique_ptr<std::vector<std::string>> get_multimodal_special_tokens() {
  auto special_tokens =
      std::make_unique<std::vector<std::string>>(std::vector<std::string>{
          "<|begin_of_text|>",
          "<|end_of_text|>",
          "<|reserved_special_token_0|>",
          "<|reserved_special_token_1|>",
          "<|reserved_special_token_2|>",
          "<|reserved_special_token_3|>",
          "<|start_header_id|>",
          "<|end_header_id|>",
          "<|eom_id|>",
          "<|eot_id|>",
          "<|image|>"});

  // pad the rest of the special tokens with reserved tokens except the last
  // one
  ssize_t reserved_special_token_num = 4;
  while (special_tokens->size() < kSpecialTokensSize - 1) {
    special_tokens->emplace_back(
        "<|reserved_special_token_" +
        std::to_string(reserved_special_token_num++) + "|>");
  }

  special_tokens->emplace_back("<|python_tag|>");

  return special_tokens;
}

// specialization

template std::shared_ptr<Tiktoken>
get_tiktoken_for_llama<std::shared_ptr<Tiktoken>, Version::Multimodal>();

template std::unique_ptr<Tiktoken>
get_tiktoken_for_llama<std::unique_ptr<Tiktoken>, Version::Multimodal>();

template std::shared_ptr<Tiktoken>
get_tiktoken_for_llama<std::shared_ptr<Tiktoken>, Version::Default>();

template std::unique_ptr<Tiktoken>
get_tiktoken_for_llama<std::unique_ptr<Tiktoken>, Version::Default>();

template std::shared_ptr<Tokenizer>
get_tiktoken_for_llama<std::shared_ptr<Tokenizer>, Version::Multimodal>();

template std::unique_ptr<Tokenizer>
get_tiktoken_for_llama<std::unique_ptr<Tokenizer>, Version::Multimodal>();

template std::shared_ptr<Tokenizer>
get_tiktoken_for_llama<std::shared_ptr<Tokenizer>, Version::Default>();

template std::unique_ptr<Tokenizer>
get_tiktoken_for_llama<std::unique_ptr<Tokenizer>, Version::Default>();
} // namespace example
