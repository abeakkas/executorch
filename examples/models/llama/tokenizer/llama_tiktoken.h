/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pytorch/tokenizers/tiktoken.h>

namespace example {

enum class Version {
  Default,
  Multimodal,
};

// Type traits to check if a type is a shared_ptr or unique_ptr of Tokenizer or
// a derived class
template <typename T>
struct is_shared_ptr_of_tokenizer : std::false_type {};

template <typename T>
struct is_shared_ptr_of_tokenizer<std::shared_ptr<T>>
    : std::is_base_of<::tokenizers::Tokenizer, T> {};

template <typename T>
struct is_unique_ptr_of_tokenizer : std::false_type {};

template <typename T>
struct is_unique_ptr_of_tokenizer<std::unique_ptr<T>>
    : std::is_base_of<::tokenizers::Tokenizer, T> {};

// Template version that can return either shared_ptr or unique_ptr
template <typename PtrType, Version V = Version::Default>
PtrType get_tiktoken_for_llama();

// For backward compatibility
std::unique_ptr<std::vector<std::string>> get_multimodal_special_tokens();

} // namespace example
