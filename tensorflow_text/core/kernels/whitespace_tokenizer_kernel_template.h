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

#ifndef THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_WHITESPACE_TOKENIZER_KERNEL_TEMPLATE_H_
#define THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_WHITESPACE_TOKENIZER_KERNEL_TEMPLATE_H_

#include <iostream>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "tensorflow/lite/kernels/shim/op_kernel.h"
#include "tensorflow/lite/kernels/shim/shape.h"
#include "tensorflow/lite/kernels/shim/tensor_view.h"
#include "tensorflow_text/core/kernels/whitespace_tokenizer.h"

namespace tensorflow {
namespace text {

template <tflite::shim::Runtime Rt>
class WhitespaceTokenizeWithOffsetsV2Op
    : public tflite::shim::OpKernelShim<WhitespaceTokenizeWithOffsetsV2Op, Rt> {
 private:
  enum Inputs {
    kInputValues = 0,
    kInputConfig
  };
  enum Outputs {
    kOutputTokens = 0,
    kOutputRowSplits,
    kOutputStartOffsets,
    kOutputEndOffsets
  };

  using typename tflite::shim::OpKernelShim<WhitespaceTokenizeWithOffsetsV2Op,
                                            Rt>::InitContext;
  using typename tflite::shim::OpKernelShim<WhitespaceTokenizeWithOffsetsV2Op,
                                            Rt>::InvokeContext;
  using typename tflite::shim::OpKernelShim<WhitespaceTokenizeWithOffsetsV2Op,
                                            Rt>::ShapeInferenceContext;

 public:
  WhitespaceTokenizeWithOffsetsV2Op() = default;
  static constexpr char kOpName[] = "WhitespaceTokenizeWithOffsets";
  static constexpr char kDoc[] = R"doc(
  )doc";

  // Attributes declaration (name, type)
  static std::vector<std::string> Attrs() { return {}; }

  // Input tensors declaration (name, type, shape)
  static std::vector<tflite::shim::TensorDeclaration> Inputs();

  // Output tensors declaration (name, type, shape)
  static std::vector<tflite::shim::TensorDeclaration> Outputs();

  // Initializes the op
  absl::Status Init(InitContext* context) { return absl::OkStatus(); }

  // Runs the operation
  absl::Status Invoke(InvokeContext* context);

  // Shape inference
  static absl::Status ShapeInference(ShapeInferenceContext* c);
};

template <tflite::shim::Runtime Rt>
std::vector<tflite::shim::TensorDeclaration>
    WhitespaceTokenizeWithOffsetsV2Op<Rt>::Inputs() {
  return {{"input_values: string", tflite::shim::Shape({-1})},
          {"input_config: string", tflite::shim::Shape({})}};
}

template <tflite::shim::Runtime Rt>
std::vector<tflite::shim::TensorDeclaration>
    WhitespaceTokenizeWithOffsetsV2Op<Rt>::Outputs() {
  return {{"output_tokens: string", tflite::shim::Shape({-1})},
          {"output_row_splits: int64", tflite::shim::Shape({-1})},
          {"output_start_offsets: int32", tflite::shim::Shape({-1})},
          {"output_end_offsets: int32", tflite::shim::Shape({-1})}};
}

template <tflite::shim::Runtime Rt>
    absl::Status WhitespaceTokenizeWithOffsetsV2Op<Rt>
        ::Invoke(InvokeContext* context) {
  // Inputs
  const auto values_statusor = context->GetInput(kInputValues);
  if (!values_statusor.ok()) {
    return values_statusor.status();
  }
  const auto values = (*values_statusor)->template As<tensorflow::tstring, 1>();

  const auto cfg_statusor = context->GetInput(kInputConfig);
  if (!cfg_statusor.ok()) {
    return cfg_statusor.status();
  }
  const auto config = (*cfg_statusor)->template AsScalar<tensorflow::tstring>();

  auto tokenizer = tensorflow::text::WhitespaceTokenizer::Create(config);

  // Outputs
  std::vector<std::string> tokens;
  std::vector<int> row_splits;
  std::vector<int> start_offsets;
  std::vector<int> end_offsets;

  // Iterate through all the values and wordpiece tokenize them.
  row_splits.push_back(0);
  for (int i = 0; i < values.Dim(0); ++i) {
    // Tokenize into subwords and record the offset locations.
    const int orig_num_tokens = tokens.size();
    tokenizer.Tokenize(values(i), &tokens, &start_offsets, &end_offsets);
    const int delta_num_tokens = tokens.size() - orig_num_tokens;
    // Record the row splits.
    row_splits.push_back(delta_num_tokens + row_splits.back());
  }

  // Allocate output & fill output tensors.
  // TODO(rnale): investigate using memcpy like previous WST
#define FILL_OUTPUT_TENSOR(name, index, dtype)                               \
  const int name##_size = name.size();                                       \
  const auto name##_statusor = context->GetOutput(index,                     \
      tflite::shim::Shape({name##_size}));                                   \
  if (!name##_statusor.ok()) {                                               \
    return name##_statusor.status();                                         \
  }                                                                          \
  auto name##_data = (*name##_statusor)->template As<dtype, 1>();            \
  for (int i = 0; i < name##_size; ++i) {                                    \
    name##_data(i) = name[i];                                                \
  }

  FILL_OUTPUT_TENSOR(tokens, kOutputTokens, tensorflow::tstring);
  FILL_OUTPUT_TENSOR(row_splits, kOutputRowSplits, tensorflow::int64);
  FILL_OUTPUT_TENSOR(start_offsets, kOutputStartOffsets, tensorflow::int32);
  FILL_OUTPUT_TENSOR(end_offsets, kOutputEndOffsets, tensorflow::int32);

#undef FILL_OUTPUT_TENSOR

  return absl::OkStatus();
}

template <tflite::shim::Runtime Rt>
absl::Status WhitespaceTokenizeWithOffsetsV2Op<Rt>::ShapeInference(
    ShapeInferenceContext* c) {
  using tflite::shim::Shape;
  const auto input_values_shape_status = c->GetInputShape(kInputValues);
  if (!input_values_shape_status.ok()) {
    return input_values_shape_status.status();
  }
  const Shape input_values_shape = *input_values_shape_status;

  const auto rank_1_shape = Shape({Shape::kUnknownDim});
  auto status = c->SetOutputShape(kOutputTokens, rank_1_shape);
  if (!status.ok()) {
    return status;
  }
  status = c->SetOutputShape(kOutputStartOffsets, rank_1_shape);
  if (!status.ok()) {
    return status;
  }
  status = c->SetOutputShape(kOutputEndOffsets, rank_1_shape);
  if (!status.ok()) {
    return status;
  }
  const int num_splits = Shape::AddDims(1, input_values_shape.Dim(0));
  status = c->SetOutputShape(kOutputRowSplits, Shape({num_splits}));
  if (!status.ok()) {
    return status;
  }
  return absl::OkStatus();
}

}  // namespace text
}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_TEXT_CORE_KERNELS_WHITESPACE_TOKENIZER_KERNEL_TEMPLATE_H_
