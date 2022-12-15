/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file binary.cc
 * \brief binary broadcast operators.
 */

#include "binary.h"

namespace tvm {
namespace relax {

RELAX_REGISTER_BINARY_BROADCAST_OP("add")
    .describe("Elementwise add with broadcasting")
    .set_support_level(1);

RELAX_REGISTER_BINARY_BROADCAST_OP("multiply")
    .describe("Elementwise multiply with broadcasting")
    .set_support_level(1);

StructInfo InferStructInfoBroadcast(const Call& call, const BlockBuilder& ctx) {
  if (call->args.size() != 2) {
    ctx->ReportFatal(Diagnostic::Error(call->span)
                     << "Binary broadcast op should have 2 arguments");
  }
  auto* lhs_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  auto* rhs_sinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[1]);
  if (!lhs_sinfo || !rhs_sinfo) {
    ctx->ReportFatal(Diagnostic::Error(call->span)
                     << "Both lhs and rhs should be Tensor for broadcasting, but got "
                     << call->args[0]->struct_info_->GetTypeKey() << " and "
                     << call->args[0]->struct_info_->GetTypeKey());
  }

  // DateType
  DataType output_dtype;
  if (lhs_sinfo->IsUnknownDtype() || rhs_sinfo->IsUnknownDtype()) {
    output_dtype = DataType::Void();
  } else if (lhs_sinfo->dtype != rhs_sinfo->dtype) {
    ctx->ReportFatal(Diagnostic::Error(call->span)
                     << "Data types " << lhs_sinfo->dtype << " and " << rhs_sinfo->dtype
                     << " must be equal for broadcasting operators");
  } else {
    output_dtype = lhs_sinfo->dtype;
  }

  // ndims
  int output_ndim;
  if (lhs_sinfo->IsUnknownNdim() || rhs_sinfo->IsUnknownNdim()) {
    output_ndim = kUnknownDim;
  } else {
    output_ndim = std::max(lhs_sinfo->ndim, rhs_sinfo->ndim);
  }

  // Shapes and ndims
  if (lhs_sinfo->shape && rhs_sinfo->shape) {
    // If all inputs have shapes, directly infer shapes
    std::vector<PrimExpr> output_shape;

    auto check_shape_expr = [&](const Expr& shape) -> const ShapeExprNode* {
      if (const auto* shape_expr = shape.as<ShapeExprNode>()) {
        return shape_expr;
      } else {
        ctx->ReportFatal(Diagnostic::Error(call->span)
                         << "Shapes are expected to be ShapeExpr, but got: " << shape << ".");
        return nullptr;
      }
    };
    const ShapeExprNode* lhs_shape = check_shape_expr(lhs_sinfo->shape.value());
    const ShapeExprNode* rhs_shape = check_shape_expr(rhs_sinfo->shape.value());
    size_t lhs_ndim = lhs_sinfo->ndim;
    size_t rhs_ndim = rhs_sinfo->ndim;
    size_t max_ndim = std::max(lhs_ndim, rhs_ndim);

    size_t i = 1;
    for (; i <= std::min(lhs_ndim, rhs_ndim); ++i) {
      const PrimExpr& dim0 = lhs_shape->values[lhs_ndim - i];
      const PrimExpr& dim1 = rhs_shape->values[rhs_ndim - i];
      if (EqualConstInt(dim0, 1)) {
        output_shape.push_back(dim1);
      } else if (EqualConstInt(dim1, 1)) {
        output_shape.push_back(dim0);
      } else if (EqualCheck(dim0, dim1)) {
        output_shape.push_back(dim0);
      } else {
        // defer the computation of output shapes to runtime
        // e.g., broadcast Tensor([m, n]), Tensor([k]) -> defer to runtime
        Call call_infer(ExternFunc(String("vm.binary_broadcast_shape_infer")),
                        {call->args[0], call->args[1]}, Attrs(), {ShapeType(max_ndim)});
        return TensorStructInfo(ctx->NormalizeArgument(call_infer), output_dtype);
      }
    }
    auto& longer_shape = (lhs_ndim > rhs_ndim) ? lhs_shape : rhs_shape;
    for (; i <= max_ndim; ++i) {
      output_shape.push_back(longer_shape->values[max_ndim - i]);
    }
    Expr shape = ShapeExpr(Array<PrimExpr>(output_shape.rbegin(), output_shape.rend()));
    return TensorStructInfo(shape, output_dtype);
  } else {
    return TensorStructInfo(output_dtype, /*ndim=*/output_ndim);
  }
}

}  // namespace relax
}  // namespace tvm
