/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relax/ir/struct_info.cc
 * \brief Relax struct info.
 */
#include <tvm/relax/struct_info.h>
#include <tvm/runtime/registry.h>

namespace tvm {
namespace relax {

ObjectStructInfo::ObjectStructInfo(Span span) {
  ObjectPtr<ObjectStructInfoNode> n = make_object<ObjectStructInfoNode>();
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ObjectStructInfoNode);

TVM_REGISTER_GLOBAL("relax.ObjectStructInfo").set_body_typed([](Span span) {
  return ObjectStructInfo(span);
});

// Prim
PrimStructInfo::PrimStructInfo(DataType dtype, Span span) {
  ObjectPtr<PrimStructInfoNode> n = make_object<PrimStructInfoNode>();
  n->dtype = dtype;
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(PrimStructInfoNode);

TVM_REGISTER_GLOBAL("relax.PrimStructInfo").set_body_typed([](DataType dtype, Span span) {
  return PrimStructInfo(dtype, span);
});

// Shape
ShapeStructInfo::ShapeStructInfo(Array<PrimExpr> values, Span span) {
  ObjectPtr<ShapeStructInfoNode> n = make_object<ShapeStructInfoNode>();
  // assign ndim first before move;
  n->ndim = static_cast<int>(values.size());
  n->values = std::move(values);
  n->span = span;
  data_ = std::move(n);
}

ShapeStructInfo::ShapeStructInfo(int ndim, Span span) {
  ObjectPtr<ShapeStructInfoNode> n = make_object<ShapeStructInfoNode>();
  n->ndim = ndim;
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(ShapeStructInfoNode);

TVM_REGISTER_GLOBAL("relax.ShapeStructInfo")
    .set_body_typed([](Optional<Array<PrimExpr>> values, int ndim, Span span) {
      if (values.defined()) {
        CHECK_EQ(ndim, kUnknownDim) << "ValueError: Cannot both specify values and ndim";
        return ShapeStructInfo(values.value(), span);
      } else {
        return ShapeStructInfo(ndim, span);
      }
    });

// Tensor
TensorStructInfo::TensorStructInfo(Expr shape, DataType dtype, Span span) {
  ObjectPtr<TensorStructInfoNode> n = make_object<TensorStructInfoNode>();
  // assign ndim before move
  Optional<ShapeStructInfo> sinfo = MatchStructInfo<ShapeStructInfo>(shape);
  ICHECK(sinfo) << "We expect shape to contain pre-set shape struct info";
  n->ndim = sinfo.get()->ndim;
  // assign rest of the fields.
  n->shape = std::move(shape);
  n->dtype = dtype;
  n->span = span;
  data_ = std::move(n);
}

TensorStructInfo::TensorStructInfo(DataType dtype, int ndim, Span span) {
  ObjectPtr<TensorStructInfoNode> n = make_object<TensorStructInfoNode>();
  n->ndim = ndim;
  n->dtype = dtype;
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TensorStructInfoNode);

TVM_REGISTER_GLOBAL("relax.TensorStructInfo")
    .set_body_typed([](Optional<Expr> shape, DataType dtype, int ndim, Span span) {
      if (shape.defined()) {
        CHECK_EQ(ndim, kUnknownDim) << "ValueError: Cannot both specify shape and ndim";
        return TensorStructInfo(shape.value(), dtype, span);
      } else {
        return TensorStructInfo(dtype, ndim, span);
      }
    });

// Tuple
TupleStructInfo::TupleStructInfo(Array<StructInfo> fields, Span span) {
  ObjectPtr<TupleStructInfoNode> n = make_object<TupleStructInfoNode>();
  n->fields = std::move(fields);
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_NODE_TYPE(TupleStructInfoNode);

TVM_REGISTER_GLOBAL("relax.TupleStructInfo")
    .set_body_typed([](Array<StructInfo> fields, Span span) {
      return TupleStructInfo(fields, span);
    });

// Func
FuncStructInfo::FuncStructInfo(Array<StructInfo> params, StructInfo ret, Span span) {
  ObjectPtr<FuncStructInfoNode> n = make_object<FuncStructInfoNode>();
  n->params = std::move(params);
  n->ret = std::move(ret);
  n->span = span;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("tvm.relax.struct_info_derive.opaque").set_body_typed([](const Call& call) {
  // register struct info derivation function.
  return ObjectStructInfo();
});

FuncStructInfo FuncStructInfo::OpaqueFunc(Optional<StructInfoDeriveFunc> derive_func, Span span) {
  ObjectPtr<FuncStructInfoNode> n = make_object<FuncStructInfoNode>();
  if (derive_func.defined()) {
    n->derive_func = derive_func;
  } else {
    static auto env_func = EnvFunc::Get("tvm.relax.struct_info_derive.opaque");
    StructInfoDeriveFunc derive_func;
    derive_func = env_func;
    n->derive_func = derive_func;
  }
  n->span = span;
  return FuncStructInfo(n);
}

TVM_REGISTER_NODE_TYPE(FuncStructInfoNode);

TVM_REGISTER_GLOBAL("relax.FuncStructInfo")
    .set_body_typed([](Array<StructInfo> params, StructInfo ret, Span span) {
      return FuncStructInfo(params, ret, span);
    });

TVM_REGISTER_GLOBAL("relax.FuncStructInfoOpaqueFunc")
    .set_body_typed([](Optional<StructInfoDeriveFunc> derive_func, Span span) {
      return FuncStructInfo::OpaqueFunc(derive_func, span);
    });

// Helper functions
void UpdateStructInfo(Expr expr, StructInfo struct_info) {
  ICHECK(!expr->struct_info_.defined())
      << "the struct+info_ of the Expr to be updated must be nullptr for idempotency";
  expr->struct_info_ = struct_info;
}

TVM_REGISTER_GLOBAL("relax.UpdateStructInfo").set_body_typed([](Expr expr, StructInfo struct_info) {
  UpdateStructInfo(expr, struct_info);
});

TVM_REGISTER_GLOBAL("ir.ExprStructInfo").set_body_typed([](Expr expr) {
  ICHECK(expr->struct_info_.defined())
      << "struct info is not populated";
  return expr->struct_info_;
});

}  // namespace relax
}  // namespace tvm
