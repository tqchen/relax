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
 * \file src/relax/backend/vm/codegen_xyz.cc
 * \brief A codegen to generate XYZ from executable.
 */
#include <tvm/ir/module.h>
#include <tvm/relax/exec_builder.h>
#include <tvm/runtime/relax_vm/executable.h>
#include <tvm/target/target.h>
#include <tvm/driver/driver_api.h>
#include <tvm/relax/attrs/builtin.h>
#include <tvm/relax/attrs/memory.h>
#include <tvm/relax/attrs/shape.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/target/target.h>
#include <tvm/tir/function.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>


#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace relax {
namespace relax_vm {

/*!
 * \brief A class to generate VMTIR for Relax functions.
 */
class CodeGenVMTIR : public ExprFunctor<Optional<PrimExpr>(const Expr&)> {
 public:
  explicit CodeGenVMTIR(relax::ExecBuilder builder, IRModule ctx_mod)
      : builder_(builder), ctx_mod_(ctx_mod) {}

 protected:
  int64_t NewRegister() { return registers_num_++; }

  static IntImm ConstInt64(int64_t value) {
    return IntImm(DataType::Int(64), value);
  }

  PrimExpr RegListGet(int64_t slot) const {
    // use 128 bits to represent any
    return tir::Call(DataType::Handle(128), tir::builtin::anylist_getitem(),
                     {reg_anylist_handle_, ConstInt64(slot)});
  }

  PrimExpr ConstListGet(int64_t slot) const {
    // use 128 bits to represent any
    return tir::Call(DataType::Handle(128), tir::builtin::anylist_getitem(),
                     {const_anylist_handle_, ConstInt64(slot)});
  }

  void EmitCallPacked(String name, const Array<PrimExpr>& args, int64_t dst_anylist_slot = -1) {
    ICHECK(!stmt_stack_.empty());
    auto seq = &stmt_stack_.back();
    Array<PrimExpr> all_args;
    // negative index indicate return value can be discarded, emit call_packed
    if (dst_anylist_slot >= 0) {
      all_args =
        {reg_anylist_handle_, ConstInt64(dst_anylist_slot)};
    }
    all_args.push_back(tir::StringImm(name));
    for (PrimExpr arg : args) {
      all_args.push_back(arg);
    }
    if (dst_anylist_slot >= 0) {
      seq->emplace_back(
        tir::Evaluate(
          tir::Call(DataType::Int(32), tir::builtin::anylist_setitem_call_packed(), all_args)));
    } else {
      seq->emplace_back(
        tir::Evaluate(
          tir::Call(DataType::Int(32), tir::builtin::tvm_call_packed(), all_args)));
    }
  }

  Optional<PrimExpr> VisitExpr_(const FunctionNode* func) final {
    Optional<String> gsymbol = func->GetAttr<String>(tvm::attr::kGlobalSymbol);
    ICHECK(gsymbol.defined()) << "there should be no local functions in Relax VM codegen phase. "
                                 "Did you forget to apply LambdaLift or AttachGlobalSymbol Pass?";
    // initialize the state
    stmt_stack_ = {std::vector<tir::Stmt>()};
    registers_num_ = 0;
    var_map_.clear();
    ctx_ptr_ = tir::Var("ctx_ptr", DataType::Handle());
    const_anylist_handle_ = tir::Var("clist", DataType::Handle());
    reg_anylist_handle_ = tir::Var("rlist", DataType::Handle());

    Array<String> param_names;
    for (Var param : func->params) {
      param_names.push_back(param->name_hint());
    }
    // declare this function.
    builder_->DeclareFunction(gsymbol.value(), vm::VMFuncInfo::FuncKind::kVMTIRFunc);

    for (size_t i = 0; i < func->params.size(); ++i) {
      int64_t r = NewRegister();
      ICHECK_EQ(static_cast<size_t>(r), i);
      this->var_map_.insert({func->params[i], RegListGet(r)});
    }
    size_t ret_reg = NewRegister();
    Optional<PrimExpr> ret = ExprFunctor::VisitExpr(func->body);

    if (ret.defined()) {
      this->EmitCallPacked("vm.builtin.copy", {ret.value()}, ret_reg);
    }

    builder_->EndFunction(gsymbol.value());
    // reset register number to be 0;
    registers_num_ = 0;
    var_map_.clear();
    stmt_stack_.clear();
    return NullOpt;
  }

  Optional<PrimExpr> VisitExpr_(const SeqExprNode* op) final {
    for (auto block : op->blocks) {
      for (Binding binding : block->bindings) {
        Optional<PrimExpr> value;
        if (auto* var_binding = binding.as<VarBindingNode>()) {
          value = this->VisitExpr(var_binding->value);
        } else if (auto* match_cast = binding.as<MatchCastNode>()) {
          value = this->VisitExpr(match_cast->value);
        } else {
          LOG(FATAL) << "Unsupported binding " << binding->GetTypeKey();
        }
        this->var_map_.insert({binding->var, value});
      }
    }
    return this->VisitExpr(op->body);
  }

  Optional<PrimExpr> VisitExpr_(const CallNode* call_node) final {
    Call call = GetRef<Call>(call_node);

    if (call_node->op == null_value_op_) {
      return tir::Call(DataType::Handle(),
                       tir::builtin::reinterpret(),
                       {IntImm(DataType::Int(64), 0)});
    }

    // allocate dst register.
    int64_t dst_reg = HasVoidStructInfo(call) ?  -1 : NewRegister();
    if (call->op.as<OpNode>()) {
      if (call_node->op == call_builtin_op_) {
    //     EmitCallBuiltin(call, dst_reg);
      } else if (call_node->op == alloc_storage_op_) {
        EmitAllocStorage(call, dst_reg);
      } else if (call_node->op == alloc_tensor_op_) {
      //  EmitAllocTensor(call, dst_reg);
      } else {
        // every "normal" operator is lowered to a global var in the IRModule. The Attrs for those
        // ops are handled in a pass when lowering them to TIR.
        LOG(FATAL) << "CodeGenVMTIR cannot handle this intrinsic now:\n" << call_node->op;
      }
    } else {
      //EmitNormalCall(call, dst_reg);
    }
    if (dst_reg >= 0) {
      return RegListGet(dst_reg);
    } else {
      return NullOpt;
    }
  }

  void EmitAllocStorage(const Call& call_node, int64_t dst_reg) {
    // Handle args of the call
    Array<PrimExpr> args;
    args.push_back(ctx_ptr_);
    for (Expr arg : call_node->args) {
      args.push_back(this->VisitExpr(arg).value());
    }
    // Handle attrs of the call
    auto alloc_attrs = call_node->attrs.as<VMAllocStorageAttrs>();
    ICHECK(alloc_attrs != nullptr) << "must be VMAllocStorageAttrs";
    args.push_back(ConstInt64(alloc_attrs->runtime_device_index));
    args.push_back(ConstListGet(builder_->ConvertConstant(alloc_attrs->dtype).value()));
    this->EmitCallPacked("vm.builtin.alloc_storage", args, dst_reg);
  }

  void EmitAllocTensor(const Call& call_node, int64_t dst_reg) {
    ICHECK_EQ(call_node->args.size(), 2);
    Array<PrimExpr> args;
    args.reserve(4);
    args.push_back(this->VisitExpr(call_node->args[0]).value());
    auto alloc_attrs = call_node->attrs.as<VMAllocTensorAttrs>();
    ICHECK(alloc_attrs != nullptr) << "must be VMAllocTensorAttrs";
    int offset = alloc_attrs->offset;
    args.push_back(ConstInt64(offset));
    args.push_back(this->VisitExpr(call_node->args[1]).value());
    // Handle `dtype`
    args.push_back(ConstListGet(builder_->ConvertConstant(alloc_attrs->dtype).value()));
    this->EmitCallPacked("vm.builtin.alloc_tensor", args, dst_reg);
  }

  void EmitCallBuiltin(const Call& call_node, int64_t dst_reg) {
    auto builtin_attrs = call_node->attrs.as<BuiltinFuncAttrs>();
    ICHECK(builtin_attrs != nullptr);
    Array<PrimExpr> args;
    // if context is required, pass as first argument.
    if (builtin_attrs->require_ctx) {
      args.push_back(ctx_ptr_);
    }
    auto* func = call_node->args[0].as<ExternFuncNode>();
    ICHECK(func) << "CallBuiltin comes with extern func";

    auto tuple_arg = Downcast<Tuple>(call_node->args[1]);

    // Handle args of the call
    for (Expr arg : tuple_arg->fields) {
      args.push_back(this->VisitExpr(arg).value());
    }

    if (builtin_attrs->int_args.defined()) {
      for (auto val : builtin_attrs->int_args) {
        args.push_back(ConstInt64(val->value));
      }
    }
    if (builtin_attrs->dtype_arg != DataType::Void()) {
      args.push_back(ConstListGet(builder_->ConvertConstant(builtin_attrs->dtype_arg).value()));
    }

    if (builtin_attrs->str_args.defined()) {
      for (auto val : builtin_attrs->str_args) {
        args.push_back(tir::StringImm(val));
      }
    }
    this->EmitCallPacked(func->global_symbol, args, dst_reg);
  }


  /*! \brief Internal ExecBuilder. */
  relax::ExecBuilder builder_;
  /*! \brief List to ctx_ptr */
  tir::Var ctx_ptr_;
  /*! \brief List to store temp object registers */
  tir::Var reg_anylist_handle_;
  /*! \brief List to store constants */
  tir::Var const_anylist_handle_;
  /*!
   * \brief Total number of virtual registers allocated.
   * \note The first two registers are reserved for special registers.
   */
  int64_t registers_num_ = 0;
  /*! \brief Stack to build up statements */
  std::vector<std::vector<tir::Stmt>> stmt_stack_;
  /*! \brief Map from var to Expr. */
  std::unordered_map<Var, Optional<PrimExpr>, ObjectPtrHash, ObjectPtrEqual> var_map_;
  /*! \brief the context module. */
  IRModule ctx_mod_;
  /*! \brief Cache ops that need to be frequently used later to reduce lookup overhead. */
  const Op& alloc_storage_op_ = Op::Get("relax.vm.builtin.alloc_storage");
  const Op& alloc_tensor_op_ = Op::Get("relax.vm.builtin.alloc_tensor");
  const Op& call_builtin_op_ = Op::Get("relax.call_builtin");
  const Op& null_value_op_ = Op::Get("relax.null_value");
};

/*!
 * \brief Create the Relax VM executable.
 */
Array<ObjectRef> VMTIRCodeGen(IRModule mod, Target target) {
  // TODO(relax-team) Revisit the param and ext_lib options.
  return {};
}

TVM_REGISTER_GLOBAL("relax.VMTIRCodeGen").set_body_typed(VMTIRCodeGen);

}  // namespace relax_vm
}  // namespace relax
}  // namespace tvm
