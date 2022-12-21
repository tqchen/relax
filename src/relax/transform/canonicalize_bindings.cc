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
 * \file tvm/relax/transform/canonicalize.cc
 * \brief Pass for simplifying modules by folding var bindings and match shape nodes.
 *        May include other forms of simplification in the future.
 *        Ideally should be used before constant folding and eliminating unused bindings.
 */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

class BindingCanonicalizer : public ExprMutator {
 public:
  BindingCanonicalizer() {}

  Expr VisitExpr_(const VarNode* op) override {
    // remap first
    Var v = Downcast<Var>(ExprMutator::VisitExpr_(op));
    if (!CanCanonicalizeVar(v)) {
      return Downcast<Expr>(v);
    }
    // visit again in case we need to do a substitution in the value
    return ExprMutator::VisitExpr_(LookupBinding(v).as<VarNode>());
  }

  Expr VisitExpr_(const DataflowVarNode* op) override {
    Var v = Downcast<Var>(ExprMutator::VisitExpr_(op));
    if (!CanCanonicalizeVar(v)) {
      return Downcast<Expr>(v);
    }
    return ExprMutator::VisitExpr_(LookupBinding(v).as<DataflowVarNode>());
  }

  void VisitBinding_(const VarBindingNode* binding) override {
    // Unlike default visitor, we do not permit the checked type to change
    // if the new value's checked type is different (this preserves user annotations)
    Expr new_value = this->VisitExpr(binding->value);
    Var new_var = this->VisitVarDef(binding->var);

    auto emit = [this](VarBinding b) {
      if (this->builder_->CurrentBlockIsDataFlow() && !b->var.as<DataflowVarNode>()) {
        this->builder_->EmitOutput(b);
      } else {
        this->builder_->Emit(b);
      }
    };

    if (new_var.same_as(binding->var) && new_value.same_as(binding->value)) {
      emit(GetRef<VarBinding>(binding));
      return;
    }

    emit(VarBinding(new_var, new_value));
  }

  void VisitBinding_(const MatchShapeNode* binding) override {
    // If we have a trivial shape check (the shape_ of LHS and RHS is the same),
    // we can canonicalize to a var binding
    Expr new_value = this->VisitExpr(binding->value);

    Var new_var;
    // since we do not permit the checked_type to change and don't make any changes
    // to the shape pattern, there is no reason to do any more checking like in the
    // original mutator
    if (binding->var.defined()) {
      new_var = this->VisitVarDef(binding->var);
    }

    // if the LHS and RHS have the same shape_, we canonicalize to a var binding instead
    // TODO(tqchen, Hzfengsy): Since MatchShape will be removed, we just comment it out to pass the
    // compilation.
    // if (new_var.defined() && new_value->shape_.defined() &&
    //     builder_->CanProveShapeEqual(Downcast<Expr>(new_var->shape_),
    //                                  Downcast<Expr>(new_value->shape_))) {
    //   builder_->Emit(VarBinding(new_var, new_value));
    //   return;
    // }

    // reemit old binding if nothing changes
    if (new_value.same_as(binding->value)) {
      if (!binding->var.defined() || (binding->var.defined() && new_var.same_as(binding->var))) {
        builder_->EmitMatchShape(GetRef<MatchShape>(binding));
        return;
      }
    }

    builder_->EmitMatchShape(MatchShape(new_value, binding->pattern, new_var));
  }

 private:
  bool AnnotationsDiffer(const ObjectRef& obj1, const ObjectRef& obj2,
                         std::function<bool(const ObjectRef&, const ObjectRef&)> check_eq) {
    // annotations differ if one is present but not the other
    // or they're both present and they differ
    bool both_present = obj1.defined() && obj2.defined();
    bool neither_present = !obj1.defined() && !obj2.defined();
    return !(both_present || neither_present) || (both_present && !check_eq(obj1, obj2));
  }

  bool CanCanonicalizeVar(Var v) {
    Optional<Expr> value = LookupBinding(v);
    // can replace only if the value is also a var
    if (!value || !value.as<VarNode>()) {
      return false;
    }
    Var parent_var = Downcast<Var>(value);

    // Cases when we conservatively do not unify:
    // 1. checked_type_ or shape_ of the child differs from that of the parent
    //    In this case, we could be overriding user annotations.
    // 2. If the child is a Var and the parent is a DataflowVar.
    //    That could result in a DataflowVar leaving the current DataflowBlock.
    bool annotations_differ = AnnotationsDiffer(v->struct_info_, parent_var->struct_info_,
                                                [&](const ObjectRef& lhs, const ObjectRef& rhs) {
                                                  return tvm::StructuralEqual()(lhs, rhs);
                                                });
    bool var_to_dataflow = (!v.as<DataflowVarNode>() && parent_var.as<DataflowVarNode>());
    return !annotations_differ && !var_to_dataflow;
  }
};

Expr CanonicalizeBindings(const Expr& e) { return BindingCanonicalizer().VisitExpr(e); }

namespace transform {

Pass CanonicalizeBindings() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(CanonicalizeBindings(f));
      };
  return CreateFunctionPass(pass_func, 1, "CanonicalizeBindings", {});
}

TVM_REGISTER_GLOBAL("relax.transform.CanonicalizeBindings").set_body_typed(CanonicalizeBindings);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
