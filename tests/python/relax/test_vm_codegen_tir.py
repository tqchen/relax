# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Test last-stage of codegen VM.

Restrictions: all shape lowered, explicit allocation.
"""
import tvm
from tvm import relax
from tvm.script import relax as R, tir as T
from tvm.ir.base import assert_structural_equal


def get_tir_mod(mod):
    builder = relax.ExecBuilder()
    return relax.vm._vmcodegen(builder, mod, exec_mode="compiled")


def test_add():
    @tvm.script.ir_module
    class Before:
        @R.function
        def foo(x: R.Tensor):
            R.func_attr({"global_symbol": "foo"})
            z = R.call_packed("test.vm.add", x, x, type_args=(R.Tensor))
            return z

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def __vmtir__foo(ctx_ptr: T.handle, r: T.handle, c: T.handle, f: T.handle):
            T.func_attr({"global_symbol": "__vmtir__foo"})
            T.anylist_setitem_call_packed(
                r, T.int64(2), "test.vm.add",
                T.anylist_getitem(r, T.int64(0)),
                T.anylist_getitem(r, T.int64(0)))
            T.anylist_setitem_call_packed(
                r, T.int64(1), "vm.builtin.copy",
                T.anylist_getitem(r, T.int64(2)))

    before = Before
    expected = Expected
    after = get_tir_mod(before)
    assert_structural_equal(expected, after)


def test_if_cond():
    @tvm.script.ir_module
    class Before:
        @R.function
        def ife(cond: R.Tensor((), "bool"), x: R.Tensor) -> R.Tensor:
            R.func_attr({"global_symbol": "ife"})
            if cond:
                w = R.call_packed("test.vm.add", x, x, type_args=(R.Tensor))
            else:
                w = R.call_packed("test.vm.mul", x, x, type_args=(R.Tensor))
            return w

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def __vmtir__ife(ctx_ptr: T.handle, r: T.handle, c: T.handle, f: T.handle):
            T.func_attr({"global_symbol": "__vmtir__ife"})
            if T.anylist_getitem(r, T.int64(0)):
                T.anylist_setitem_call_packed(
                    r, T.int64(4),
                    "test.vm.add",
                    T.anylist_getitem(r, T.int64(1)),
                    T.anylist_getitem(r, T.int64(1)))
                T.anylist_setitem_call_packed(
                    r, T.int64(3),
                    "vm.builtin.copy",
                    T.anylist_getitem(r, T.int64(4)))
            else:
                T.anylist_setitem_call_packed(
                    r, T.int64(5),
                    "test.vm.mul",
                    T.anylist_getitem(r, T.int64(1)),
                    T.anylist_getitem(r, T.int64(1)))
                T.anylist_setitem_call_packed(
                    r, T.int64(3),
                    "vm.builtin.copy",
                    T.anylist_getitem(r, T.int64(5)))
            T.anylist_setitem_call_packed(
                r, T.int64(2), "vm.builtin.copy",
                T.anylist_getitem(r, T.int64(3)))

    before = Before
    expected = Expected
    after = get_tir_mod(before)
    assert_structural_equal(expected, after)


def test_const():
    @tvm.script.ir_module
    class Before:
        @R.function
        def main(x: R.Tensor):
            R.func_attr({"global_symbol": "main"})
            y = R.const([1, 2])
            z = (y, R.const([3, 4]), x)
            return z

    @tvm.script.ir_module
    class Expected:
        @T.prim_func
        def __vmtir__main(ctx_ptr: T.handle, r: T.handle, c: T.handle, f: T.handle):
            # function attr dict
            T.func_attr({"global_symbol": "__vmtir__main"})
            # body
            T.anylist_setitem_call_packed(
                r, T.int64(2),
                "runtime.Tuple",
                T.anylist_getitem(c, T.int64(0)),
                T.anylist_getitem(c, T.int64(1)),
                T.anylist_getitem(r, T.int64(0)))
            T.anylist_setitem_call_packed(
                r, T.int64(1),
                "vm.builtin.copy",
                T.anylist_getitem(r, T.int64(2)))
    before  = Before
    expected = Expected
    after = get_tir_mod(before)
    assert_structural_equal(expected, after)
