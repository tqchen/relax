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

"""Tests analysis functions of struct info"""

import tvm
import tvm.testing
from tvm import relax as rx
from tvm import tir


def test_get_static_type_basic():
    # object
    s0 = rx.ObjectStructInfo()
    tvm.ir.assert_structural_equal(rx.analysis.get_static_type(s0), rx.ObjectType())

    # prim
    s1 = rx.PrimStructInfo("float32")
    tvm.ir.assert_structural_equal(rx.analysis.get_static_type(s1), tvm.ir.PrimType("float32"))


def test_get_static_type_shape():
    # shape
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")

    s2 = rx.ShapeStructInfo([1, n + 1, m])
    s3 = rx.ShapeStructInfo(ndim=2)

    tvm.ir.assert_structural_equal(rx.analysis.get_static_type(s2), rx.ShapeType(ndim=3))

    tvm.ir.assert_structural_equal(rx.analysis.get_static_type(s3), rx.ShapeType(ndim=2))


def test_get_static_type_tensor():
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    s4 = rx.TensorStructInfo([1, n + 1, m], "int64")

    tvm.ir.assert_structural_equal(
        rx.analysis.get_static_type(s4), rx.DynTensorType(ndim=3, dtype="int64")
    )


def test_get_static_type_tuple():
    # tuple
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    s0 = rx.ObjectStructInfo()
    s2 = rx.ShapeStructInfo([1, n + 1, m])
    s4 = rx.TensorStructInfo([1, n + 1, m], "int64")
    t0 = rx.TupleStructInfo([s4, s0])
    t1 = rx.TupleStructInfo([t0, s2])

    tvm.ir.assert_structural_equal(
        rx.analysis.get_static_type(t1),
        rx.TupleType(
            [
                rx.TupleType([rx.DynTensorType(ndim=3, dtype="int64"), rx.ObjectType()]),
                rx.ShapeType(ndim=3),
            ]
        ),
    )


def test_get_static_type_func():
    # tuple
    def fn_info(c):
        n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
        x = rx.TensorStructInfo([c, n, m], "float32")
        y = rx.TensorStructInfo([c, n, 1], "float32")
        z = rx.TensorStructInfo([c, n], "float32")
        return rx.FuncStructInfo([x, y], z)

    def fn_type():
        x = rx.DynTensorType(ndim=3, dtype="float32")
        y = rx.DynTensorType(ndim=3, dtype="float32")
        z = rx.DynTensorType(ndim=2, dtype="float32")
        return rx.FuncType([x, y], z)

    f0 = fn_info(1)

    tvm.ir.assert_structural_equal(rx.analysis.get_static_type(fn_info(1)), fn_type())


def test_erase_to_well_defined_basic():
    s0 = rx.ObjectStructInfo()
    tvm.ir.assert_structural_equal(rx.analysis.erase_to_well_defined(s0), s0)

    # prim
    s1 = rx.PrimStructInfo("float32")
    tvm.ir.assert_structural_equal(rx.analysis.erase_to_well_defined(s1), s1)


def test_erase_to_well_defined_shape():
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")

    s2 = rx.ShapeStructInfo([1, n + 1, m])
    s3 = rx.ShapeStructInfo(ndim=2)
    # have undefined
    tvm.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(s2), rx.ShapeStructInfo(ndim=3)
    )
    # all defined
    tvm.ir.assert_structural_equal(rx.analysis.erase_to_well_defined(s2, [n, m]), s2)

    # partial defined
    tvm.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(s2, [n]), rx.ShapeStructInfo(ndim=3)
    )


def test_erase_to_well_defined_tensor():
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    rshape = rx.Var("shape", type_annotation=rx.ShapeType(ndim=2))
    s0 = rx.TensorStructInfo(rshape, dtype="int32")

    # undefined
    tvm.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(s0, [], []), rx.TensorStructInfo(ndim=2, dtype="int32")
    )

    # defined
    tvm.ir.assert_structural_equal(rx.analysis.erase_to_well_defined(s0, [], [rshape]), s0)

    s1 = rx.TensorStructInfo([m + 1, n], dtype="float32")

    tvm.ir.assert_structural_equal(rx.analysis.erase_to_well_defined(s1, [n, m], [rshape]), s1)

    tvm.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(s1, [m], [rshape]),
        rx.TensorStructInfo(ndim=2, dtype="float32"),
    )

    s2 = rx.TensorStructInfo([1, 2], dtype="float32")

    tvm.ir.assert_structural_equal(rx.analysis.erase_to_well_defined(s2, [], []), s2)


def test_erase_to_well_defined_tuple():
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    s0 = rx.ObjectStructInfo()
    s2 = rx.ShapeStructInfo([1, m])
    s4 = rx.TensorStructInfo([1, n + 1, m], "int64")
    t0 = rx.TupleStructInfo([s4, s0])
    t1 = rx.TupleStructInfo([t0, s2])

    tvm.ir.assert_structural_equal(
        rx.analysis.erase_to_well_defined(t1, [m]),
        rx.TupleStructInfo(
            [
                rx.TupleStructInfo(
                    [rx.TensorStructInfo(ndim=3, dtype="int64"), rx.ObjectStructInfo()]
                ),
                rx.ShapeStructInfo([1, m]),
            ]
        ),
    )


def test_erase_to_well_defined_func():
    def fn_info(c):
        n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
        x = rx.TensorStructInfo([c, n, m], "float32")
        y = rx.TensorStructInfo([c, n, 1], "float32")
        z = rx.TensorStructInfo([c, n], "float32")
        return rx.FuncStructInfo([x, y], z)

    f0 = fn_info(1)

    tvm.ir.assert_structural_equal(rx.analysis.erase_to_well_defined(f0), f0)


def test_is_base_of():
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    obj0 = rx.ObjectStructInfo()
    prim0 = rx.PrimStructInfo("int32")
    prim1 = rx.PrimStructInfo("float32")

    shape0 = rx.ShapeStructInfo(ndim=-1)
    shape1 = rx.ShapeStructInfo(ndim=2)
    shape2 = rx.ShapeStructInfo(ndim=3)
    shape3 = rx.ShapeStructInfo([1, 2, 3])
    shape4 = rx.ShapeStructInfo([1, n, 3])

    tensor0 = rx.TensorStructInfo(ndim=-1, dtype="int32")
    tensor1 = rx.TensorStructInfo(ndim=-1, dtype="float32")
    tensor2 = rx.TensorStructInfo(ndim=2, dtype="int32")
    tensor3 = rx.TensorStructInfo(ndim=2, dtype="float32")
    tensor4 = rx.TensorStructInfo([n, m], "int32")
    tensor5 = rx.TensorStructInfo([n, m, 1], "int32")
    tensor6 = rx.TensorStructInfo([n, m, 2], "int32")

    # obj
    assert obj0.is_base_of(prim0)
    assert obj0.is_base_of(shape1)
    assert obj0.is_base_of(tensor2)

    # prim
    assert not prim0.is_base_of(obj0)
    assert prim0.is_base_of(prim0)
    assert not prim0.is_base_of(prim1)

    # shape
    assert not shape0.is_base_of(obj0)
    assert not shape0.is_base_of(prim0)
    # unknown dim
    assert shape0.is_base_of(shape1)
    # ndim mismatch
    assert not shape1.is_base_of(shape2)
    # lhs do not have symbolic value but ndim match
    assert shape2.is_base_of(shape3)
    # rhs do not symbolic but lhs do
    assert not shape3.is_base_of(shape2)
    # shape mismatch
    assert not shape3.is_base_of(shape4)
    assert shape4.is_base_of(rx.ShapeStructInfo([1, n, 3]))

    # tensor
    assert not tensor0.is_base_of(obj0)
    assert not tensor0.is_base_of(prim0)
    assert not tensor0.is_base_of(shape0)

    # dtype mismatch
    assert not tensor0.is_base_of(tensor1)
    assert not tensor0.is_base_of(tensor3)
    assert not tensor3.is_base_of(tensor4)
    assert not tensor1.is_base_of(tensor2)

    # ndim mismatch
    assert not tensor2.is_base_of(tensor5)

    # shape mismatch
    assert not tensor5.is_base_of(tensor6)

    # match
    assert tensor0.is_base_of(rx.TensorStructInfo(ndim=-1, dtype="int32"))
    assert tensor0.is_base_of(tensor2)
    assert tensor0.is_base_of(tensor4)
    assert tensor0.is_base_of(tensor5)
    assert tensor0.is_base_of(tensor6)
    assert tensor2.is_base_of(tensor4)
    assert tensor4.is_base_of(rx.TensorStructInfo([n, m], dtype="int32"))

    # tuple
    t0 = rx.TupleStructInfo([obj0, tensor0])
    t1 = rx.TupleStructInfo([prim0, tensor4])
    t2 = rx.TupleStructInfo([obj0, tensor0, obj0])
    t3 = rx.TupleStructInfo([tensor0, obj0])

    assert t0.is_base_of(t1)
    assert not t0.is_base_of(t2)
    assert not t0.is_base_of(t3)

    assert rx.TupleStructInfo([t0, t1]).is_base_of(rx.TupleStructInfo([t1, t1]))

    assert not rx.TupleStructInfo([t0, t1]).is_base_of(rx.TupleStructInfo([t1, t0]))

    def fn_info_shape(c):
        n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
        x = rx.TensorStructInfo([c, n, m], "float32")
        y = rx.TensorStructInfo([c, n, 1], "float32")
        z = rx.TensorStructInfo([c, n], "float32")
        return rx.FuncStructInfo([x, y], z)

    def fn_info_erased():
        x = rx.TensorStructInfo(ndim=3, dtype="float32")
        y = rx.TensorStructInfo(ndim=3, dtype="float32")
        z = rx.TensorStructInfo(ndim=2, dtype="float32")
        return rx.FuncStructInfo([x, y], z)

    assert fn_info_shape(1).is_base_of(fn_info_shape(1))
    assert fn_info_erased().is_base_of(fn_info_shape(1))
    assert not fn_info_shape(1).is_base_of(fn_info_erased())

    fopaque = rx.FuncStructInfo.opaque_func()
    assert fopaque.is_base_of(fn_info_shape(1))


def _check_lca(lhs, rhs, target):
    tvm.ir.assert_structural_equal(rx.analysis.struct_info_lca(lhs, rhs), target)
    tvm.ir.assert_structural_equal(rx.analysis.struct_info_lca(rhs, lhs), target)


def test_struct_info_lca():
    n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
    obj0 = rx.ObjectStructInfo()
    prim0 = rx.PrimStructInfo("int32")
    prim1 = rx.PrimStructInfo("float32")

    shape0 = rx.ShapeStructInfo(ndim=-1)
    shape1 = rx.ShapeStructInfo(ndim=2)
    shape2 = rx.ShapeStructInfo(ndim=3)
    shape3 = rx.ShapeStructInfo([1, 2, 3])
    shape4 = rx.ShapeStructInfo([1, n, 3])

    tensor0 = rx.TensorStructInfo(ndim=-1, dtype="int32")
    tensor1 = rx.TensorStructInfo(ndim=-1, dtype="float32")
    tensor2 = rx.TensorStructInfo(ndim=2, dtype="int32")
    tensor3 = rx.TensorStructInfo(ndim=2, dtype="float32")
    tensor4 = rx.TensorStructInfo([n, m], "int32")
    tensor5 = rx.TensorStructInfo([n, m, 1], "int32")
    tensor6 = rx.TensorStructInfo([n, m, 2], "int32")

    # obj
    _check_lca(obj0, prim0, obj0)
    _check_lca(obj0, prim1, obj0)

    # shape
    _check_lca(shape0, tensor0, obj0)
    _check_lca(shape0, shape1, shape0)
    _check_lca(shape1, shape2, shape0)
    _check_lca(shape1, shape3, shape0)

    _check_lca(shape2, shape3, shape2)
    _check_lca(shape3, shape4, shape2)
    _check_lca(shape4, rx.ShapeStructInfo([1, n, 3]), shape4)

    # tensor
    _check_lca(tensor0, prim0, obj0)
    _check_lca(tensor0, tensor1, rx.TensorStructInfo(ndim=-1, dtype=None))
    _check_lca(tensor0, tensor2, tensor0)
    _check_lca(tensor0, tensor4, tensor0)

    _check_lca(tensor2, tensor4, tensor2)
    _check_lca(tensor5, tensor6, rx.TensorStructInfo(ndim=3, dtype="int32"))
    _check_lca(tensor4, tensor5, rx.TensorStructInfo(ndim=-1, dtype="int32"))
    _check_lca(tensor4, rx.TensorStructInfo([n, m], dtype="int32"), tensor4)

    # tuple
    t0 = rx.TupleStructInfo([obj0, tensor0])
    t1 = rx.TupleStructInfo([prim0, tensor4])
    t2 = rx.TupleStructInfo([obj0, tensor0, obj0])
    t3 = rx.TupleStructInfo([tensor0, obj0])

    _check_lca(t0, t1, t0)
    _check_lca(t0, t2, obj0)
    _check_lca(t0, t3, rx.TupleStructInfo([obj0, obj0]))

    t5 = rx.TupleStructInfo([t0, t1])
    t6 = rx.TupleStructInfo([t1, t2])

    _check_lca(t5, t6, rx.TupleStructInfo([t0, obj0]))

    t7 = rx.TupleStructInfo([])
    _check_lca(t7, rx.TupleStructInfo([]), t7)

    def fn_info_shape(c):
        n, m = tir.Var("n", "int64"), tir.Var("m", "int64")
        x = rx.TensorStructInfo([c, n, m], "float32")
        y = rx.TensorStructInfo([c, n, 1], "float32")
        z = rx.TensorStructInfo([c, n], "float32")
        return rx.FuncStructInfo([x, y], z)

    def fn_info_erased():
        x = rx.TensorStructInfo(ndim=3, dtype="float32")
        y = rx.TensorStructInfo(ndim=3, dtype="float32")
        z = rx.TensorStructInfo(ndim=2, dtype="float32")
        return rx.FuncStructInfo([x, y], z)

    fopaque0 = lambda: rx.FuncStructInfo.opaque_func()
    fopaque1 = lambda: rx.FuncStructInfo.opaque_func(ret=prim0)
    fopaque2 = lambda: rx.FuncStructInfo.opaque_func(
        ret=rx.TensorStructInfo(ndim=2, dtype="float32")
    )

    _check_lca(fn_info_shape(1), fn_info_shape(2), fn_info_erased())
    _check_lca(fn_info_shape(2), fn_info_shape(2), fn_info_shape(2))

    _check_lca(fopaque0(), fopaque1(), fopaque0())
    _check_lca(fopaque0(), fn_info_shape(1), fopaque0())
    _check_lca(fopaque2(), fn_info_shape(1), fopaque2())


if __name__ == "__main__":
    tvm.testing.main()
