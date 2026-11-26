import json
import math
import struct

import pytest

import correctionlib._core as core
from correctionlib import schemav2 as schema


def test_access_formula_ast():
    """Test that we can access a Formula's AST from Python"""

    c = schema.Correction(
        name="test",
        version=2,
        inputs=[schema.Variable(name="x", type="real")],
        output=schema.Variable(name="a scale", type="real"),
        data=schema.Formula(
            nodetype="formula",
            expression="23.*log(x)",
            parser="TFormula",
            variables=["x"],
        ),
    )

    formula = core.Formula.from_string(
        c.data.model_dump_json(),
        [core.Variable.from_string(v.model_dump_json()) for v in c.inputs],
    )
    ast = formula.ast

    assert ast.nodetype == core.FormulaAst.NodeType.BINARY
    assert ast.data == core.FormulaAst.BinaryOp.TIMES
    assert ast.children[0].nodetype == core.FormulaAst.NodeType.LITERAL
    assert ast.children[0].data == 23.0
    assert ast.children[1].nodetype == core.FormulaAst.NodeType.UNARY
    assert ast.children[1].data == core.FormulaAst.UnaryOp.LOG
    assert ast.children[1].children[0].nodetype == core.FormulaAst.NodeType.VARIABLE
    # the index of variable x in the inputs
    assert ast.children[1].children[0].data == 0


def test_repeated_formula_rebinds_inputs_and_parameters():
    def load(variable, parameters, expression="x+[0]"):
        item = {
            "name": "corr",
            "version": 1,
            "inputs": [{"name": name, "type": "real"} for name in ("pt", "eta")],
            "output": {"name": "out", "type": "real"},
            "data": {
                "nodetype": "formula",
                "parser": "TFormula",
                "expression": expression,
                "variables": [variable],
                "parameters": parameters,
            },
        }
        return core.CorrectionSet.from_string(
            json.dumps({"schema_version": 2, "corrections": [item]})
        )["corr"]

    first = load("pt", [4.0])
    second = load("eta", [7.0])
    with pytest.raises(RuntimeError, match="Insufficient parameters"):
        load("pt", [])
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Failed to parse"):
            load("pt", [], "?")
    third = load("eta", [-3.0])
    assert [corr.evaluate(10.0, 2.0) for corr in (first, second, third)] == [
        14.0, 9.0, -1.0
    ]


@pytest.mark.parametrize(("expression", "parameters"), [
    (
        "max(0.0001,((max(min(x,[1]),[0])<[12])*([11]))+((max(min(x,[1]),[0])>=[12])*"
        "([2]+([3]/(pow(log10(max(min(x,[1]),[0])),2)+[4]))+"
        "([5]*exp(-([6]*((log10(max(min(x,[1]),[0]))-[7])*(log10(max(min(x,[1]),[0]))-[7])))))+"
        "([8]*exp(-([9]*((log10(max(min(x,[1]),[0]))-[10])*(log10(max(min(x,[1]),[0]))-[10]))))))))",
        [2.9782283, 782.64795, 0.65412325, 4.7065787, 4.3917775, -0.10357406,
         4.401645, 1.2357429, -0.67566425, 33.83841, -6.700246, 1.4943192, 8.0],
    ),
    (
        "sqrt([2]*abs([2])/(max(min(x,[1]),[0])*max(min(x,[1]),[0]))+"
        "[3]*[3]*pow(max(min(x,[1]),[0]),[5])+[4]*[4])",
        [15., 1500., -1.27, .82, .027, -.54],
    ),
])
def test_bound_jet_formula_matches_generic_ast(expression, parameters):
    inputs = [{"name": n, "type": "real"} for n in ("rho", "pt")]
    base = {"version": 1, "inputs": inputs, "output": {"name": "out", "type": "real"}}
    formula = {"nodetype": "formula", "expression": expression,
               "parser": "TFormula", "variables": ["pt"]}
    bound = dict(base, name="bound", data=dict(formula, parameters=parameters))
    reference = dict(base, name="reference", generic_formulas=[formula],
                     data={"nodetype": "formularef", "index": 0, "parameters": parameters})
    cset = core.CorrectionSet.from_string(json.dumps(
        {"schema_version": 2, "corrections": [bound, reference]}))
    # The L2 payload differs by one ULP at 57.21875 if pow(log10(pt),2)
    # is strength-reduced to multiplication. Also exercise both clamp edges
    # and both sides of its piecewise threshold.
    values = [57.21875, -math.inf, math.inf, math.nan]
    for cut in (parameters[0], parameters[1], 8.0):
        values.extend([math.nextafter(cut, -math.inf), cut, math.nextafter(cut, math.inf)])
    for pt in values:
        expected = cset["reference"].evaluate(24., pt)
        actual = cset["bound"].evaluate(24., pt)
        assert struct.pack("d", actual) == struct.pack("d", expected)
