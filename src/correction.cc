#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
#include <rapidjson/document.h>
#pragma GCC diagnostic pop
#include <rapidjson/filereadstream.h>
#include <rapidjson/error/en.h>
#include <optional>
#include <deque>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <cstdlib> // std::abort
#include <random>
#include "correction.h"
#define XXH_INLINE_ALL 1
#include "xxhash.h"
#include "pcg_random.hpp"
#if __has_include(<zlib.h>)
#include <zlib.h>
#include "gzfilereadstream.h"
#define WITH_ZLIB 1
#endif

using namespace correction;

//! helper function for parsing flow behavior from string
_FlowBehavior parse_flowbehavior(const rapidjson::Value& flowbehavior) {
  if ( flowbehavior == "clamp" ) {
    return _FlowBehavior::clamp;
  }
  else if ( flowbehavior == "error" ) {
    return _FlowBehavior::error;
  }
  else if ( flowbehavior == "wrap" ) {
    return _FlowBehavior::wrap;
  }
  else {
    return _FlowBehavior::value;
  }
}

class correction::JSONObject {
  public:
    JSONObject(rapidjson::Value::ConstObject&& json) : json_(json) { }
    // necessary to force use of const Value::GetObject() method
    // must check if json is an object in calling code!
    JSONObject(const rapidjson::Document& json) : json_(json.GetObject()) { }

    template<typename T>
    T getRequired(const char * key) const {
      const auto it = json_.FindMember(key);
      if ( it != json_.MemberEnd() ) {
        if ( it->value.template Is<T>() ) {
          return it->value.template Get<T>();
        } else {
          throw std::runtime_error(
              "Encountered invalid type for required attribute '"
              + std::string(key) + "'");
        }
      }
      throw std::runtime_error(
          "Object missing required attribute '"
          + std::string(key) + "'");
    }

    const rapidjson::Value& getRequiredValue(const char * key) const {
      const auto it = json_.FindMember(key);
      if ( it != json_.MemberEnd() ) {
        return it->value;
      }
      throw std::runtime_error(
          "Object missing required attribute '"
          + std::string(key) + "'");
    }

    template<typename T>
    std::optional<T> getOptional(const char * key) const {
      const auto it = json_.FindMember(key);
      if ( it != json_.MemberEnd() ) {
        if ( it->value.template Is<T>() ) {
          return it->value.template Get<T>();
        } else if ( it->value.IsNull() ) {
          return std::nullopt;
        } else {
          throw std::runtime_error(
              "Encountered invalid type for optional attribute '"
              + std::string(key) + "'");
        }
      }
      return std::nullopt;
    }

    // escape hatch
    inline auto FindMember(const char * key) const { return json_.FindMember(key); }
    inline auto MemberEnd() const { return json_.MemberEnd(); }

  private:
    rapidjson::Value::ConstObject json_;
};

template<>
std::string_view JSONObject::getRequired<std::string_view>(const char * key) const {
  const auto it = json_.FindMember(key);
  if ( it != json_.MemberEnd() ) {
    if ( it->value.IsString() ) {
      return std::string_view(it->value.GetString(), it->value.GetStringLength());
    } else {
      throw std::runtime_error(
          "Encountered invalid type for required attribute '"
          + std::string(key) + "'");
    }
  }
  throw std::runtime_error(
      "Object missing required attribute '"
      + std::string(key) + "'");
}

namespace {
  Content resolve_content(const rapidjson::Value& json, const Correction& context) {
    if ( json.IsNumber() ) { return json.GetDouble(); }
    else if ( json.IsObject() && json.HasMember("nodetype") ) {
      auto obj = JSONObject(json.GetObject());
      auto type = obj.getRequired<std::string_view>("nodetype");
      if ( type == "binning" ) { return Binning(obj, context); }
      else if ( type == "multibinning" ) { return MultiBinning(obj, context); }
      else if ( type == "category" ) { return Category(obj, context); }
      else if ( type == "formula" ) { return Formula(obj, context); }
      else if ( type == "formularef" ) { return FormulaRef(obj, context); }
      else if ( type == "transform" ) { return Transform(obj, context); }
      else if ( type == "hashprng" ) { return HashPRNG(obj, context); }
      else { throw std::runtime_error("Unrecognized Content object nodetype"); }
    }
    throw std::runtime_error("Invalid Content node type");
  }

  struct node_evaluate {
    double operator() (double node) { return node; }

    template <class Node>
    double operator() (const Node &node) {
      return node.evaluate(values);
    }

    const std::vector<Variable::Type>& values;
  };

  // Per-thread scratch storage for Transform::evaluate.
  // Depth indexing keeps nested Transform evaluations re-entrant-safe.
  class TransformScratch {
  public:
    explicit TransformScratch(const std::vector<Variable::Type>& values):
      slot_(acquire_slot())
    {
      slot_ = values;
      ++depth_;
    }

    ~TransformScratch() {
      release_slot();
    }

    std::vector<Variable::Type>& values() {
      return slot_;
    }

  private:
    static std::vector<Variable::Type>& acquire_slot() {
      if (depth_ == slots_.size()) {
        slots_.emplace_back();
      }
      return slots_[depth_];
    }

    static void release_slot() {
      depth_--;
    }

    std::vector<Variable::Type>& slot_;
    static thread_local std::deque<std::vector<Variable::Type>> slots_;
    static thread_local std::size_t depth_;
  };

  thread_local std::deque<std::vector<Variable::Type>> TransformScratch::slots_;
  thread_local std::size_t TransformScratch::depth_ = 0;

  double clampedAffine(double x, const double* p) {
    // Keep the multiply and addition independently rounded as in FormulaAst.
    volatile double product = p[0] * std::max(std::min(x, p[3]), p[2]);
    return product + p[1];
  }

  // The reference AST rounds each operation to double and uses libm pow.
  // Enable the shortcuts on the validated GNU/Linux target. In particular,
  // a target with FMA could contract an affine term across AST boundaries.
#if defined(__x86_64__) && defined(__GLIBC__) && defined(__GCC_IEC_559) && __GCC_IEC_559 >= 2 && \
    defined(__FLT_EVAL_METHOD__) && __FLT_EVAL_METHOD__ == 0 && !defined(__FAST_MATH__) && \
    !defined(__FMA__) && !defined(__FMA4__)
  constexpr bool useBoundJetFormulas = true;
#else
  constexpr bool useBoundJetFormulas = false;
#endif

  constexpr std::string_view l2RelativeExpression =
      "max(0.0001,((max(min(x,[1]),[0])<[12])*([11]))+((max(min(x,[1]),[0])>=[12])*"
      "([2]+([3]/(pow(log10(max(min(x,[1]),[0])),2)+[4]))+"
      "([5]*exp(-([6]*((log10(max(min(x,[1]),[0]))-[7])*(log10(max(min(x,[1]),[0]))-[7])))))+"
      "([8]*exp(-([9]*((log10(max(min(x,[1]),[0]))-[10])*(log10(max(min(x,[1]),[0]))-[10]))))))))";
  constexpr std::string_view ptResolutionExpression =
      "sqrt([2]*abs([2])/(max(min(x,[1]),[0])*max(min(x,[1]),[0]))+"
      "[3]*[3]*pow(max(min(x,[1]),[0]),[5])+[4]*[4])";

  double evaluateL2Relative(double pt, const std::vector<double>& p) {
    const double logPt = std::log10(pt);
    const double firstOffset = logPt - p[7];
    const double secondOffset = logPt - p[10];
    // Folding pow(logPt, 2) into multiplication changes some results by one ULP.
    const volatile double exponent = 2.;
    const double response = p[2] + p[3] / (std::pow(logPt, exponent) + p[4])
        + p[5] * std::exp(-(p[6] * (firstOffset * firstOffset)))
        + p[8] * std::exp(-(p[9] * (secondOffset * secondOffset)));
    return std::max(0.0001, (pt < p[12]) * p[11] + (pt >= p[12]) * response);
  }

  double evaluatePtResolution(double pt, const std::vector<double>& p) {
    return std::sqrt(p[2] * std::abs(p[2]) / (pt * pt)
                     + p[3] * p[3] * std::pow(pt, p[5]) + p[4] * p[4]);
  }

  std::size_t find_bin_idx(const Variable::Type& value_variant,
                           const std::variant<_UniformBins, _NonUniformBins> &bins_,
                           const _FlowBehavior &flow,
                           std::size_t variableIdx,
                           const char *name)
  {
    double value = std::visit([](auto&& arg) -> double {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, int>) return static_cast<double>(arg);
      else if constexpr (std::is_same_v<T, double>) return arg;
      else throw std::logic_error("I should not have ever seen a string");
      }, value_variant);
    if ( auto *bins = std::get_if<_UniformBins>(&bins_) ) { // uniform binning
      if (value < bins->low || value >= bins->high) {
        switch (flow) {
          case _FlowBehavior::value:
            return bins->n; // the default value is stored at the end of the content array, after the last bin
          case _FlowBehavior::clamp:
            return value < bins->low ? 0 : bins->n - 1; // assuming we always have at least 1 bin
          case _FlowBehavior::wrap:
            break;
          case _FlowBehavior::error:
            const std::string belowOrAbove = value < bins->low ? "below" : "above";
            auto msg = "Index " + belowOrAbove + " bounds in " + name + " for input argument " + std::to_string(variableIdx) + " value: " + std::to_string(value);
            throw std::runtime_error(std::move(msg));
        }
      }

      double norm_value = ((value - bins->low) / (bins->high - bins->low));
      if (flow == _FlowBehavior::wrap) {
        norm_value -= std::floor(norm_value);
      }
      std::size_t binIdx = bins->n * norm_value;
      return binIdx;
    }

    // otherwise we have non-uniform binning
    using namespace std::string_literals;
    const auto& bins = std::get<_NonUniformBins>(bins_);
    if ( flow == _FlowBehavior::wrap ) {
      double low = bins[0];
      double high = bins[bins.size() - 1];
      double norm_value = (value - low) / (high - low);
      norm_value -= std::floor(norm_value);
      value = low + norm_value * (high - low);
    }

    auto it = std::upper_bound(std::begin(bins), std::end(bins), value);
    if ( it == std::begin(bins) ) { // underflow
      if ( flow == _FlowBehavior::value ) {
        return bins.size() - 1; // the default value is stored at the end of the content array, after the last bin
      }
      else if ( flow == _FlowBehavior::error ) {
        throw std::runtime_error("Index below bounds in "s + name + " for input argument " + std::to_string(variableIdx) + " value: " + std::to_string(value));
      }
      else if ( flow == _FlowBehavior::wrap ) {
        throw std::logic_error("I should not have ever seen an underflow");
      }
      else { // clamp
        it++;
      }
    }
    else if ( it == std::end(bins) ) { // overflow
      if ( flow == _FlowBehavior::value ) {
        return bins.size() - 1;
      }
      else if ( flow == _FlowBehavior::error ) {
        throw std::runtime_error("Index above bounds in "s + name + " for input argument " + std::to_string(variableIdx) + " value: " + std::to_string(value));
      }
      else if ( flow == _FlowBehavior::wrap ) {
        throw std::logic_error("I should not have ever seen an overflow");
      }
      else { // clamp
        it--;
      }
    }

    // -1 because upper_bound returns the edge _after_ the bin we are interested in
    const std::size_t binIdx = std::distance(std::begin(bins), it) - 1;
    return binIdx;
  }

  size_t find_input_index(const std::string_view name, const std::vector<Variable> &inputs) {
    size_t idx = 0;
    for (const auto& var : inputs) {
      if ( name == var.name() ) return idx;
      idx++;
    }
    throw std::runtime_error("Error: could not find variable " + std::string(name) + " in inputs");
  }

  double parse_edge(const rapidjson::Value& edge) {
    if ( edge.IsDouble() ) {
      return edge.GetDouble();
    } else if ( edge.IsString() ) {
      std::string_view str = edge.GetString();
      if ((str == "inf") || (str == "+inf")) return std::numeric_limits<double>::infinity();
      else if (str == "-inf") return -std::numeric_limits<double>::infinity();
    }
    throw std::runtime_error("Invalid edge type");
  }

  std::vector<double> parse_bin_edges(const rapidjson::Value::ConstArray& edges) {
    std::vector<double> result;
    result.reserve(edges.Size());
    for (const auto& edge : edges) {
      double val = parse_edge(edge);
      if ( result.size() > 0 && result.back() >= val ) {
        throw std::runtime_error("binning edges are not monotone increasing");
      }
      result.push_back(val);
    }
    return result;
  }
} // end of anonymous namespace

Variable::Variable(const JSONObject& json) :
  name_(json.getRequired<const char *>("name")),
  description_(json.getOptional<const char*>("description").value_or(""))
{
  auto type = json.getRequired<std::string_view>("type");
  if (type == "string") { type_ = VarType::string; }
  else if (type == "int") { type_ = VarType::integer; }
  else if (type == "real") { type_ = VarType::real; }
  else { throw std::runtime_error("Unrecognized variable type"); }
}

std::string Variable::typeStr() const {
  if ( type_ == VarType::string ) { return "string"; }
  else if ( type_ == VarType::integer ) { return "int"; }
  else if ( type_ == VarType::real ) { return "real"; }
  return "";
}

void Variable::validate(const Type& t) const {
  if ( std::holds_alternative<std::string>(t) ) {
    if ( type_ != VarType::string ) {
      throw std::runtime_error("Input " + name() + " has wrong type: got string expected " + typeStr());
    }
  }
  else if ( std::holds_alternative<int>(t) ) {
    if ( type_ != VarType::integer ) {
      throw std::runtime_error("Input " + name() + " has wrong type: got int expected " + typeStr());
    }
  }
  else if ( std::holds_alternative<double>(t) ) {
    if ( type_ != VarType::real ) {
      throw std::runtime_error("Input " + name() + " has wrong type: got real-valued expected " + typeStr());
    }
  }
}

Variable Variable::from_string(const char * data) {
  rapidjson::Document json;
  rapidjson::ParseResult ok = json.Parse(data);
  if (!ok) {
    throw std::runtime_error(
        std::string("JSON parse error: ") + rapidjson::GetParseError_En(ok.Code())
        + " at offset " + std::to_string(ok.Offset())
        );
  }
  if ( ! json.IsObject() ) { throw std::runtime_error("Expected Variable object"); }
  return Variable(json);
}

Formula::Formula(const JSONObject& json, const Correction& context, bool generic)
  : Formula(json, context.inputs(), generic) {}

Formula::Formula(const JSONObject& json, const std::vector<Variable>& inputs, bool generic) :
  expression_(json.getRequired<const char *>("expression")),
  generic_(generic)
{
  auto parser_type = json.getRequired<std::string_view>("parser");
  if (parser_type == "TFormula") { type_ = FormulaAst::ParserType::TFormula; }
  else if (parser_type == "numexpr") {
    type_ = FormulaAst::ParserType::numexpr;
    throw std::runtime_error("numexpr formula parser is not yet supported");
  }
  else { throw std::runtime_error("Unrecognized formula parser type"); }

  std::vector<size_t> variableIdx;
  for (const auto& item : json.getRequired<rapidjson::Value::ConstArray>("variables")) {
    auto idx = find_input_index(item.GetString(), inputs);
    if ( inputs[idx].type() != Variable::VarType::real ) {
      throw std::runtime_error("Formulas only accept real-valued inputs, got type "
          + inputs[idx].typeStr() + " for variable " + inputs[idx].name());
    }
    variableIdx.push_back(idx);
  }

  std::vector<double> params;
  if ( auto items = json.getOptional<rapidjson::Value::ConstArray>("parameters") ) {
    for (const auto& item : *items) {
      params.push_back(item.GetDouble());
    }
  }

  if (generic && variableIdx.size() == 1 && expression_ == "[0]*max(min(x,[3]),[2])+[1]")
    affineVariable_ = static_cast<int>(variableIdx.front());
  body_.reset(new Body{FormulaAst::parse(type_, expression_, params, variableIdx, !generic), {}});
  if (useBoundJetFormulas && !generic && variableIdx.size() == 1) {
    if (params.size() == 13 && expression_ == l2RelativeExpression)
      body_->jetFormula = JetFormula::L2Relative;
    else if (params.size() == 6 && expression_ == ptResolutionExpression)
      body_->jetFormula = JetFormula::PtResolution;
    if (body_->jetFormula != JetFormula::None) {
      body_->jetPtIndex = variableIdx.front();
      body_->jetParameters = std::move(params);
    }
  }
}

Formula::Ref Formula::from_string(const char * data, std::vector<Variable>& inputs) {
  rapidjson::Document json;
  rapidjson::ParseResult ok = json.Parse(data);
  if (!ok) {
    throw std::runtime_error(
        std::string("JSON parse error: ") + rapidjson::GetParseError_En(ok.Code())
        + " at offset " + std::to_string(ok.Offset())
        );
  }
  if ( ! json.IsObject() ) { throw std::runtime_error("Expected Formula object"); }
  return std::make_shared<Formula>(json, inputs);
}

double Formula::evaluate(const std::vector<Variable::Type>& values) const {
  if ( generic_ ) {
    throw std::runtime_error("Generic formulas must be evaluated with parameters");
  }
  if (useBoundJetFormulas && body_->jetFormula != JetFormula::None) {
    const auto& p = body_->jetParameters;
    const double pt = std::max(std::min(std::get<double>(values[body_->jetPtIndex]), p[1]), p[0]);
    return body_->jetFormula == JetFormula::L2Relative
        ? evaluateL2Relative(pt, p) : evaluatePtResolution(pt, p);
  }
  return body_->ast.evaluate(values, {});
}

double Formula::evaluate(const std::vector<Variable::Type>& values, const std::vector<double>& params) const {
  if (affineVariable_ >= 0 && params.size() >= 4)
    return clampedAffine(std::get<double>(values[affineVariable_]), params.data());
  return body_->ast.evaluate(values, params);
}

FormulaRef::FormulaRef(const JSONObject& json, const Correction& context) {
  formula_ = context.formula_ref(json.getRequired<int>("index"));
  for (const auto& item : json.getRequired<rapidjson::Value::ConstArray>("parameters")) {
    parameters_.push_back(item.GetDouble());
  }
}

double FormulaRef::evaluate(const std::vector<Variable::Type>& values) const {
  return formula_->evaluate(values, parameters_);
}

Transform::Transform(const JSONObject& json, const Correction& context) {
  variableIdx_ = find_input_index(json.getRequired<std::string_view>("input"), context.inputs());
  const auto& variable = context.inputs()[variableIdx_];
  if ( variable.type() == Variable::VarType::string ) {
    throw std::runtime_error("Transform cannot rewrite string inputs");
  }
  rule_ = std::make_unique<Content>(resolve_content(json.getRequiredValue("rule"), context));
  content_ = std::make_unique<Content>(resolve_content(json.getRequiredValue("content"), context));
}

double Transform::evaluate(const std::vector<Variable::Type>& values) const {
  TransformScratch scratch(values);
  auto& new_values = scratch.values();
  double vnew = std::visit(node_evaluate{values}, *rule_);
  auto& v = new_values[variableIdx_];
  if ( std::holds_alternative<double>(v) ) {
    v = vnew;
  }
  else if ( std::holds_alternative<int>(v) ) {
    v = (int) std::round(vnew);
  }
  else {
    throw std::logic_error("I should not have ever seen a string");
  }
  return std::visit(node_evaluate{new_values}, *content_);
}

HashPRNG::HashPRNG(const JSONObject& json, const Correction& context)
{
  const auto& inputs = json.getRequired<rapidjson::Value::ConstArray>("inputs");
  variablesIdx_.reserve(inputs.Size());
  for (const auto& input : inputs) {
    if ( ! input.IsString() ) { throw std::runtime_error("invalid hashprng input type"); }
    size_t idx = find_input_index(input.GetString(), context.inputs());
    if ( context.inputs().at(idx).type() == Variable::VarType::string ) {
      throw std::runtime_error("HashPRNG cannot use string inputs as entropy sources");
    }
    variablesIdx_.push_back(idx);
  }
  auto dist = json.getRequired<std::string_view>("distribution");
  if (dist == "stdflat") { dist_ = Distribution::stdflat; }
  else if (dist == "stdnormal") { dist_ = Distribution::stdnormal; }
  else if (dist == "normal") { dist_ = Distribution::normal; }
  else { throw std::runtime_error("invalid distribution type for hashprng"); }

}

double HashPRNG::evaluate(const std::vector<Variable::Type>& values) const {
  pcg32_oneseq gen;
  size_t nbytes = sizeof(uint64_t)*variablesIdx_.size();
  uint64_t* seedData = (uint64_t*) alloca(nbytes);
  for(size_t i=0; i<variablesIdx_.size(); ++i) {
    if ( auto v = std::get_if<int>(&values[variablesIdx_[i]]) ) {
      seedData[i] = static_cast<uint64_t>(*v);
    }
    else if ( auto v = std::get_if<double>(&values[variablesIdx_[i]]) ) {
      seedData[i] = *reinterpret_cast<const uint64_t*>(v);
    }
    else { throw std::logic_error("I should not have ever seen a string"); }
  }
  gen.seed(XXH64((const void*) seedData, nbytes, 0ul));
  switch (dist_) {
    case Distribution::stdflat:
      return std::uniform_real_distribution<>()(gen);
    case Distribution::stdnormal:
      // if not a temporary, it may reuse the spare value
      return std::normal_distribution<>()(gen);
    case Distribution::normal:
      double u, v, s;
      do {
        // cheap but wrong-ish https://www.pcg-random.org/using-pcg-c.html#generating-doubles
        u = std::ldexp(gen(), -31) - 1;
        v = std::ldexp(gen(), -31) - 1;
        s = u*u + v*v;
      } while ( s>= 1.0 || s == 0.0 );
      return u * std::sqrt(-2.0 * std::log(s) / s);
    default:
      std::abort(); // never reached if the switch is exhaustive
  };
}

Binning::Binning(const JSONObject& json, const Correction& context)
{
  const auto& content = json.getRequired<rapidjson::Value::ConstArray>("content");

  // set bins_
  const auto &edgesObj = json.getRequiredValue("edges");
  if ( edgesObj.IsArray() ) { // non-uniform binning
    std::vector<double> edges = parse_bin_edges(edgesObj.GetArray());
    if ( edges.size() != content.Size() + 1 ) {
      throw std::runtime_error("Inconsistency in Binning: number of content nodes does not match binning");
    }
    _NonUniformBins bins{std::move(edges)};
    bins_ = std::move(bins);
  } else if ( edgesObj.IsObject() ) { // UniformBinning
    const JSONObject uniformBins{edgesObj.GetObject()};
    const auto n = uniformBins.getRequired<uint32_t>("n");
    if ( n == 0 ) {
      throw std::runtime_error("Error when processing Binning with UniformBinning: number of bins is zero");
    }
    if ( n != content.Size() ) {
      throw std::runtime_error("Inconsistency in Binning: number of content nodes does not match binning");
    }
    const auto low = uniformBins.getRequired<double>("low");
    const auto high = uniformBins.getRequired<double>("high");
    bins_ = _UniformBins{n, low, high};
  } else {
    throw std::runtime_error ("Error when processing Binning: edges are neither an array nor a UniformBinning object");
  }

  variableIdx_ = find_input_index(json.getRequired<std::string_view>("input"), context.inputs());
  if ( context.inputs().at(variableIdx_).type() == Variable::VarType::string ) {
    throw std::runtime_error("Binning cannot use string inputs as binning variables");
  }
  Content default_value{0.};
  const auto& flowbehavior = json.getRequiredValue("flow");
  flow_ = parse_flowbehavior(flowbehavior);
  if (flow_ == _FlowBehavior::value) {
      default_value = resolve_content(flowbehavior, context);
  }

  // set bin contents
  contents_.reserve(content.Size() + 1);
  for (size_t i=0; i < content.Size(); ++i)
    contents_.push_back(resolve_content(content[i], context));
  contents_.push_back(std::move(default_value));
}

double Binning::evaluate(const std::vector<Variable::Type>& values) const
{
  std::size_t binIdx = find_bin_idx(values[variableIdx_], bins_, flow_, variableIdx_, "Binning");
  const Content& child = contents_[binIdx];
  return std::visit(node_evaluate{values}, child);
}

MultiBinning::MultiBinning(const JSONObject& json, const Correction& context)
{
  const auto& inputs = json.getRequired<rapidjson::Value::ConstArray>("inputs");

  const auto& edges = json.getRequired<rapidjson::Value::ConstArray>("edges");
  axes_.reserve(edges.Size());
  size_t idx {0};
  for (const auto& dimension : edges) {
    const auto& input = inputs[idx];
    if ( dimension.IsArray() ) { // non-uniform binning
      std::vector<double> dim_edges = parse_bin_edges(dimension.GetArray());
      if ( ! input.IsString() ) { throw std::runtime_error("invalid multibinning input type"); }
      size_t variableIdx = find_input_index(input.GetString(), context.inputs());
      if ( context.inputs().at(variableIdx).type() == Variable::VarType::string ) {
        throw std::runtime_error("MultiBinning cannot use string inputs as binning variables");
      }
      axes_.push_back({variableIdx, 0, _NonUniformBins(std::move(dim_edges))});
    } else if ( dimension.IsObject() ) { // UniformBinning
      const JSONObject uniformBins{dimension.GetObject()};
      const auto n = uniformBins.getRequired<uint32_t>("n");
      if ( n == 0 ) {
        auto msg = "Error when processing MultiBinning: number of bins for dimension " + std::to_string(idx) + " is zero";
        throw std::runtime_error(std::move(msg));
      }
      const auto low = uniformBins.getRequired<double>("low");
      const auto high = uniformBins.getRequired<double>("high");
      size_t variableIdx = find_input_index(input.GetString(), context.inputs());
      if ( context.inputs().at(variableIdx).type() == Variable::VarType::string ) {
        throw std::runtime_error("MultiBinning cannot use string inputs as binning variables");
      }
      axes_.push_back({variableIdx, 0, _UniformBins{n, low, high}});
    } else {
      auto msg = "Error when processing MultiBinning: edges for dimension " + std::to_string(idx) + " are neither an array nor a UniformBinning object";
      throw std::runtime_error (std::move(msg));
    }
    ++idx;
  }

  const auto& content = json.getRequired<rapidjson::Value::ConstArray>("content");
  size_t stride {1};
  --idx; // now corresponds to the last dimension
  for (auto it=axes_.rbegin(); it != axes_.rend(); ++it) {
    it->stride = stride;
    stride *= nbins(idx);
    --idx;
  }
  content_.reserve(content.Size() + 1); // + 1 for default value
  for (const auto& item : content) {
    content_.push_back(resolve_content(item, context));
  }
  if ( content_.size() != stride ) {
    throw std::runtime_error("Inconsistency in MultiBinning: number of content nodes does not match binning");
  }

  const auto& flowbehavior = json.getRequiredValue("flow");
  flow_ = parse_flowbehavior(flowbehavior);
  if (flow_ == _FlowBehavior::value) {
      content_.push_back(resolve_content(flowbehavior, context));
  }
}

double MultiBinning::evaluate(const std::vector<Variable::Type>& values) const
{
  size_t idx {0};
  size_t localidx {0};
  size_t dim {0};

  for (const auto& [variableIdx, stride, edgesVariant] : axes_) {
    localidx = find_bin_idx(values[variableIdx], edgesVariant, flow_, variableIdx, "MultiBinning");
    if ( localidx == nbins(dim) ) // find_bin_idx is indicating we need to return the default value
      return std::visit(node_evaluate{values}, content_.back());
    idx += localidx * stride;
    ++dim;
  }

  const Content& child = content_.at(idx);
  return std::visit(node_evaluate{values}, child);
}

size_t MultiBinning::nbins(size_t dimension) const
{
  if ( const auto *bins = std::get_if<_UniformBins>(&axes_[dimension].bins) )
    return bins->n; // using uniform bins

  // otherwise we must have non-uniform bins
  const auto &bins = std::get<_NonUniformBins>(axes_[dimension].bins);
  return bins.size() - 1;
}

Category::Category(const JSONObject& json, const Correction& context)
{
  variableIdx_ = find_input_index(json.getRequired<std::string_view>("input"), context.inputs());
  const auto& variable = context.inputs()[variableIdx_];
  if ( variable.type() == Variable::VarType::string ) {
    map_ = StrMap();
  } // (default-constructed as IntMap)
  for (const auto& kv_pair : json.getRequired<rapidjson::Value::ConstArray>("content"))
  {
    if ( ! (kv_pair.IsObject() && kv_pair.HasMember("key") && kv_pair.HasMember("value")) ) {
      throw std::runtime_error("Expected CategoryItem object");
    }
    if ( kv_pair["key"].IsString() ) {
      if ( variable.type() != Variable::VarType::string ) {
        throw std::runtime_error("Category got a key of type string, but its input is type " + variable.typeStr());
      }
      std::get<StrMap>(map_).try_emplace(kv_pair["key"].GetString(), resolve_content(kv_pair["value"], context));
    }
    else if ( kv_pair["key"].IsInt() ) {
      if ( variable.type() != Variable::VarType::integer ) {
        throw std::runtime_error("Category got a key of type int, but its input is type " + variable.typeStr());
      }
      std::get<IntMap>(map_).try_emplace(kv_pair["key"].GetInt(), resolve_content(kv_pair["value"], context));
    }
    else {
      throw std::runtime_error("Invalid key type in Category");
    }
  }

  const auto def = json.FindMember("default");
  if ( def != json.MemberEnd() && ! def->value.IsNull() ) {
    default_ = std::make_unique<Content>(resolve_content(def->value, context));
  }
}

double Category::evaluate(const std::vector<Variable::Type>& values) const {
  const Content* child = nullptr;
  if ( auto pval = std::get_if<std::string>(&values[variableIdx_]) ) {
    const auto& m = std::get<StrMap>(map_);
    auto it = m.find(*pval);
    if ( it != m.end() ) {
      child = &it->second;
    } else if ( default_ ) {
      child = default_.get();
    } else {
      throw std::out_of_range("Index not available in Category for input argument " + std::to_string(variableIdx_) + " val: " + *pval);
    }
  }
  else if ( auto pval = std::get_if<int>(&values[variableIdx_]) ) {
    const auto& m = std::get<IntMap>(map_);
    auto it = m.find(*pval);
    if ( it != m.end() ) {
      child = &it->second;
    } else if ( default_ ) {
      child = default_.get();
    } else {
      throw std::out_of_range("Index not available in Category for input argument " + std::to_string(variableIdx_) + " val: " + std::to_string(*pval));
    }
  } else {
    throw std::runtime_error("Invalid variable type");
  }

  return std::visit(node_evaluate{values}, *child);
}

Correction::Correction(const JSONObject& json) :
  name_(json.getRequired<const char *>("name")),
  description_(json.getOptional<const char*>("description").value_or("")),
  version_(json.getRequired<int>("version")),
  output_(json.getRequired<rapidjson::Value::ConstObject>("output"))
{
  if ( output_.type() != Variable::VarType::real ) { throw std::runtime_error("Outputs can only be real-valued"); }
  for (const auto& item : json.getRequired<rapidjson::Value::ConstArray>("inputs")) {
    if ( ! item.IsObject() ) { throw std::runtime_error("invalid input item type"); }
    inputs_.emplace_back(item.GetObject());
  }
  if ( const auto& items = json.getOptional<rapidjson::Value::ConstArray>("generic_formulas") ) {
    for (const auto& item : *items) {
      if ( ! item.IsObject() ) { throw std::runtime_error("invalid generic_formulas item type"); }
      formula_refs_.push_back(std::make_shared<Formula>(item.GetObject(), *this, true));
    }
  }

  data_ = resolve_content(json.getRequiredValue("data"), *this);
  initialized_ = true;
}

double Correction::evaluate(const std::vector<Variable::Type>& values) const {
  if ( ! initialized_ ) {
    throw std::logic_error("Not initialized");
  }
  if ( values.size() > inputs_.size() ) {
    throw std::runtime_error("Too many inputs");
  }
  else if ( values.size() < inputs_.size() ) {
    throw std::runtime_error("Insufficient inputs");
  }
  for (size_t i=0; i < inputs_.size(); ++i) {
    inputs_[i].validate(values[i]);
  }
  return std::visit(node_evaluate{values}, data_);
}

std::optional<std::vector<double>> Correction::match_formula(
    const std::vector<Variable::Type>& values, const FormulaAst& pattern,
    size_t parameter_count) const {
  if (values.size() != inputs_.size()) return std::nullopt;
  for (size_t i = 0; i < inputs_.size(); ++i) inputs_[i].validate(values[i]);
  const Content* node = &data_;
  while (const auto* category = std::get_if<Category>(node)) {
    const auto* branches = std::get_if<Category::StrMap>(&category->map_);
    const auto* key = std::get_if<std::string>(&values[category->variableIdx_]);
    if (!branches || !key) return std::nullopt;
    const auto it = branches->find(*key);
    node = it == branches->end() ? category->default_.get() : &it->second;
    if (!node) return std::nullopt;
  }
  const auto* formula = std::get_if<Formula>(node);
  if (!formula) return std::nullopt;

  using Type = FormulaAst::NodeType;
  std::vector<double> parameters(parameter_count);
  std::vector<bool> seen(parameter_count, false);
  const auto matches = [&](auto& self, const FormulaAst& actual, const FormulaAst& expected) -> bool {
    if (expected.nodetype() == Type::Parameter) {
      if (actual.nodetype() != Type::Literal) return false;
      const auto i = std::get<size_t>(expected.data());
      const auto value = std::get<double>(actual.data());
      if (i >= parameter_count || !std::isfinite(value)) return false;
      if (seen[i] && (parameters[i] != value || std::signbit(parameters[i]) != std::signbit(value)))
        return false;
      parameters[i] = value;
      seen[i] = true;
      return true;
    }
    if (expected.nodetype() != actual.nodetype() || expected.data() != actual.data() ||
        expected.children().size() != actual.children().size()) return false;
    if (expected.nodetype() == Type::Literal &&
        std::signbit(std::get<double>(expected.data())) != std::signbit(std::get<double>(actual.data())))
      return false;
    for (size_t i = 0; i < expected.children().size(); ++i)
      if (!self(self, actual.children()[i], expected.children()[i])) return false;
    return true;
  };
  if (!matches(matches, formula->ast(), pattern) || std::find(seen.begin(), seen.end(), false) != seen.end())
    return std::nullopt;
  return parameters;
}

struct BinnedVariationGroup::Plan {
  struct Node {
    const Binning* table = nullptr;
    std::size_t coordinate = 0;
    std::vector<Node> children;
    std::vector<double> values;
  };
  Correction::Ref correction;
  std::size_t dimensions = 0;
  std::size_t variations = 0;
  Node root;
};

std::unique_ptr<BinnedVariationGroup> BinnedVariationGroup::prepare(
    Correction::Ref correction, const std::vector<std::vector<std::string>>& stringCases) {
  if (!correction || stringCases.size() < 2)
    return nullptr;
  auto plan = std::make_shared<Plan>();
  plan->correction = std::move(correction);
  plan->variations = stringCases.size();
  std::vector<std::size_t> offsets;
  std::size_t strings = 0;
  for (const auto& input : plan->correction->inputs()) {
    if (input.type() == Variable::VarType::integer) return nullptr;
    offsets.push_back(input.type() == Variable::VarType::string ? strings++ : plan->dimensions++);
  }
  for (const auto& item : stringCases)
    if (item.size() != strings) return nullptr;

  const auto sameEdges = [](const Binning& a, const Binning& b) {
    if (a.bins_.index() != b.bins_.index()) return false;
    if (const auto* x = std::get_if<_UniformBins>(&a.bins_)) {
      const auto& y = std::get<_UniformBins>(b.bins_);
      return x->n == y.n && x->low == y.low && x->high == y.high;
    }
    return std::get<_NonUniformBins>(a.bins_) == std::get<_NonUniformBins>(b.bins_);
  };
  const auto build = [&](const auto& recurse, Plan::Node& node,
                         std::vector<const Content*> contents) -> bool {
    for (std::size_t variation = 0; variation < contents.size(); ++variation) {
      auto*& content = contents[variation];
      while (const auto* category = std::get_if<Category>(content)) {
        const auto* map = std::get_if<Category::StrMap>(&category->map_);
        if (!map) return false;
        const auto result = map->find(stringCases[variation][offsets[category->variableIdx_]]);
        content = result == map->end() ? category->default_.get() : &result->second;
        if (!content) return false;
      }
    }
    if (std::holds_alternative<double>(*contents.front())) {
      for (const auto* content : contents) {
        const auto* value = std::get_if<double>(content);
        if (!value) return false;
        node.values.push_back(*value);
      }
      return true;
    }
    const auto* first = std::get_if<Binning>(contents.front());
    if (!first || plan->correction->inputs()[first->variableIdx_].type() != Variable::VarType::real)
      return false;
    std::vector<const Binning*> tables;
    for (const auto* content : contents) {
      const auto* table = std::get_if<Binning>(content);
      if (!table || table->variableIdx_ != first->variableIdx_ ||
          table->flow_ != first->flow_ || table->contents_.size() != first->contents_.size() ||
          !sameEdges(*first, *table))
        return false;
      tables.push_back(table);
    }
    node.table = first;
    node.coordinate = offsets[first->variableIdx_];
    node.children.resize(first->contents_.size());
    for (std::size_t bin = 0; bin < node.children.size(); ++bin) {
      for (std::size_t variation = 0; variation < tables.size(); ++variation)
        contents[variation] = &tables[variation]->contents_[bin];
      if (!recurse(recurse, node.children[bin], contents)) return false;
    }
    return true;
  };
  if (!build(build, plan->root, std::vector<const Content*>(stringCases.size(), &plan->correction->data_)) ||
      !plan->root.table)
    return nullptr;
  return std::unique_ptr<BinnedVariationGroup>(new BinnedVariationGroup(std::move(plan)));
}

double BinnedVariationGroup::evaluate(std::size_t variation, const double* coordinates,
                                      std::size_t size, Workspace& workspace) const {
  if (size != plan_->dimensions || variation >= plan_->variations)
    throw std::invalid_argument("Binned variation lookup input count mismatch");
  if (workspace.plan_ != plan_ ||
      !std::equal(coordinates, coordinates + size, workspace.coordinates_.begin())) {
    for (std::size_t i = 0; i < size; ++i)
      if (!std::isfinite(coordinates[i]))
        throw std::runtime_error("Non-finite binned variation coordinate");
    auto* node = &plan_->root;
    while (const auto* table = node->table) {
      const auto bin = find_bin_idx(coordinates[node->coordinate], table->bins_, table->flow_,
                                    table->variableIdx_, "Binning");
      node = &node->children[bin];
    }
    workspace.coordinates_.assign(coordinates, coordinates + size);
    workspace.plan_ = plan_;
    workspace.leaf_ = &node->values;
  }
  return (*workspace.leaf_)[variation];
}

std::unique_ptr<BinnedFormulaGroup> BinnedFormulaGroup::prepare(
    const std::vector<Correction::Ref>& sources) {
  if (sources.size() < 2 || !sources.front()) return nullptr;
  auto group = std::unique_ptr<BinnedFormulaGroup>(new BinnedFormulaGroup);
  std::vector<const Binning*> tables;
  for (const auto& source : sources) {
    if (!source || source->inputs().size() != 2) return nullptr;
    for (size_t i = 0; i < 2; ++i) {
      if (source->inputs()[i].type() != Variable::VarType::real ||
          source->inputs()[i].name() != sources.front()->inputs()[i].name()) return nullptr;
    }
    const auto* outer = std::get_if<Binning>(&source->data_);
    if (!outer || outer->variableIdx_ != 0 || outer->flow_ != _FlowBehavior::value)
      return nullptr;
    const auto* edges = std::get_if<_NonUniformBins>(&outer->bins_);
    const auto* fallback = std::get_if<double>(&outer->contents_.back());
    if (!fallback || !edges || edges->size() < 2) return nullptr;
    if (tables.empty()) group->outerEdges_ = *edges;
    if (*edges != group->outerEdges_) return nullptr;
    group->defaults_.push_back(*fallback);
    tables.push_back(outer);
  }

  std::vector<const Binning*> rows(tables.size());
  for (size_t i = 0; i + 1 < group->outerEdges_.size(); ++i) {
    for (size_t k = 0; k < tables.size(); ++k) {
      const auto* row = std::get_if<Binning>(&tables[k]->contents_[i]);
      if (!row || row->variableIdx_ != 1 || row->flow_ != _FlowBehavior::clamp)
        return nullptr;
      const auto* edges = std::get_if<_NonUniformBins>(&row->bins_);
      if (!edges || edges->size() < 2) return nullptr;
      if (group->innerEdges_.empty()) group->innerEdges_ = *edges;
      if (*edges != group->innerEdges_) return nullptr;
      rows[k] = row;
    }
    for (size_t j = 0; j + 1 < group->innerEdges_.size(); ++j) {
      for (const auto* row : rows) {
        const auto* leaf = std::get_if<FormulaRef>(&row->contents_[j]);
        if (!leaf || leaf->formula_->affineVariable_ != 1 || leaf->parameters_.size() != 4)
          return nullptr;
        std::array<double, 4> p;
        std::copy_n(leaf->parameters_.begin(), p.size(), p.begin());
        group->coefficients_.push_back(p);
      }
    }
  }
  return group;
}

void BinnedFormulaGroup::evaluate(double first, double second, double* outputs, size_t capacity) const {
  if (!outputs || capacity < size()) throw std::invalid_argument("Insufficient grouped output storage");
  const auto outer = std::upper_bound(outerEdges_.begin(), outerEdges_.end(), first);
  if (outer == outerEdges_.begin() || outer == outerEdges_.end()) {
    std::copy(defaults_.begin(), defaults_.end(), outputs);
    return;
  }
  const size_t i = outer - outerEdges_.begin() - 1;
  const auto inner = std::upper_bound(innerEdges_.begin(), innerEdges_.end(), second);
  const auto bounded = std::clamp(inner, innerEdges_.begin() + 1, innerEdges_.end() - 1);
  const size_t j = bounded - innerEdges_.begin() - 1;
  const auto* row = coefficients_.data() + (i * (innerEdges_.size() - 1) + j) * size();
  for (size_t k = 0; k < size(); ++k) outputs[k] = clampedAffine(second, row[k].data());
}

CompoundCorrection::CompoundCorrection(const JSONObject& json, const CorrectionSet& context) :
  name_(json.getRequired<const char *>("name")),
  description_(json.getOptional<const char*>("description").value_or("")),
  output_(json.getRequired<rapidjson::Value::ConstObject>("output"))
{
  if ( output_.type() != Variable::VarType::real ) { throw std::runtime_error("Outputs can only be real-valued"); }
  for (const auto& item : json.getRequired<rapidjson::Value::ConstArray>("inputs")) {
    if ( ! item.IsObject() ) { throw std::runtime_error("invalid inputs item type"); }
    inputs_.emplace_back(item.GetObject());
  }
  for (const auto& item : json.getRequired<rapidjson::Value::ConstArray>("inputs_update")) {
    if ( ! item.IsString() ) { throw std::runtime_error("invalid inputs_update item type"); }
    size_t idx = input_index(item.GetString());
    if ( inputs_[idx].type() != Variable::VarType::real ) {
      throw std::runtime_error("CompoundCorrection updatable inputs must be real-valued");
    }
    inputs_update_.emplace_back(idx);
  }

  const auto& op = json.getRequired<std::string_view>("input_op");
  if ( op == "+" ) { input_op_ = UpdateOp::Add; }
  else if ( op == "*" ) { input_op_ = UpdateOp::Multiply; }
  else if ( op == "/" ) { input_op_ = UpdateOp::Divide; }
  else { throw std::runtime_error("Invalid CompoundCorrection input update op"); }

  const auto& out_op = json.getRequired<std::string_view>("output_op");
  if ( out_op == "+" ) { output_op_ = UpdateOp::Add; }
  else if ( out_op == "*" ) { output_op_ = UpdateOp::Multiply; }
  else if ( out_op == "/" ) { output_op_ = UpdateOp::Divide; }
  else if ( out_op == "last" ) { output_op_ = UpdateOp::Last; }
  else { throw std::runtime_error("Invalid CompoundCorrection output update op"); }

  for (const auto& item : json.getRequired<rapidjson::Value::ConstArray>("stack")) {
    if ( ! item.IsString() ) { throw std::runtime_error("Invalid type"); }
    Correction::Ref corr;
    try {
      corr = context.at(item.GetString());
    } catch (std::out_of_range& ex) {
      throw std::runtime_error("CompoundCorrection constituent "
          + std::string(item.GetString()) + " not found in the CorrectionSet");
    }
    std::vector<size_t> inmap;
    for (const auto& input : corr->inputs()) {
      inmap.push_back(input_index(input.name()));
    }
    stack_.emplace_back(std::move(inmap), corr);
  }
}

size_t CompoundCorrection::input_index(const std::string_view name) const {
  size_t idx = 0;
  for (const auto& var : inputs_) {
    if ( name == var.name() ) return idx;
    idx++;
  }
  throw std::runtime_error("Error: could not find variable " + std::string(name) + " in inputs");
}

double CompoundCorrection::evaluate(const std::vector<Variable::Type>& values) const {
  static thread_local Workspace workspace;
  return evaluate_with_workspace(values, workspace);
}

double CompoundCorrection::evaluate_with_workspace(const std::vector<Variable::Type>& values,
                                                   Workspace& workspace) const {
  if ( values.size() > inputs_.size() ) {
    throw std::runtime_error("Too many inputs");
  }
  else if ( values.size() < inputs_.size() ) {
    throw std::runtime_error("Insufficient inputs");
  }
  for (size_t i=0; i < inputs_.size(); ++i) {
    inputs_[i].validate(values[i]);
  }
  auto& ivalues = workspace.inputs;
  auto& cvalues = workspace.component;
  ivalues = values;
  cvalues.reserve(values.size());
  double out = 0.;
  double sf = 0.;
  bool start{true};
  for(const auto& [inmap, corr] : stack_) {
    cvalues.clear();
    for(size_t pos : inmap) cvalues.push_back(ivalues[pos]);
    sf = corr->evaluate(cvalues);
    for(size_t pos : inputs_update_) {
      switch ( input_op_ ) {
        case UpdateOp::Add: std::get<double>(ivalues[pos]) += sf; break;
        case UpdateOp::Multiply: std::get<double>(ivalues[pos]) *= sf; break;
        case UpdateOp::Divide: std::get<double>(ivalues[pos]) /= sf; break;
        case UpdateOp::Last: throw std::logic_error("Illegal update op");
      }
    }
    if ( start ) { out = sf; start = false; }
    else {
      switch ( output_op_ ) {
        case UpdateOp::Add: out += sf; break;
        case UpdateOp::Multiply: out *= sf; break;
        case UpdateOp::Divide: out /= sf; break;
        case UpdateOp::Last: out = sf;
      }
    }
  }
  return out;
}

std::unique_ptr<CorrectionSet> CorrectionSet::from_file(const std::string& fn) {
  rapidjson::Document json;
  FILE* fp = fopen(fn.c_str(), "rb");
  if ( fp == nullptr ) {
    throw std::runtime_error("Failed to open file: " + fn);
  }
  constexpr unsigned char magicref[2] = {0x1f, 0x8b};
  unsigned char magic[2];
  if (fread(magic, sizeof *magic, 2, fp) != 2) {
    throw std::runtime_error("Failed to read file magic: " + fn);
  }
  rewind(fp);
  char readBuffer[65536];
  rapidjson::ParseResult ok;
  if (memcmp(magic, magicref, sizeof(magic)) == 0) {
    fclose(fp);
#ifdef WITH_ZLIB
    gzFile_s* fpz = gzopen(fn.c_str(), "r");
    rapidjson::GzFileReadStream is(fpz, readBuffer, sizeof(readBuffer));
    ok = json.ParseStream<rapidjson::kParseNanAndInfFlag>(is);
    gzclose(fpz);
#else
    throw std::runtime_error("Gzip-compressed JSON files are only supported if ZLIB is found when the package is built");
#endif
  } else {
    rapidjson::FileReadStream is(fp, readBuffer, sizeof(readBuffer));
    ok = json.ParseStream<rapidjson::kParseNanAndInfFlag>(is);
    fclose(fp);
  }
  if (!ok) {
    throw std::runtime_error(
        std::string("JSON parse error: ") + rapidjson::GetParseError_En(ok.Code())
        + " at offset " + std::to_string(ok.Offset())
        );
  }
  if ( ! json.IsObject() ) { throw std::runtime_error("Expected CorrectionSet object"); }
  return std::make_unique<CorrectionSet>(json);
}

std::unique_ptr<CorrectionSet> CorrectionSet::from_string(const char * data) {
  rapidjson::Document json;
  rapidjson::ParseResult ok = json.Parse<rapidjson::kParseNanAndInfFlag>(data);
  if (!ok) {
    throw std::runtime_error(
        std::string("JSON parse error: ") + rapidjson::GetParseError_En(ok.Code())
        + " at offset " + std::to_string(ok.Offset())
        );
  }
  if ( ! json.IsObject() ) { throw std::runtime_error("Expected CorrectionSet object"); }
  return std::make_unique<CorrectionSet>(json);
}

CorrectionSet::CorrectionSet(const JSONObject& json) {
  schema_version_ = json.getRequired<int>("schema_version");
  if ( schema_version_ > evaluator_version ) {
    throw std::runtime_error("Evaluator is designed for schema v" + std::to_string(evaluator_version) + " and is not forward-compatible");
  }
  else if ( schema_version_ < evaluator_version ) {
    throw std::runtime_error("Evaluator is designed for schema v" + std::to_string(evaluator_version) + " and is not backward-compatible");
  }
  description_ = json.getOptional<const char*>("description").value_or("");
  for (const auto& item : json.getRequired<rapidjson::Value::ConstArray>("corrections")) {
    if ( ! item.IsObject() ) { throw std::runtime_error("Expected Correction object"); }
    auto corr = std::make_shared<Correction>(item.GetObject());
    if ( corrections_.find(corr->name()) != corrections_.end() ) {
      throw std::runtime_error("Duplicate Correction name: " + corr->name());
    }
    corrections_[corr->name()] = corr;
  }
  if (auto items = json.getOptional<rapidjson::Value::ConstArray>("compound_corrections")) {
    for (const auto& item : *items) {
      if ( ! item.IsObject() ) { throw std::runtime_error("Expected CompoundCorrection object"); }
      auto corr = std::make_shared<CompoundCorrection>(item.GetObject(), *this);
      if ( compoundcorrections_.find(corr->name()) != compoundcorrections_.end() ) {
        throw std::runtime_error("Duplicate CompoundCorrection name: " + corr->name());
      }
      compoundcorrections_[corr->name()] = corr;
    }
  }
}

bool CorrectionSet::validate() {
  // TODO: validate with https://rapidjson.org/md_doc_schema.html
  return true;
}
