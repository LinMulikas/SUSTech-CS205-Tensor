#include "tensor.h"
#include "grad.h"

#include <utility>

namespace grad{
using std::make_shared, std::shared_ptr, std::vector;
using std::initializer_list;
using std::ostream, std::cin, std::cout, std::endl;

Node::Node(ts::Tensor &v, ts::Tensor &g){
    value = make_shared<ts::Tensor>(v);
    grad = make_shared<ts::Tensor>(g);
}

ValueNode::ValueNode(){
    this->value = nullptr;
//    cout << *this->value << endl;
    this->grad = nullptr;
}

ValueNode::ValueNode(ts::Tensor &ts){
    this->value = make_shared<ts::Tensor>(ts);
//    cout << *this->value << endl;
    this->grad = make_shared<ts::Tensor>(ts::zeros_like(ts));
}

void AddNode::forward(){
    this->value = make_shared<ts::Tensor>(ts::add_no_grad((*lhs->value), (*rhs->value)));
}

void AddNode::backward(){
//    *grad = ts::ones_like(*lhs->value);
    grad = make_shared<ts::Tensor>(ts::ones_like(*lhs->value));
    *lhs->grad = (*lhs->grad) + (*grad);
    *rhs->grad = (*rhs->grad) + (*grad);
    lhs->backward();
    rhs->backward();
}

void SubNode::forward(){
    this->value = make_shared<ts::Tensor>(ts::add_no_grad((*lhs->value), (*rhs->value)));
}

void SubNode::backward(){
//    *grad = ts::ones_like(*lhs->value);
    grad = make_shared<ts::Tensor>(ts::ones_like(*lhs->value));
    *lhs->grad = (*lhs->grad) + (*grad);
    *rhs->grad = (*rhs->grad) - (*grad);
    lhs->backward();
    rhs->backward();
}

void MulNode::forward(){
    this->value = make_shared<ts::Tensor>(ts::mul_pt_no_grad((*lhs->value), (*rhs->value)));
}

void MulNode::backward(){
//    *grad = ts::ones_like(*lhs->value);
    grad = make_shared<ts::Tensor>(ts::ones_like(*lhs->value));
    *lhs->grad = ts::mul_pt_no_grad((*rhs->value), (*grad));
//    cout << "Value: " << (*rhs->value) << endl;
    *rhs->grad = ts::mul_pt_no_grad((*lhs->value), (*grad));
    lhs->backward();
    rhs->backward();
}

void DivNode::forward(){
    this->value = make_shared<ts::Tensor>(ts::div_pt_no_grad((*lhs->value), (*rhs->value)));
}

void DivNode::backward(){
//    *grad = ts::ones_like(*lhs->value);
    grad = make_shared<ts::Tensor>(ts::ones_like(*lhs->value));
    *lhs->grad = ts::div_pt_no_grad((*grad), (*rhs->value));
    *rhs->grad = ts::sub_no_grad(
            (*rhs->grad),
            ts::mul_pt_no_grad(
                    *lhs->value,
                    ts::mul_pt_no_grad(
                            (*rhs->value).inv_pt(),
                            (*rhs->value).inv_pt())));
    lhs->backward();
    rhs->backward();
}

};