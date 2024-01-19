#include "tensor.h"
#include "grad.h"
#include <type_traits>

namespace grad{
using std::make_shared, std::shared_ptr, std::vector;
using std::initializer_list;
using std::ostream, std::cin, std::cout, std::endl;

Node operator+(Node &node1, Node &node2){
    // cout << "Test operator+" << endl;
    return AddNode(node1, node2);
}

ostream &operator<<(ostream &os, Node &node){
    os << node.getTensor();
    return os;
}


// Out-class implements



/*
    Node implement
*/
void Node::eval(){}

Node::Node(const grad::Node &that){
    value = that.value;
    grad = that.grad;
    parents = that.parents;
    children = that.children;
}

Node::Node(ts::Tensor &ts){
    value = make_shared<ts::Tensor>(ts);
}

Node::Node(vector<shared_ptr<grad::Node>> &pars){
    parents = pars;
}

grad::Node Node::gradTo(Node &that) throw(){
    that.eval();
    Node result;
    result.value = make_shared<ts::Tensor>(ts::zeros_like(*that.value));
    return result;
}

/*
 * Variable implement
 */

Variable::Variable(ts::Tensor &ts){
    value = make_shared<ts::Tensor>(ts);
}

grad::Node Variable::gradTo(Node &that) throw(){
    if(&that != this){
        ts::Tensor tensor = ts::zeros_like(*that.value_ptr());
        Node result{tensor};
        return result;
    }
    if(&that == this){
        Node result{*this->value};
        return result;
    }
}

/*
    AddNode implement
*/

AddNode::AddNode(shared_ptr<Node> node1, shared_ptr<Node> node2){
    parents.push_back(node1);
    parents.push_back(node2);
}

AddNode::AddNode(Node &node1, Node &node2){
    // std::cout << "Test AddNode constructor." << std::endl;
    parents.push_back(make_shared<Node>(node1));
    parents.push_back(make_shared<Node>(node2));
}

void AddNode::eval() throw(){
    // cout << "test AddNode::eval" << endl;
    if(value == nullptr){
        for(shared_ptr<Node> node: parents){
            node->eval();
        }
        // The parents node must be size 2.
        ts::Tensor &ts1 = parents[0]->getTensor();
        ts::Tensor &ts2 = parents[1]->getTensor();
//        cout << "test" << endl;
//        cout << ts1 << endl;
//        cout << ts2 << endl;
        value = make_shared<ts::Tensor>(ts::add(ts1, ts2));
    }
}

grad::Node AddNode::gradTo(Node &that) throw(){
    grad::Node lhs = parents[0]->gradTo(that);
    grad::Node rhs = parents[1]->gradTo(that);
    grad::AddNode result{lhs, rhs};
    result.eval();
    return result;
}

/*
    SubNode implement
*/

SubNode::SubNode(shared_ptr<Node> node1, shared_ptr<Node> node2){
    parents.push_back(node1);
    parents.push_back(node2);
}

SubNode::SubNode(Node &node1, Node &node2){
    // std::cout << "Test AddNode constructor." << std::endl;
    parents.push_back(make_shared<Node>(node1));
    parents.push_back(make_shared<Node>(node2));
}

void SubNode::eval() throw(){
    // cout << "test AddNode::eval" << endl;
    if(value == nullptr){
        for(shared_ptr<Node> node: parents){
            node->eval();
        }
        // The parents node must be size 2.
        ts::Tensor &ts1 = parents[0]->getTensor();
        ts::Tensor &ts2 = parents[1]->getTensor();
        value = make_shared<ts::Tensor>(ts::sub(ts1, ts2));
    }
}

grad::Node SubNode::gradTo(Node &that) throw(){
    grad::Node lhs = parents[0]->gradTo(that);
    grad::Node rhs = parents[1]->gradTo(that);
    grad::SubNode result{lhs, rhs};
    result.eval();
    return result;
}

/*
 * Sin
 */

SinNode::SinNode(Node &node){
    value = make_shared<ts::Tensor>(ts::Sin(*node.value_ptr()));
}

SinNode::SinNode(ts::Tensor &tensor){
    value = make_shared<ts::Tensor>(ts::Sin(tensor));
}

void SinNode::eval() throw(){
    if(value == nullptr){
        parents[0]->eval();
        // The parents node must be size 2.
        ts::Tensor &tensor = parents[0]->getTensor();
        value = make_shared<ts::Tensor>(ts::Sin(tensor));
    }
}

grad::Node SinNode::gradTo(grad::Node &that) throw(){
//    return ts::Cos(*this->value) * this->parents[0]->gradTo(that);
}


/*
 * Autograd implement
 */

grad::Node autograd(ts::Tensor &input, ts::Tensor &output) throw(){
    if(input.node_ptr() == nullptr)
        input.init_node();
    if(output.node_ptr() == nullptr)
        output.init_node();
    return output.node().gradTo(input.node());
}

// Autograd.






};