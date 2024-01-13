#include "tensor.h"
#include "grad.h"
#include <type_traits>

namespace grad{
using std::make_shared, std::shared_ptr, std::vector;
using std::initializer_list;
using std::ostream, std::cin, std::cout, std::endl;

Node operator+(Node &node1, Node &node2){
    // cout << "Test operator+" << endl;
    return Add(node1, node2);
}

ostream &operator<<(ostream &os, Node &node){
    os << node.getTensor();
    return os;
}


// Out-class implements

/*
    Node
*/
void Node::eval(){}


/*
    Add
*/

Add::Add(shared_ptr<Node> node1, shared_ptr<Node> node2){
    parents.push_back(node1);
    parents.push_back(node2);
}

Add::Add(Node &node1, Node &node2){
    // std::cout << "Test Add constructor." << std::endl;
    parents.push_back(make_shared<Node>(node1));
    parents.push_back(make_shared<Node>(node2));
}

// Add::eval
void Add::eval() throw(){
    // cout << "test Add::eval" << endl;
    if(value == nullptr){
        for(shared_ptr<Node> node: parents){
            node->eval();
        }
        // The parents node must be size 2.
        ts::Tensor &ts1 = parents[0]->getTensor();
        ts::Tensor &ts2 = parents[1]->getTensor();
        value = make_shared<ts::Tensor>(ts1 + ts2);
    }
}

// Autograd
ts::Tensor autograd(ts::Tensor &input, ts::Tensor &output) throw(){
    if(input.node_ptr() == nullptr)
        input.init_node();
    if(output.node_ptr() == nullptr)
        output.init_node();
    return output.node().gradTo(input.node());
}

// Autograd.


ts::Tensor Node::gradTo(Node &that) throw(){
    return ts::zeros_like(*that.value_ptr());
}

ts::Tensor Variable::gradTo(Node &that) throw(){
    if(&that != this) return ts::zeros_like(*that.value_ptr());
    if(&that == this) return *value;
}

ts::Tensor Add::gradTo(Node &that) throw(){
    ts::Tensor lhs = parents[0]->gradTo(that);
    ts::Tensor rhs = parents[1]->gradTo(that);
    return ts::add(lhs, rhs);
}

};