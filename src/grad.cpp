#include "tensor.h"
#include "grad.h"

namespace grad{
using std::make_shared, std::shared_ptr, std::vector;
using std::initializer_list;
using std::ostream, std::cin, std::cout, std::endl;

Node operator+(Node &node1, Node &node2){
    // cout << "Test operator+" << endl;
    return Add_node(node1, node2);
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
    Add_node
*/

Add_node::Add_node(shared_ptr<Node> node1, shared_ptr<Node> node2){
    parents.push_back(node1);
    parents.push_back(node2);
}

Add_node::Add_node(Node &node1, Node &node2){
    // std::cout << "Test Add_node constructor." << std::endl;
    parents.push_back(make_shared<Node>(node1));
    parents.push_back(make_shared<Node>(node2));
}

// Add_node::eval
void Add_node::eval() throw(){
    // cout << "test Add_node::eval" << endl;
    if(value == nullptr){
        for(shared_ptr<Node> node : parents){
            node->eval();
        }
        // The parents node must be size 2.
        ts::Tensor &ts1 = parents[0]->getTensor();
        ts::Tensor &ts2 = parents[1]->getTensor();
        value = make_shared<ts::Tensor>(ts1 + ts2);
    }
}

// Autograd

ts::Tensor &autograd(grad::Node &x, grad::Node &y) throw(){
    if(typeid(x) == typeid(Node) || typeid(y) == typeid(Node)){
        throw std::invalid_argument("The input, output need to be Variable.");
    }
    else{
        if(typeid(x) == typeid(Variable)
           && typeid(y) == typeid(Variable)){
            if(&x != &y){
                throw std::invalid_argument("The input, output has no relationship.");
            }
            else{
                // TODO: cat two variable and return.
            }
        }
    }

}
// Autograd.


ts::Tensor &Node::gradTo(Node &that) throw(){}

ts::Tensor &Variable::gradTo(Node &that) throw(){
    if(&that != this) throw std::invalid_argument("The input, output has no releationship.");
    if(&that == this) return *value;
}

ts::Tensor &Add_node::gradTo(Node &that) throw(){
    if(&(*parents[0]) == &that || &(*parents[1]) == &that){
    }
}

};