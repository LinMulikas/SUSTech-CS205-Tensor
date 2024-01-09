#include "tensor.h"
#include "autograd.h"

namespace grad{
using std::make_shared, std::shared_ptr, std::vector;
using std::initializer_list;
using std::ostream, std::cin, std::cout, std::endl;

Node operator+(Node &node1, Node &node2){
    // cout << "Test operator+" << endl;
    return Add_node(node1, node2);
}

ostream &operator<<(ostream &os, Node &node){
    os << node.getVal();
    return os;
}


/*
    Node
*/
void Node::eval(){}

/*
    Add_node
*/

Add_node::Add_node(Node &node1, Node &node2) throw(){
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
        ts::Tensor &ts1 = parents[0]->getVal();
        ts::Tensor &ts2 = parents[1]->getVal();
        value = make_shared<ts::Tensor>(ts1 + ts2);
    }
}

};