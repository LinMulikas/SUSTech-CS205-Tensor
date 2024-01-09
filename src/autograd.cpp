#include "tensor.h"
#include "autograd.h"

namespace grad{
using std::make_shared, std::shared_ptr, std::vector;
using std::initializer_list;
using std::ostream, std::cin, std::cout, std::endl;


ostream &operator<<(ostream &os, Node &node){
    os << node.getVal();
    return os;
}




// Add_node
void Add_node::eval(){
    if(value == nullptr){
        for(shared_ptr<Node> node : parents){
            node->eval();
        }
        // TODO: Broadcast?
        // The parents node must be size 2.
        ts::Tensor &ts1 = parents[0]->getVal();
        ts::Tensor &ts2 = parents[1]->getVal();
        cout << ts1 << endl;
        cout << ts2 << endl;

    }
}

};