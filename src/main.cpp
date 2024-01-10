#include <iostream>
#include "tensor.h"
#include "grad.h"

using ts::Tensor, ts::zeros, ts::rand;

using ts::VariantData;

using grad::Variable, grad::Add_node;

using std::cout, std::endl;

int main(){
    // ? Autograd test

    /*
        Add test
    */

    int shape[2] = {3, 2};

    ts::global_require_grad = true;

    Tensor ts1 = rand<int>(shape);
    cout << "ts1(rand):" << endl;
    cout << ts1 << endl;

    Tensor ts2 = rand<double>(shape);
    cout << ts2 << endl;
    Tensor added = ts1 + ts2;
    cout << *added.get_node_ptr() << endl;

    auto added_node = dynamic_cast<grad::Add_node &>(*added.get_node_ptr());
    cout << (typeid(added_node) == typeid(grad::Add_node)) << endl;


    return 0;
}
