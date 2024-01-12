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

    return 0;
}
