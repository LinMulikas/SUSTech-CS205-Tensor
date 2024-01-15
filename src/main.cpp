#include <iostream>
#include "tensor.h"
#include "grad.h"

using ts::Tensor, ts::zeros, ts::rand;

using ts::VariantData;

using grad::Node, grad::Variable, grad::AddNode;

using std::cout, std::endl;

int main(){
    // ? Autograd test

    /*
        AddNode node test
    */

    int shape[2] = {3, 2};

    Tensor ts1 = rand<int>(shape);
    Tensor ts2 = rand<double>(shape);
    ts1.require_grad();
    Tensor ts3 = ts1 - ts2;
    cout << ts1 << endl;
    cout << ts2 << endl;

    auto grad1 = grad::autograd(ts2, ts3);
    cout << grad1 << endl;

//    int shape[1]{10};
//    Tensor ts1 = rand<int>(shape);
//    cout << ts1 << endl;
//    cout << ts::Exp(ts1) << endl;

    return 0;
}
