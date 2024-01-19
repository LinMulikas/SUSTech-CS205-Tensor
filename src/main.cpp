#include <iostream>
#include "tensor.h"
#include "grad.h"

using ts::Tensor, ts::zeros, ts::rand;

using ts::VariantData;

using grad::Node, grad::Variable, grad::AddNode;

using std::cout, std::endl;


int main(){
    // ? Autograd test

//    /*
//        AddNode node test
//    */
//
//    int shape[1] = {2};
//
//
//    Tensor ts1 = rand<int>(shape);
//    Tensor ts2 = rand<double>(shape);
//    ts1.require_grad();
//    cout << ts1 << endl;
//    cout << ts2 << endl;
//
//    Tensor ts3 = ts1 - ts2;
//    auto grad1 = grad::autograd(ts2, ts3);
//
////    auto grad2 = grad::autograd(ts1, ts4);
//    cout << grad1 << endl;

    int shape[3]{2, 3, 4};
    Tensor ts = rand<int>(shape, 3);
    cout << ts << endl;
    ts.test_mean();
    return 0;
}
