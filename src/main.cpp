#include <iostream>
#include "tensor.h"
#include "autograd.h"

using ts::Tensor, ts::zeros, ts::rand;

using ts::VariantData;

using std::cout, std::endl;

int main(){
    // ? Autograd test

    /*
        Variable Test
    */

    int shape[2] = {3, 2};

    ts::global_require_grad = true;

    Tensor ts1 = rand<int>(shape);
    cout << "ts1(rand):" << endl;
    cout << ts1 << endl;


    Tensor ts2 = rand<double>(shape);
    cout << "ts2(rand):" << endl;
    cout << ts2 << endl;

    Tensor add_1_2 = ts1 + ts2;
    cout << add_1_2 << endl;

    return 0;
}
