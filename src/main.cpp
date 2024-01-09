#include <iostream>
#include "tensor.h"
#include "autograd.h"

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
    cout << "ts2(rand):" << endl;
    cout << ts2 << endl;

    Tensor add_1_2 = ts1 + ts2;
    cout << "add_1_2" << endl;
    cout << add_1_2 << endl;

    /*
        Variable test
    */

    Variable var1{ts1};
    Variable var2{ts2};
    cout << "var1.value" << endl;
    cout << var1 << endl;

    Add_node added_node{var1 + var2};
    cout << "added_ Node: " << endl;
    cout << added_node << endl;

    return 0;
}
