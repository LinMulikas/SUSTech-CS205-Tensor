#include <iostream>
#include "tensor.h"
#include "grad.h"

using ts::Tensor, ts::zeros, ts::rand;

using ts::VariantData;

// using grad::Variable, grad::Add;

// using grad::Variable, grad::Add;

using std::cout, std::endl;

int main(){
    // ? Autograd test

    /*
        Add test
    */

    int shape[3] = {2, 2,2};
    

    Tensor ts1 = rand<int>(shape);
    Tensor ts2 = rand<double>(shape);
    Tensor ts3 = ts1 + ts2;
    cout << ts1 << endl;
    cout << ts2 << endl;

    cout<<ts1.sum(1)<<endl;
    

    return 0;
}
